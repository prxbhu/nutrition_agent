from asyncio import events
from google.adk.agents import Agent, SequentialAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner
from google.adk.tools import FunctionTool, google_search
from google.adk.plugins.logging_plugin import LoggingPlugin
from google.adk.sessions import InMemorySessionService
from google.genai import types
from dotenv import load_dotenv
import os
import json
import asyncio
import traceback
from typing import Dict, Any
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters

load_dotenv()

APP_NAME="nutrition_agent"
USER_ID="user1234"
SESSION_ID="1234"
MODEL_ID="gemini-2.0-flash"

retry_config=types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504], # Retry on these HTTP errors
)



# MCP Toolset for nutrition data
nutrition_server = McpToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command="/home/prxbhu/.nvm/versions/node/v22.17.0/bin/node",
            args=["/home/prxbhu/Documents/mcp-opennutrition/build/index.js"],
            tool_filter=["search-food-by-name"]
        ),
        timeout=30,
    )
)


#sample data
def analyze_health_metrics() -> str:
    """
    Fetch the questionnaire and measurements data of the patient.
    
    Returns:
        A JSON string containing questionnaire responses and measurements data
    """
    try:
        questionnaire_path = "/home/prxbhu/Documents/nutritionist-agent/quest.json"
        measurements_path = "/home/prxbhu/Documents/nutritionist-agent/measurements.json"
        
        with open(questionnaire_path, 'r') as q_file:
            questionnaire = json.load(q_file)
        
        with open(measurements_path, 'r') as m_file:
            measurements = json.load(m_file)
        
        analysis = {
            "questionnaire": questionnaire,
            "measurements": measurements
        }
        
        return json.dumps(analysis, indent=2)
    except FileNotFoundError as e:
        return json.dumps({"error": f"File not found: {str(e)}"})
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"Invalid JSON format: {str(e)}"})



# Agent 1: Patient Data Retrieval and Analysis
patient_data_agent = Agent(
    model=Gemini(model="gemini-2.0-flash-exp", retry_options=retry_config),
    name="patient_data_agent",
    instruction="""You are a patient data retrieval and analysis specialist. Your role is to:

    1. USE the analyze_health_metrics tool to fetch patient data
    2. Thoroughly analyze the questionnaire responses:
    - Extract age, gender, height, weight, BMI
    - Identify dietary preference (vegetarian/non-vegetarian)
    - Note medical conditions (diabetes, hypertension, cholesterol issues, etc.)
    - Document genetic conditions
    - Review lifestyle factors (exercise frequency, water intake, alcohol consumption, smoking)
    - Identify meal preparation preferences and eating habits
    - Note any food allergies or restrictions

    3. Analyze blood test measurements:
    - Compare each value against reference ranges
    - Flag any abnormal values (high or low)
    - Identify nutritional deficiencies or concerns
    - Note any critical health markers

    4. Create a comprehensive health summary that includes:
    - Patient demographics and anthropometrics (age, gender, height, weight, BMI)
    - Complete dietary preferences and restrictions
    - All medical and genetic conditions
    - Lifestyle assessment
    - Blood parameter analysis with abnormalities highlighted
    - Key nutritional considerations based on health status

    Format your output clearly with sections:
    **PATIENT PROFILE:**
    - Demographics
    - Anthropometrics

    **DIETARY PREFERENCES:**
    - Type (veg/non-veg)
    - Restrictions
    - Meal preparation preferences

    **MEDICAL CONDITIONS:**
    - List all conditions

    **LIFESTYLE FACTORS:**
    - Exercise, water intake, habits

    **BLOOD TEST ANALYSIS:**
    - Normal values
    - Abnormal values (flagged)
    - Nutritional concerns

    **KEY NUTRITIONAL CONSIDERATIONS:**
    - Summary of dietary needs based on health status

    IMPORTANT: You MUST call the analyze_health_metrics tool first to get the patient data before providing any analysis.""",
    tools=[FunctionTool(func=analyze_health_metrics)],
    output_key="patient_health_data"
)

# Agent 2: Nutrition Requirements Calculator
nutrition_calculator_agent = Agent(
    model=Gemini(model="gemini-2.0-flash-exp", retry_options=retry_config),
    name="nutrition_calculator_agent",
    instruction="""You are a nutritional requirements calculator. Based on the patient health data provided by the previous agent: {patient_health_data}

                1. Calculate daily caloric needs:
                - Use Harris-Benedict or Mifflin-St Jeor equation for BMR
                - Formula for males: BMR = 10 × weight(kg) + 6.25 × height(cm) - 5 × age(years) + 5
                - Formula for females: BMR = 10 × weight(kg) + 6.25 × height(cm) - 5 × age(years) - 161
                - Apply activity level multiplier:
                    * Sedentary (little/no exercise): BMR × 1.2
                    * Lightly active (1-3 days/week): BMR × 1.375
                    * Moderately active (3-5 days/week): BMR × 1.55
                    * Very active (6-7 days/week): BMR × 1.725
                    * Super active (physical job + exercise): BMR × 1.9

                2. Determine optimal macronutrient distribution:
                - Protein: Calculate based on body weight and activity
                    * Sedentary: 0.8-1.0 g/kg
                    * Active/fitness: 1.6-2.2 g/kg
                    * Adjust for medical conditions (reduce if kidney issues)
                - Carbohydrates: Adjust based on activity level and medical conditions
                    * Standard: 45-65% of calories
                    * Diabetes: Lower GI carbs, controlled portions
                - Fats: Calculate remainder, ensuring minimum essential fat needs
                    * Standard: 20-35% of calories
                    * Focus on healthy fats for high cholesterol

                3. Apply medical condition adjustments:
                - Diabetes: Lower glycemic index carbs, controlled carb portions, higher fiber
                - Hypertension: Low sodium (< 2300mg/day or < 1500mg if severe)
                - High cholesterol: Limit saturated fats, increase fiber (25-30g/day), omega-3 rich foods
                - Thyroid issues: Ensure adequate iodine and selenium
                - Vitamin/mineral deficiencies: Note supplementation needs

                4. Provide a clear nutritional prescription with this exact format:

                **DAILY NUTRITIONAL TARGETS:**
                - Total Calories: [X] kcal
                - Protein: [X]g ([Y]%)
                - Carbohydrates: [X]g ([Y]%)
                - Fats: [X]g ([Y]%)

                **SPECIAL DIETARY GUIDELINES:**
                - [List specific guidelines like "Low sodium", "High fiber", "Low GI carbs", etc.]

                **MEAL DISTRIBUTION:**
                - Breakfast: [X]% ([Y] kcal)
                - Mid-morning snack: [X]% ([Y] kcal) [if applicable]
                - Lunch: [X]% ([Y] kcal)
                - Evening snack: [X]% ([Y] kcal) [if applicable]
                - Dinner: [X]% ([Y] kcal)

                **ADDITIONAL RECOMMENDATIONS:**
                - [Any specific nutrient focuses, timing considerations, hydration goals]

                Use web search if you need current nutritional guidelines or specific medical dietary recommendations.""",
    tools=[google_search],
    output_key="nutrition_requirements"
)

# Agent 3: Meal Planning
meal_planning_agent = Agent(
    model=Gemini(model="gemini-2.0-flash-exp", retry_options=retry_config),
    name="meal_planning_agent",
    instruction="""You are an expert meal planner specializing in Indian cuisine. Create a personalized daily meal plan that meets the nutritional requirements from the previous agent: {nutrition_requirements}

                **CRITICAL: HOW TO USE THE search-food-by-name TOOL:**

                1. The tool accepts a "name" parameter with the food name
                2. Search for foods using simple English names like:
                - "chapati" or "roti" or "wheat bread"
                - "rice" or "white rice" or "basmati rice"
                - "dal" or "lentils" or "moong dal"
                - "paneer" or "cottage cheese"
                - "chicken breast" or "chicken"
                - "milk" or "low fat milk"
                - "apple" or "banana"
                - "egg" or "boiled egg"

                3. The tool returns nutritional information per 100g:
                - calories (kcal)
                - protein (g)
                - carbohydrates (g)
                - fats (g)

                4. Calculate portions: If chapati per 100g has 300 kcal, and you need 150 kcal, serve 50g (half portion)

                **MEAL PLANNING PROCESS:**

                Step 1: For each meal, decide what foods to include based on:
                - Nutritional targets from previous agent
                - Dietary preferences (veg/non-veg)
                - Medical conditions
                - Cultural appropriateness

                Step 2: Use search-food-by-name to look up nutritional info for each food item

                Step 3: Calculate portions to meet calorie/macro targets for that meal

                Step 4: Format the meal with:
                - Food name and quantity (e.g., "2 medium chapatis (60g)")
                - Preparation notes if needed
                - Nutritional breakdown

                **MEAL STRUCTURE:**
                Follow the meal distribution from the previous agent (breakfast, lunch, dinner, snacks)

                **GUIDELINES:**
                - Vegetarian: Dal, paneer, legumes, dairy, nuts for protein
                - Non-vegetarian: Chicken, fish, eggs
                - Indian staples: Roti, rice, dal, sabzi, raita
                - Medical considerations:
                * Diabetes: Whole grains, low GI foods, controlled portions
                * Hypertension: No added salt, high potassium foods
                * High cholesterol: Oats, fiber-rich foods, healthy fats

                **OUTPUT FORMAT:**

                **DAILY MEAL PLAN**

                **BREAKFAST (7:00-8:00 AM)**
                - [Food item 1] ([quantity])
                - [Food item 2] ([quantity])
                - [Food item 3] ([quantity])
                Preparation: [any special notes]
                Nutrition: Calories: X kcal | Protein: Xg | Carbs: Xg | Fats: Xg

                **MID-MORNING SNACK (10:30 AM)** [if applicable]
                [Same format]

                **LUNCH (1:00-2:00 PM)**
                [Same format]

                **EVENING SNACK (4:30 PM)** [if applicable]
                [Same format]

                **DINNER (7:30-8:30 PM)**
                [Same format]

                **DAILY TOTALS:**
                - Calories: X kcal (Target: Y kcal) - [% achieved]
                - Protein: Xg (Target: Yg) - [% achieved]
                - Carbs: Xg (Target: Yg) - [% achieved]
                - Fats: Xg (Target: Yg) - [% achieved]

                IMPORTANT: Always use search-food-by-name to get accurate nutritional data before finalizing portions.""",
    tools=[nutrition_server, google_search],
    output_key="meal_plan"
)

# Agent 4: Meal Plan Validation
meal_validation_agent = Agent(
    model=Gemini(model="gemini-2.0-flash-exp", retry_options=retry_config),
    name="meal_validation_agent",
    instruction="""You are a meal plan validator. Review the meal plan from the previous agent: {meal_plan}

                **VALIDATION PROCESS:**

                1. **Nutritional Accuracy:**
                - Use search-food-by-name to verify 3-4 key foods
                - Check if total calories match target (±100 kcal acceptable)
                - Verify macros are within ±10% of targets
                - Ensure calculations are correct

                2. **Medical Compliance:**
                - Verify no contraindicated foods for medical conditions
                - Confirm dietary restrictions respected (veg/non-veg)
                - Check allergen considerations
                - Validate special needs (low sodium, high fiber, low GI)

                3. **Practical Assessment:**
                - Balanced meal distribution
                - Realistic portion sizes
                - Variety across meals
                - Cultural appropriateness

                **IF APPROVED, provide:**

                **✅ MEAL PLAN APPROVED**

                **Nutritional Targets Status:**
                - Calories: [Actual] vs [Target] - ✓ [Status]
                - Protein: [Actual]g vs [Target]g - ✓ [Status]
                - Carbs: [Actual]g vs [Target]g - ✓ [Status]
                - Fats: [Actual]g vs [Target]g - ✓ [Status]

                **Medical Compliance:** ✓ All requirements met
                - [Key compliance points]

                **Key Highlights:**
                - [3-4 positive aspects]

                **Special Considerations:**
                - [Important notes for patient]

                **Usage Tips:**
                - [Practical advice]

                **IF ISSUES FOUND, provide:**

                **⚠️ MEAL PLAN REQUIRES REVISION**

                **Issues Identified:**
                1. [Specific issue]
                2. [Specific issue]

                **Required Corrections:**
                - [What needs changing]
                - [Which meal/food needs adjustment]

                Use search-food-by-name to verify nutritional accuracy.""",
    tools=[nutrition_server, google_search],
    output_key="validated_meal_plan"
)

# Root Sequential Agent - Orchestrates the workflow
nutritionist_agent = SequentialAgent(
    name="nutritionist_agent",
    sub_agents=[
        patient_data_agent,
        nutrition_calculator_agent,
        meal_planning_agent,
        meal_validation_agent
    ],
    description="""You are the orchestrator of a comprehensive AI nutritionist system.

                Your workflow processes user requests through four specialized agents in sequence:

                1. PATIENT DATA AGENT: Retrieves and analyzes health data from questionnaire and lab results
                2. NUTRITION CALCULATOR: Determines personalized caloric and macronutrient requirements
                3. MEAL PLANNING AGENT: Creates a detailed, personalized meal plan using nutritional database
                4. VALIDATION AGENT: Reviews and approves the final meal plan

                When a user requests "generate a meal plan for the user":
                - Start the sequential workflow automatically
                - Each agent receives full context from all previous agents
                - Maintain information flow throughout the process
                - Return the final validated meal plan to the user

                Ensure smooth coordination and comprehensive output."""
)


async def main():
    """Main execution function to run the nutritionist agent"""
    
    # Initialize session service and runner
    session_service = InMemorySessionService()
    await session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)
    runner = Runner(agent=nutritionist_agent, app_name=APP_NAME, session_service=session_service, plugins=[LoggingPlugin()])
    # runner = InMemoryRunner(
    #     agent=nutritionist_agent,
    #     plugins=[LoggingPlugin()]
    # )
    
    # User query
    query = "Generate a meal plan for the user"
    content = types.Content(role='user', parts=[types.Part(text=query)])
    
    print("=" * 80)
    print("NUTRITIONIST AI AGENT SYSTEM")
    print("=" * 80)
    print(f"\nProcessing request: {query}\n")
    print("-" * 80)
    
    try:
        # Execute the agent workflow
        events = runner.run(user_id=USER_ID, session_id=SESSION_ID, new_message=content)
        
         # Display final meal plan
        
        print("\n" + "=" * 80)
        print("FINAL MEAL PLAN")
        print("=" * 80)
        for event in events:
            if event.is_final_response():
                final_response = event.content.parts[0].text
                print("Agent Response:", final_response)

        
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        traceback.print_exc()
    finally:
        # Cleanup
        await runner.close()

# Entry point
if __name__ == "__main__":
    asyncio.run(main())