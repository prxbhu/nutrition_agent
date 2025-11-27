from asyncio import events
from google.adk.agents import Agent, SequentialAgent, LoopAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner
from google.adk.tools import FunctionTool, google_search
from google.adk.tools.agent_tool import AgentTool
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
MODEL_ID="gemini-2.5-flash-lite"

retry_config=types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504], # Retry on these HTTP errors
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


# Web Search Agent instead of direct google_search tool use
web_search_agent = Agent(
    name="web_search_agent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""You are a web search specialist. Your job is to perform web searches and return relevant results.

                When given a search query, use the google_search tool to find information and return the most relevant results in a clear, structured format.

                **Response Format:**
                Provide a concise summary of the most relevant information found, focusing on:
                - Key facts and data
                - Reliable sources
                - Relevant details to the query

                Be factual and cite sources when possible.""",
    tools=[google_search],
    output_key="search_results"
)

web_search_tool = AgentTool(web_search_agent)



# Agent 1: Patient Data Retrieval and Analysis
patient_data_agent = Agent(
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    name="patient_data_agent",
    instruction="""You are a patient data retrieval and analysis specialist.

                **TASK:**
                1. Call the analyze_health_metrics tool to fetch patient data
                2. Analyze the data comprehensively
                3. Output a structured JSON report

                **OUTPUT FORMAT - You MUST respond with ONLY valid JSON in this exact structure:**

                ```json
                {
                "patient_profile": {
                    "age": <number or "unknown">,
                    "gender": "<male/female/unknown>",
                    "height_cm": <number>,
                    "weight_kg": <number>,
                    "bmi": <number>,
                    "bmi_category": "<underweight/normal/overweight/obese>"
                },
                "dietary_preferences": {
                    "type": "<vegetarian/non-vegetarian/eggetarian>",
                    "restrictions": ["<list of restrictions>"],
                    "meal_prep_preference": "<home-cooked/outside/mixed>",
                    "eating_frequency": "<description>"
                },
                "medical_conditions": {
                    "current": ["<list of current conditions>"],
                    "family_history": ["<list of family history conditions>"],
                    "allergies": ["<list of allergies>"]
                },
                "lifestyle_factors": {
                    "exercise_minutes_per_week": <number>,
                    "exercise_level": "<sedentary/lightly_active/moderately_active/very_active>",
                    "water_intake_glasses": "<range or number>",
                    "alcohol": "<never/occasional/moderate/frequent>",
                    "smoking": "<never/former/current>",
                    "sleep_hours": <number>,
                    "stress_level": "<low/moderate/high>",
                    "screen_time_hours": <number>
                },
                "blood_test_analysis": {
                    "abnormal_values": [
                    {
                        "parameter": "<name>",
                        "value": <number>,
                        "unit": "<unit>",
                        "reference_range": "<range>",
                        "status": "<high/low>",
                        "clinical_significance": "<description>"
                    }
                    ],
                    "deficiencies": ["<list of identified deficiencies>"],
                    "health_risks": ["<list of health risks identified>"]
                },
                "key_nutritional_considerations": [
                    "<consideration 1>",
                    "<consideration 2>",
                    "<consideration 3>"
                ]
                }
                ```

                **IMPORTANT:** 
                - Output ONLY the JSON, no additional text before or after
                - Ensure all JSON is valid and properly formatted
                - Use the analyze_health_metrics tool first to get the data""",
    tools=[FunctionTool(func=analyze_health_metrics)],
    output_key="patient_health_data"
)

# Agent 2: Nutrition Requirements Calculator
nutrition_calculator_agent = Agent(
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    name="nutrition_calculator_agent",
    instruction="""You are a nutritional requirements calculator.

                **INPUT DATA:**
                You will receive patient health data in JSON format:
                {patient_health_data}

                **TASK:**
                1. Calculate daily caloric needs using Mifflin-St Jeor equation
                2. Determine macronutrient distribution
                3. Apply medical condition adjustments
                4. Calculate meal distribution

                **CALCULATION FORMULAS:**
                - BMR (Male): 10 × weight(kg) + 6.25 × height(cm) - 5 × age(years) + 5
                - BMR (Female): 10 × weight(kg) + 6.25 × height(cm) - 5 × age(years) - 161
                - Activity Multipliers: Sedentary=1.2, Lightly Active=1.375, Moderately Active=1.55, Very Active=1.725

                **OUTPUT FORMAT - You MUST respond with ONLY valid JSON in this exact structure:**

                ```json 
                {
                "daily_targets": {
                    "total_calories": <number>,
                    "protein_grams": <number>,
                    "protein_percentage": <number>,
                    "carbohydrates_grams": <number>,
                    "carbohydrates_percentage": <number>,
                    "fats_grams": <number>,
                    "fats_percentage": <number>,
                    "fiber_grams": <number>,    
                    "sodium_mg": <number>
                },
                "calculations": {
                    "bmr": <number>,
                    "activity_level": "<sedentary/lightly_active/moderately_active/very_active>",
                    "activity_multiplier": <number>,
                    "tdee": <number>
                },
                "special_dietary_guidelines": [
                    "<guideline 1>",
                    "<guideline 2>",
                    "<guideline 3>"
                ],
                "meal_distribution": {
                    "breakfast": {
                    "percentage": <number>,
                    "calories": <number>
                    },
                    "mid_morning_snack": {
                    "percentage": <number>,
                    "calories": <number>,
                    "include": <boolean>
                    },
                    "lunch": {
                    "percentage": <number>,
                    "calories": <number>
                    },
                    "evening_snack": {
                    "percentage": <number>,
                    "calories": <number>,
                    "include": <boolean>
                    },
                    "dinner": {
                    "percentage": <number>,
                    "calories": <number>
                    }
                },
                "medical_adjustments": {
                    "diabetes": "<adjustments if applicable>",
                    "hypertension": "<adjustments if applicable>",
                    "high_cholesterol": "<adjustments if applicable>",
                    "vitamin_deficiencies": "<adjustments if applicable>"
                },
                "additional_recommendations": [
                    "<recommendation 1>",
                    "<recommendation 2>"
                ]
                }
                ```

                **IMPORTANT:** 
                - Output ONLY the JSON, no additional text
                - Use web search using the web_search_tool if you need current nutritional guidelines
                - Ensure calculations are accurate""",
    tools=[web_search_tool],
    output_key="nutrition_requirements"
)

# Agent 3: Initial Meal Planning Agent
initial_meal_planner_agent = Agent(
    model=Gemini(model="gemini-2.0-flash-exp", retry_options=retry_config),
    name="initial_meal_planner_agent",
    instruction="""You are an expert meal planner specializing in Indian cuisine.

                **INPUT DATA:**
                Patient Health Data: {patient_health_data}
                Nutrition Requirements: {nutrition_requirements}

                **TASK:**
                Create an initial personalized Indian meal plan that meets the nutritional targets.

                **PROCESS:**
                1. Review patient preferences and restrictions from patient_health_data
                2. Use nutrition_requirements for calorie and macro targets
                3. Use web search to find accurate nutritional values for Indian foods
                4. Calculate portions to meet targets
                5. Ensure medical compliance

                **SEARCH EXAMPLES:**
                - "nutritional value of chapati per 100g"
                - "calories in dal 1 cup"
                - "paneer nutrition facts"

                **OUTPUT FORMAT - You MUST respond with ONLY valid JSON in this exact structure:**

                ```json
                {
                "meal_plan": {
                    "breakfast": {
                    "time": "7:00-8:00 AM",
                    "target_calories": <number>,
                    "foods": [
                        {
                        "name": "<food name>",
                        "quantity": "<amount with unit>",
                        "calories": <number>,
                        "protein_g": <number>,
                        "carbs_g": <number>,
                        "fats_g": <number>
                        }
                    ],
                    "preparation_notes": "<any special instructions>",
                    "totals": {
                        "calories": <number>,
                        "protein_g": <number>,
                        "carbs_g": <number>,
                        "fats_g": <number>
                    }
                    },
                    "mid_morning_snack": {
                    "time": "10:30-11:00 AM",
                    "target_calories": <number>,
                    "include": <boolean>,
                    "foods": [],
                    "preparation_notes": "",
                    "totals": {
                        "calories": <number>,
                        "protein_g": <number>,
                        "carbs_g": <number>,
                        "fats_g": <number>
                    }
                    },
                    "lunch": {
                    "time": "1:00-2:00 PM",
                    "target_calories": <number>,
                    "foods": [],
                    "preparation_notes": "",
                    "totals": {
                        "calories": <number>,
                        "protein_g": <number>,
                        "carbs_g": <number>,
                        "fats_g": <number>
                    }
                    },
                    "evening_snack": {
                    "time": "4:30-5:00 PM",
                    "target_calories": <number>,
                    "include": <boolean>,
                    "foods": [],
                    "preparation_notes": "",
                    "totals": {
                        "calories": <number>,
                        "protein_g": <number>,
                        "carbs_g": <number>,
                        "fats_g": <number>
                    }
                    },
                    "dinner": {
                    "time": "7:30-8:30 PM",
                    "target_calories": <number>,
                    "foods": [],
                    "preparation_notes": "",
                    "totals": {
                        "calories": <number>,
                        "protein_g": <number>,
                        "carbs_g": <number>,
                        "fats_g": <number>
                    }
                    }
                },
                "daily_totals": {
                    "calories": <number>,
                    "protein_g": <number>,
                    "carbs_g": <number>,
                    "fats_g": <number>,
                    "fiber_g": <number>
                },
                "target_comparison": {
                    "calories_percentage": <number>,
                    "protein_percentage": <number>,
                    "carbs_percentage": <number>,
                    "fats_percentage": <number>
                },
                "important_notes": [
                    "<note 1>",
                    "<note 2>"
                ]
                }
                ```

                **IMPORTANT:** 
                - Output ONLY valid JSON, no additional text
                - Use web search using the web_search_tool if you need to verify nutritional values
                - Ensure all meals follow medical guidelines from patient data""",
    tools=[web_search_tool],
    output_key="current_meal_plan"
)

# Agent 4: Meal Plan Critic
meal_plan_critic_agent = Agent(
    model=Gemini(model="gemini-2.0-flash-exp", retry_options=retry_config),
    name="meal_plan_critic_agent",
    instruction="""You are a meal plan critic and validator.

                **INPUT DATA:**
                Patient Health Data: {patient_health_data}
                Nutrition Requirements: {nutrition_requirements}
                Current Meal Plan: {current_meal_plan}

                **TASK:**
                Evaluate the meal plan and determine if it's approved or needs refinement.

                **EVALUATION CRITERIA:**
                1. Nutritional Accuracy: Calories within ±100 kcal, macros within ±10%
                2. Medical Compliance: Follows all dietary guidelines for medical conditions
                3. Practical Assessment: Realistic portions, cultural appropriateness
                4. Completeness: All required meals included with proper calculations

                **OUTPUT FORMAT - You MUST respond with ONLY valid JSON in this exact structure:**

                ```json
                {
                "status": "<APPROVED or NEEDS_REVISION>",
                "nutritional_accuracy": {
                    "calories_status": "<within_range/too_high/too_low>",
                    "calories_difference": <number>,
                    "protein_status": "<within_range/too_high/too_low>",
                    "carbs_status": "<within_range/too_high/too_low>",
                    "fats_status": "<within_range/too_high/too_low>",
                    "overall_score": "<excellent/good/needs_improvement/poor>"
                },
                "medical_compliance": {
                    "diabetes_compliance": "<compliant/non_compliant/not_applicable>",
                    "hypertension_compliance": "<compliant/non_compliant/not_applicable>",
                    "cholesterol_compliance": "<compliant/non_compliant/not_applicable>",
                    "dietary_restrictions_followed": <boolean>,
                    "overall_score": "<excellent/good/needs_improvement/poor>"
                },
                "practical_assessment": {
                    "portion_sizes": "<realistic/unrealistic>",
                    "meal_variety": "<excellent/good/poor>",
                    "cultural_appropriateness": "<appropriate/inappropriate>",
                    "ease_of_preparation": "<easy/moderate/difficult>",
                    "overall_score": "<excellent/good/needs_improvement/poor>"
                },
                "issues": [
                    {
                    "category": "<nutritional/medical/practical/completeness>",
                    "severity": "<critical/major/minor>",
                    "problem": "<description of the problem>",
                    "suggestion": "<specific actionable fix>"
                    }
                ],
                "summary": "<brief summary of evaluation>",
                "approval_reason": "<why approved or why not approved>"
                }
                ```

                **CRITICAL RULES:**
                - Set "status" to "APPROVED" ONLY if:
                - Nutritional accuracy overall_score is "excellent" or "good"
                - Medical compliance overall_score is "excellent" or "good"
                - Practical assessment overall_score is "excellent" or "good"
                - No critical or major issues found
                - Set "status" to "NEEDS_REVISION" if any criteria are not met
                - Always provide specific, actionable suggestions in issues array

                **IMPORTANT:** 
                - Output ONLY valid JSON, no additional text
                - Use web search using the web_search_tool to verify nutritional values if needed""",
    tools=[web_search_tool],
    output_key="critique"
)

def exit_loop() -> Dict[str, str]:
    """
    Call this function ONLY when the meal plan critique status is 'APPROVED', 
    indicating the meal plan is complete and no more changes are needed.
    
    Returns:
        A dictionary with approval status
    """
    return {
        "status": "approved",
        "message": "Meal plan approved. Exiting refinement loop."
    }
exit_tool = FunctionTool(func=exit_loop)

# Agent 5: Meal Plan Refiner
meal_plan_refiner_agent = Agent(
    model=Gemini(model="gemini-2.0-flash-exp", retry_options=retry_config),
    name="meal_plan_refiner_agent",
    instruction="""You are a meal plan refiner.

                **INPUT DATA:**
                Patient Health Data: {patient_health_data}
                Nutrition Requirements: {nutrition_requirements}
                Current Meal Plan: {current_meal_plan}
                Critique: {critique}

                **TASK:**
                1. Parse the critique JSON to check the status
                2. If status is "APPROVED", call the exit_tool immediately
                3. If status is "NEEDS_REVISION", refine the meal plan to address ALL issues

                **DECISION LOGIC:**
                ```
                IF critique.status == "APPROVED":
                    CALL exit_tool
                    STOP - do not output anything else
                ELSE:
                    Refine the meal plan
                    Output refined meal plan in JSON format
                ```

                **REFINEMENT PROCESS:**
                1. Read each issue from critique.issues array
                2. For each issue, make specific changes to address it
                3. Use web search to get accurate nutritional data for any new/modified foods
                4. Recalculate all nutritional values
                5. Ensure the refined plan maintains the same JSON structure

                **OUTPUT FORMAT - If refining, output ONLY valid JSON in the SAME structure as initial meal plan:**

                ```json
                {
                "meal_plan": {
                    "breakfast": { ... },
                    "mid_morning_snack": { ... },
                    "lunch": { ... },
                    "evening_snack": { ... },
                    "dinner": { ... }
                },
                "daily_totals": { ... },
                "target_comparison": { ... },
                "important_notes": [ ... ],
                "refinement_notes": [
                    "<what was changed and why>",
                    "<issue addressed>"
                ]
                }
                ```

                **IMPORTANT:** 
                - If critique.status is "APPROVED", ONLY call exit_tool, do NOT output JSON
                - If refining, output ONLY valid JSON, no additional text
                - Address ALL issues from the critique
                - Use web search to verify nutritional accuracy""",
    tools=[exit_tool, web_search_tool],
    output_key="current_meal_plan"
)

# Loop Agent: Meal Plan Refinement Loop
meal_plan_refinement_loop = LoopAgent(
    name="meal_plan_refinement_loop",
    sub_agents=[meal_plan_critic_agent, meal_plan_refiner_agent],
    max_iterations=3,
)

# Root Sequential Agent - Orchestrates the workflow
nutritionist_agent = SequentialAgent(
    name="nutritionist_agent",
    sub_agents=[
        patient_data_agent,
        nutrition_calculator_agent,
        initial_meal_planner_agent,
        meal_plan_refinement_loop
    ],
    description="""You are the orchestrator of a comprehensive AI nutritionist system  with structured JSON data flow.

                **Workflow:**
                1. Patient Data Agent → outputs patient_health_data (JSON)
                2. Nutrition Calculator → uses patient_health_data, outputs nutrition_requirements (JSON)
                3. Initial Meal Planner → uses both JSONs, outputs current_meal_plan (JSON)
                4. Refinement Loop (max 3 iterations):
                - Critic → evaluates meal plan, outputs critique (JSON with status)
                - Refiner → if APPROVED calls exit_tool, else refines and outputs updated meal plan (JSON)

                All agents communicate via structured JSON, ensuring reliable data passing."""
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
        agent_responses = []
        iteration_count = 0
        # Display final meal plan 
        # print("\n" + "=" * 80)
        # print("FINAL MEAL PLAN")
        # print("=" * 80)
        # for event in events:
        #     if event.is_final_response():
        #         final_response = event.content.parts[0].text
        #         print("Agent Response:", final_response)
        
        for event in events:
            if hasattr(event, 'is_final_response') and event.is_final_response():
                if hasattr(event, 'content') and event.content.parts:
                    response = event.content.parts[0].text
                    agent_name = event.author if hasattr(event, 'author') else "Unknown"
                    
                    # Track iterations
                    if 'critic' in agent_name.lower():
                        iteration_count += 1
                    
                    agent_responses.append({
                        'agent': agent_name,
                        'response': response,
                        'iteration': iteration_count if iteration_count > 0 else None
                    })
        
        # Parse and display final meal plan
        if agent_responses:
            final_response = agent_responses[-1]['response']
            
            # Try to parse as JSON for prettier display
            try:
                final_json = json.loads(final_response)
                
                print("\n" + "=" * 80)
                print("FINAL REFINED & VALIDATED MEAL PLAN (JSON)")
                print("=" * 80)
                print(json.dumps(final_json, indent=2))
                
                # Extract and display key information
                if 'meal_plan' in final_json:
                    print("\n" + "=" * 80)
                    print("MEAL SUMMARY")
                    print("=" * 80)
                    
                    for meal_name, meal_data in final_json['meal_plan'].items():
                        if meal_data.get('include', True):
                            print(f"\n{meal_name.upper().replace('_', ' ')}:")
                            print(f"  Time: {meal_data.get('time', 'N/A')}")
                            print(f"  Calories: {meal_data.get('totals', {}).get('calories', 0)} kcal")
                            print(f"  Foods:")
                            for food in meal_data.get('foods', []):
                                print(f"    - {food.get('name')} ({food.get('quantity')})")
                    
                    if 'daily_totals' in final_json:
                        print(f"\nDAILY TOTALS:")
                        totals = final_json['daily_totals']
                        print(f"  Calories: {totals.get('calories', 0)} kcal")
                        print(f"  Protein: {totals.get('protein_g', 0)}g")
                        print(f"  Carbs: {totals.get('carbs_g', 0)}g")
                        print(f"  Fats: {totals.get('fats_g', 0)}g")
                
            except json.JSONDecodeError:
                # If not JSON, display as text
                print("\n" + "=" * 80)
                print("FINAL OUTPUT")
                print("=" * 80)
                print(final_response)
            
            # # Show refinement summary
            # print("\n" + "=" * 80)
            # print("REFINEMENT SUMMARY")
            # print("=" * 80)
            # print(f"Total refinement iterations: {iteration_count}")
            
            # # Show critique history
            # critic_responses = [r for r in agent_responses if 'critic' in r['agent'].lower()]
            # if critic_responses:
            #     print(f"\nCritique iterations: {len(critic_responses)}")
            #     for idx, critic in enumerate(critic_responses, 1):
            #         print(f"\n--- Iteration {idx} Critique ---")
            #         try:
            #             critique_json = json.loads(critic['response'])
            #             print(f"Status: {critique_json.get('status', 'UNKNOWN')}")
            #             print(f"Summary: {critique_json.get('summary', 'N/A')}")
            #             if critique_json.get('issues'):
            #                 print(f"Issues found: {len(critique_json['issues'])}")
            #         except:
            #             print(critic['response'][:300] + "...")
                        
        else:
            print("\n⚠️ No final response received from agents")

        
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        traceback.print_exc()
    finally:
        try:
            await runner.close()
            print("\n✅ Runner closed successfully")
        except Exception as cleanup_error:
            print(f"⚠️ Cleanup error (can be ignored): {cleanup_error}")


# Entry point
if __name__ == "__main__":
    asyncio.run(main())