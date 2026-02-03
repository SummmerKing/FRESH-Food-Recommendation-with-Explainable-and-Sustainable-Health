

import os
import json
from groq import Groq

# 1. SETUP CLIENT SAFELY
# Set this in your terminal: export GROQ_API_KEY="gsk_..."
api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    # Fallback only for local testing, NEVER commit this to GitHub
    api_key = ""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"" 

client = Groq(api_key=api_key)

def query_agent(user_query, user_profile, pantry_items, nutrition_insight):
    """
    Step 1: Analyzes the user's vague request + Nutrition Advice 
    to generate specific search keywords ('likes') for the Neural Network.
    """
    
    # We LIMIT pantry to top 50 to save tokens, but give enough for context
    pantry_str = ', '.join(pantry_items[:50])
    
    system_prompt = f"""
    You are the FRESH AI Logic Unit (Query Translation Agent).
    
    INPUT CONTEXT:
    - User Query: "{user_query}"
    - Diet: {user_profile.get('diet', 'Any')}
    - BMI/Goal: {user_profile.get('bmi', 'Normal')}
    - Available Pantry: {pantry_str}
    - NUTRITIONIST INSIGHT: "{nutrition_insight}"
    
    TASK:
    Translate the User Query and Nutrition Insight into specific food search terms (keywords) 
    that will help us find the best recipes in our database.
    
    RULES:
    1. If the Nutritionist suggests "High Protein", add keywords like "Lentils", "Paneer", "Chicken".
    2. If the user asks for "Dinner", set time_budget higher (e.g., 45-60 mins).
    3. Use the Pantry list to bias keywords towards ingredients they actually have.
    
    OUTPUT JSON FORMAT:
    {{
        "likes": ["keyword1", "keyword2", "keyword3", "ingredient_from_pantry"],
        "time_budget": <int_minutes>,
        "meal_type": "breakfast" | "lunch" | "dinner",
        "reasoning": "Brief explanation of why you chose these keywords."
    }}
    """
    
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile", # Great choice for logic
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Generate search parameters."}
            ],
            temperature=0.1, # Keep it strict/deterministic
            response_format={"type": "json_object"} 
        )
        
        return json.loads(completion.choices[0].message.content)
        
    except Exception as e:
        print(f"❌ Query Agent Failed: {e}")
        # Fallback defaults if LLM crashes
        return {
            "likes": [user_query], 
            "time_budget": 45, 
            "meal_type": "dinner", 
            "reasoning": "Fallback mode"
        }