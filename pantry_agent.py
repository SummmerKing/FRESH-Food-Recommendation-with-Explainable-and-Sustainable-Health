import os
import json
from groq import Groq

# Reuse your existing API Key setup
client = Groq(api_key=os.environ.get("GROQ_API_KEY", "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"))

def expand_pantry_item(raw_input):
    """
    Input: "atta"
    Output: ["atta", "whole wheat flour", "chapati flour", "roti"]
    """
    
    system_prompt = """
    You are a Culinary Data Standardizer.
    Task: Take a raw ingredient input and return a JSON list containing:
    1. The original term.
    2. The standard English scientific/culinary name.
    3. Common dishes or forms it implies (e.g., 'rice flour' implies 'dosa batter').
    
    Example Input: "atta"
    Example JSON Output: {"tags": ["atta", "whole wheat flour", "chapati", "roti"]}
    
    Example Input: "maggi"
    Example JSON Output: {"tags": ["maggi", "instant noodles", "ramen"]}
    
    OUTPUT JSON ONLY.
    """
    
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": raw_input}
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        data = json.loads(completion.choices[0].message.content)
        return data.get("tags", [raw_input])
    except:
        return [raw_input] # Fallback