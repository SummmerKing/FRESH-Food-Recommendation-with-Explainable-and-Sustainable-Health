import requests
import json

url = "http://localhost:8001/generate_meal_plan"

payload = {
    "user_id": "sathvik_demo",
    "pantry": ["paneer","dal","lentils", "bread", "rice", "spinach"],
    "likes": ["spicy", "dosa","healthy", "indian"],
    "diet": "veg",
    "time_budget": 45,
    "num_recs": 3  # Asking for 2 options per meal
}

print("Generating Meal Plan...")
try:
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        data = response.json()
        plan = data['meal_plan']
        
        for meal, recipes in plan.items():
            print(f"\n--- {meal.upper()} ({len(recipes)} options) ---")
            if not recipes:
                print("   No recipes found matching filters.")
                continue
            for r in recipes:
                print(f"   • {r['title']} (Score: {r['score']:.3f})")
                print(f"     Reason: {r['reason']}")
    else:
        print("Error:", response.text)
except Exception as e:
    print("Connection Failed:", e)