# scripts/build_sample_recipes_sampled.py
"""
Build a balanced sample of up to N recipes per meal_type (breakfast/lunch/dinner).
This helps quick testing and ensures each meal has candidates.
"""
import pandas as pd
import json
from pathlib import Path
import random

REPO_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = REPO_ROOT / r"C:\Users\sathv\OneDrive\Desktop\FRESH\csv_versions\clean_recipes_with_cleaned_ingredients.csv"
OUT_PATH = REPO_ROOT / "data" / "sample_recipes.json"

N_PER_MEAL = 20   # adjust: how many recipes per meal to include (max)

df = pd.read_csv(CSV_PATH)

# normalize meal_type text
df["meal_norm"] = df["meal"].fillna("").astype(str).str.strip().str.lower()
# fallback: try to infer if meal blank (not implemented here)
meals = ["breakfast", "lunch", "dinner"]
selected = []

for meal in meals:
    pool = df[df["meal_norm"] == meal]
    if pool.empty:
        pool = df  # if no explicit tag, allow any (optional)
    # sample up to N_PER_MEAL without replacement
    sample = pool.sample(n=min(N_PER_MEAL, len(pool)), random_state=42)
    for idx, row in sample.iterrows():
        rid = f"r{idx+1:03d}"
        # ingredients parse
        ingredients = []
        val = row.get("cleaned_ingredients", "")
        if isinstance(val, str) and val.startswith("["):
            try:
                ingredients = json.loads(val.replace("'", '"'))
            except:
                ingredients = [i.strip() for i in val.strip("[]").split(",") if i.strip()]
        else:
            parts = []
            for p in str(row.get("ingredients","")).split("\n"):
                parts += p.split(",")
            ingredients = [p.strip() for p in parts if p.strip()]

        diet = row.get("diet_tags", row.get("diet", []))
        if isinstance(diet, str) and diet.startswith("["):
            try:
                diet = json.loads(diet.replace("'", '"'))
            except:
                pass

        recipe = {
            "recipe_id": rid,
            "title": row.get("dish") or row.get("title") or rid,
            "dish": row.get("dish") or row.get("title") or rid,
            "ingredients": ingredients,
            "cleaned_ingredients": ingredients,
            "time": row.get("time",""),
            "time_minutes": int(row.get("time_minutes")) if not pd.isna(row.get("time_minutes")) else None,
            "steps": row.get("steps","") if not pd.isna(row.get("steps","")) else "",
            "diet": diet,
            "meal_type": meal
        }
        selected.append(recipe)

# remove duplicates by recipe_id/title
seen = set()
unique = []
for r in selected:
    key = (r["title"].strip().lower())
    if key in seen:
        continue
    seen.add(key)
    unique.append(r)

Path(OUT_PATH).parent.mkdir(exist_ok=True, parents=True)
with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(unique, f, indent=2, ensure_ascii=False)

print(f"Wrote {len(unique)} recipes to {OUT_PATH}")
