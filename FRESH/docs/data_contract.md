# Data Contract

This document defines the input/output schemas used in the Food Recommendation System.

---

## 1. Recipe Schema (`sample_recipes.json`)
Each recipe in the database has the following fields:

```json
{
  "recipe_id": "r001",
  "title": "Thandai (Indian Almond Drink)",
  "diet": ["vegetarian","vegan","gluten free"],
  "ingredients": ["water","sugar","almonds","cardamom"],
  "time": 15,
  "steps": "Soak sugar... grind... chill...",
  "meal_type": "Dinner"
}
