# recipe_data_cleaner.py
"""
Script to clean and validate recipe metadata
Fixes:
1. Contradictory dish names (e.g., "veg chicken curry")
2. Wrong diet tags (e.g., "fish curry" tagged as vegetarian)
3. Incorrect meal type assignments (e.g., "rava idli" as dinner)
"""

import json
import os
import re
from typing import Dict, List, Set, Tuple
from collections import defaultdict

# ========== CONFIGURATION ==========
METADATA_PATH = "/data1/home/sathvik/Documents/FRESH/recipes_metadata_full.json"
OUTPUT_PATH = "/data1/home/sathvik/Documents/FRESH/recipes_metadata_full_CLEANED.json"
REPORT_PATH = "/data1/home/sathvik/Documents/FRESH/cleaning_report.txt"

# ========== DETECTION PATTERNS ==========

# Non-vegetarian ingredients/keywords
NON_VEG_KEYWORDS = {
    'chicken', 'fish', 'mutton', 'lamb', 'beef', 'pork', 'prawn', 'shrimp',
    'crab', 'lobster', 'egg', 'meat', 'turkey', 'duck', 'bacon', 'sausage',
    'ham', 'salmon', 'tuna', 'anchovies', 'seafood', 'keema', 'tikka'
}

# Vegetarian prefixes that might be misleading
VEG_PREFIXES = {'veg', 'vegetarian', 'veggie', 'soya', 'soy', 'mock', 'plant-based'}

# Breakfast items (should be breakfast, not dinner)
BREAKFAST_ITEMS = {
    'idli', 'dosa', 'upma', 'poha', 'paratha', 'puri', 'chilla', 'uttapam',
    'pancake', 'waffle', 'toast', 'cereal', 'oatmeal', 'porridge', 'smoothie',
    'juice', 'omelette', 'scrambled', 'rava', 'pesarattu', 'appam', 'puttu'
}

# Snack items (should be snack, not main meal)
SNACK_ITEMS = {
    'samosa', 'pakora', 'vada', 'bhaji', 'cutlet', 'tikki', 'chips', 'fries',
    'sandwich', 'roll', 'wrap', 'toast', 'bruschetta', 'nachos'
}

# Light items that could be breakfast or snack
LIGHT_ITEMS = {
    'salad', 'soup', 'smoothie', 'shake', 'juice', 'lassi'
}

# ========== CLEANING FUNCTIONS ==========

def detect_contradictory_name(title: str, diet: List[str]) -> Tuple[bool, str]:
    """
    Detect contradictory names like 'veg chicken' or 'fish' with vegetarian tag
    Returns: (has_issue, corrected_diet)
    """
    title_lower = title.lower()
    diet_str = ' '.join([str(d).lower() for d in diet]) if isinstance(diet, list) else str(diet).lower()
    
    # Check for non-veg keywords in title
    found_nonveg = []
    for keyword in NON_VEG_KEYWORDS:
        if keyword in title_lower:
            found_nonveg.append(keyword)
    
    if not found_nonveg:
        return False, None
    
    # Check if it's a "veg chicken" type situation (has both veg prefix and non-veg keyword)
    has_veg_prefix = any(prefix in title_lower for prefix in VEG_PREFIXES)
    
    # Check current diet tags
    is_tagged_veg = 'vegetarian' in diet_str or 'vegan' in diet_str
    
    if has_veg_prefix and found_nonveg:
        # Case: "Veg Chicken Curry" - keep as vegetarian, it's likely soy/mock meat
        return False, None
    
    if found_nonveg and is_tagged_veg:
        # Case: "Fish Curry" tagged as vegetarian - WRONG
        return True, "non-vegetarian"
    
    if found_nonveg and not is_tagged_veg:
        # Ensure it's properly tagged as non-veg
        if 'non-vegetarian' not in diet_str:
            return True, "non-vegetarian"
    
    return False, None


def detect_wrong_meal_type(title: str, current_meal_type: str, time_minutes: int = 30) -> Tuple[bool, str]:
    """
    Detect wrong meal type assignments
    Returns: (has_issue, corrected_meal_type)
    """
    title_lower = title.lower()
    current_type = str(current_meal_type).lower().strip()
    
    # Check for breakfast items
    for item in BREAKFAST_ITEMS:
        if item in title_lower:
            if current_type not in ['breakfast', 'brunch']:
                return True, 'breakfast'
    
    # Check for snacks
    for item in SNACK_ITEMS:
        if item in title_lower:
            if current_type not in ['snack', 'appetizer']:
                return True, 'snack'
    
    # Light items - prefer breakfast if current is dinner/lunch
    for item in LIGHT_ITEMS:
        if item in title_lower and time_minutes < 20:
            if current_type in ['dinner', 'lunch']:
                return True, 'breakfast'
    
    return False, None


def detect_ingredient_diet_mismatch(ingredients: List[str], diet: List[str]) -> Tuple[bool, str, List[str]]:
    """
    Check if ingredients contradict diet tags
    Returns: (has_issue, corrected_diet, found_non_veg_ingredients)
    """
    if not ingredients:
        return False, None, []
    
    ingredients_text = ' '.join([str(ing).lower() for ing in ingredients])
    diet_str = ' '.join([str(d).lower() for d in diet]) if isinstance(diet, list) else str(diet).lower()
    
    found_nonveg = []
    for keyword in NON_VEG_KEYWORDS:
        if keyword in ingredients_text:
            found_nonveg.append(keyword)
    
    if not found_nonveg:
        return False, None, []
    
    is_tagged_veg = 'vegetarian' in diet_str or 'vegan' in diet_str
    
    if found_nonveg and is_tagged_veg:
        return True, "non-vegetarian", found_nonveg
    
    return False, None, []


def clean_recipe_metadata(metadata: Dict) -> Tuple[Dict, Dict]:
    """
    Clean a single recipe's metadata
    Returns: (cleaned_metadata, issues_found)
    """
    issues = {
        'contradictory_name': False,
        'wrong_diet_tag': False,
        'wrong_meal_type': False,
        'ingredient_mismatch': False,
        'details': []
    }
    
    cleaned = metadata.copy()
    title = cleaned.get('title', '')
    diet = cleaned.get('diet', [])
    meal_type = cleaned.get('meal_type', 'dinner')
    time_minutes = cleaned.get('time_minutes', 30)
    ingredients = cleaned.get('ingredients', [])
    
    # Ensure diet is a list
    if isinstance(diet, str):
        diet = [diet]
    
    # Check 1: Contradictory name
    has_name_issue, corrected_diet = detect_contradictory_name(title, diet)
    if has_name_issue and corrected_diet:
        issues['contradictory_name'] = True
        issues['details'].append(f"Title '{title}' contradicts diet tag. Changed to: {corrected_diet}")
        cleaned['diet'] = [corrected_diet]
        diet = [corrected_diet]
    
    # Check 2: Ingredient-diet mismatch
    has_ing_issue, corrected_diet_ing, found_nonveg = detect_ingredient_diet_mismatch(ingredients, diet)
    if has_ing_issue and corrected_diet_ing:
        issues['ingredient_mismatch'] = True
        issues['details'].append(
            f"Ingredients contain {', '.join(found_nonveg)} but tagged as vegetarian. Changed to: {corrected_diet_ing}"
        )
        cleaned['diet'] = [corrected_diet_ing]
    
    # Check 3: Wrong meal type
    has_meal_issue, corrected_meal = detect_wrong_meal_type(title, meal_type, time_minutes)
    if has_meal_issue and corrected_meal:
        issues['wrong_meal_type'] = True
        issues['details'].append(
            f"Meal type '{meal_type}' incorrect for '{title}'. Changed to: {corrected_meal}"
        )
        cleaned['meal_type'] = corrected_meal
    
    return cleaned, issues


def analyze_dataset(metadata_list: List[Dict]) -> Dict:
    """
    Analyze the entire dataset for common issues
    """
    stats = {
        'total_recipes': len(metadata_list),
        'contradictory_names': [],
        'wrong_diet_tags': [],
        'wrong_meal_types': [],
        'ingredient_mismatches': [],
        'diet_distribution': defaultdict(int),
        'meal_type_distribution': defaultdict(int)
    }
    
    for recipe in metadata_list:
        title = recipe.get('title', '')
        diet = recipe.get('diet', [])
        meal_type = recipe.get('meal_type', 'unknown')
        
        # Count distributions
        if isinstance(diet, list):
            for d in diet:
                stats['diet_distribution'][str(d)] += 1
        else:
            stats['diet_distribution'][str(diet)] += 1
        
        stats['meal_type_distribution'][str(meal_type)] += 1
        
        # Check for issues
        has_name_issue, _ = detect_contradictory_name(title, diet)
        if has_name_issue:
            stats['contradictory_names'].append(title)
        
        has_meal_issue, _ = detect_wrong_meal_type(title, meal_type)
        if has_meal_issue:
            stats['wrong_meal_types'].append((title, meal_type))
        
        has_ing_issue, _, found_nonveg = detect_ingredient_diet_mismatch(
            recipe.get('ingredients', []), diet
        )
        if has_ing_issue:
            stats['ingredient_mismatches'].append((title, found_nonveg))
    
    return stats


# ========== MAIN PROCESSING ==========

def main():
    print("=" * 80)
    print("RECIPE DATA CLEANING & VALIDATION")
    print("=" * 80)
    
    # Load data
    print(f"\n📂 Loading data from: {METADATA_PATH}")
    if not os.path.exists(METADATA_PATH):
        print(f"❌ Error: File not found: {METADATA_PATH}")
        return
    
    with open(METADATA_PATH, 'r', encoding='utf-8') as f:
        metadata_list = json.load(f)
    
    print(f"✅ Loaded {len(metadata_list)} recipes")
    
    # Analyze before cleaning
    print(f"\n📊 Analyzing dataset for issues...")
    stats_before = analyze_dataset(metadata_list)
    
    print(f"\n{'='*80}")
    print("ISSUES FOUND:")
    print(f"{'='*80}")
    print(f"🔴 Contradictory names: {len(stats_before['contradictory_names'])}")
    if stats_before['contradictory_names'][:5]:
        print("   Examples:")
        for name in stats_before['contradictory_names'][:5]:
            print(f"   - {name}")
    
    print(f"\n🔴 Wrong meal types: {len(stats_before['wrong_meal_types'])}")
    if stats_before['wrong_meal_types'][:5]:
        print("   Examples:")
        for name, meal in stats_before['wrong_meal_types'][:5]:
            print(f"   - {name} (currently: {meal})")
    
    print(f"\n🔴 Ingredient-diet mismatches: {len(stats_before['ingredient_mismatches'])}")
    if stats_before['ingredient_mismatches'][:5]:
        print("   Examples:")
        for name, ingredients in stats_before['ingredient_mismatches'][:5]:
            print(f"   - {name} (contains: {', '.join(ingredients)})")
    
    # Clean data
    print(f"\n{'='*80}")
    print("CLEANING DATA...")
    print(f"{'='*80}")
    
    cleaned_metadata = []
    all_issues = []
    issue_counts = {
        'contradictory_name': 0,
        'wrong_diet_tag': 0,
        'wrong_meal_type': 0,
        'ingredient_mismatch': 0
    }
    
    for recipe in metadata_list:
        cleaned_recipe, issues = clean_recipe_metadata(recipe)
        cleaned_metadata.append(cleaned_recipe)
        
        if any(issues.values()):
            all_issues.append({
                'recipe_id': recipe.get('recipe_id', recipe.get('title')),
                'title': recipe.get('title', ''),
                'issues': issues
            })
            
            for key in issue_counts:
                if issues.get(key, False):
                    issue_counts[key] += 1
    
    print(f"\n✅ Cleaned {len(cleaned_metadata)} recipes")
    print(f"\n📊 Issues fixed:")
    print(f"   - Contradictory names: {issue_counts['contradictory_name']}")
    print(f"   - Wrong diet tags: {issue_counts['wrong_diet_tag']}")
    print(f"   - Wrong meal types: {issue_counts['wrong_meal_type']}")
    print(f"   - Ingredient mismatches: {issue_counts['ingredient_mismatch']}")
    
    # Save cleaned data
    print(f"\n💾 Saving cleaned data to: {OUTPUT_PATH}")
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(cleaned_metadata, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Saved cleaned data")
    
    # Generate report
    print(f"\n📝 Generating detailed report: {REPORT_PATH}")
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write("RECIPE DATA CLEANING REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Total recipes processed: {len(metadata_list)}\n")
        f.write(f"Issues found and fixed: {len(all_issues)}\n\n")
        
        f.write("BEFORE CLEANING:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Diet distribution:\n")
        for diet, count in sorted(stats_before['diet_distribution'].items(), key=lambda x: -x[1]):
            f.write(f"  {diet}: {count}\n")
        
        f.write(f"\nMeal type distribution:\n")
        for meal, count in sorted(stats_before['meal_type_distribution'].items(), key=lambda x: -x[1]):
            f.write(f"  {meal}: {count}\n")
        
        f.write("\n\nDETAILED ISSUES:\n")
        f.write("=" * 80 + "\n\n")
        
        for issue_data in all_issues[:100]:  # Limit to first 100 for readability
            f.write(f"Recipe: {issue_data['title']}\n")
            f.write(f"ID: {issue_data['recipe_id']}\n")
            for detail in issue_data['issues']['details']:
                f.write(f"  - {detail}\n")
            f.write("\n")
    
    print(f"✅ Report saved")
    
    # Analyze after cleaning
    stats_after = analyze_dataset(cleaned_metadata)
    
    print(f"\n{'='*80}")
    print("AFTER CLEANING:")
    print(f"{'='*80}")
    print(f"✅ Remaining contradictory names: {len(stats_after['contradictory_names'])}")
    print(f"✅ Remaining wrong meal types: {len(stats_after['wrong_meal_types'])}")
    print(f"✅ Remaining ingredient mismatches: {len(stats_after['ingredient_mismatches'])}")
    
    print(f"\n📊 Updated distributions:")
    print(f"\nDiet:")
    for diet, count in sorted(stats_after['diet_distribution'].items(), key=lambda x: -x[1])[:10]:
        print(f"  {diet}: {count}")
    
    print(f"\nMeal Type:")
    for meal, count in sorted(stats_after['meal_type_distribution'].items(), key=lambda x: -x[1]):
        print(f"  {meal}: {count}")
    
    print(f"\n{'='*80}")
    print("✅ CLEANING COMPLETE!")
    print(f"{'='*80}")
    print(f"\n📁 Files created:")
    print(f"   1. Cleaned metadata: {OUTPUT_PATH}")
    print(f"   2. Detailed report: {REPORT_PATH}")
    print(f"\n💡 Next steps:")
    print(f"   1. Review the report: cat {REPORT_PATH}")
    print(f"   2. Backup original: cp {METADATA_PATH} {METADATA_PATH}.backup")
    print(f"   3. Replace with cleaned: cp {OUTPUT_PATH} {METADATA_PATH}")
    print(f"   4. Restart your recommender system")


if __name__ == "__main__":
    main()