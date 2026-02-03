"""
Extract fresh metadata from full_dataset.csv using the existing NER column
This creates a clean recipes_metadata_FINAL.json file
"""

import pandas as pd
import json
import ast
from tqdm import tqdm
import re

def clean_text(text):
    """Clean text fields"""
    if pd.isna(text) or text == '':
        return None
    return str(text).strip()

def parse_list_column(value):
    """Safely parse list-like columns"""
    if pd.isna(value) or value == '':
        return []
    
    # If already a list
    if isinstance(value, list):
        return [str(item).strip() for item in value if item]
    
    # Try to parse as string representation of list
    try:
        parsed = ast.literal_eval(str(value))
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if item]
    except:
        pass
    
    # Fallback: split by common delimiters
    if isinstance(value, str):
        return [item.strip() for item in re.split(r'[,;\n]', value) if item.strip()]
    
    return []

def extract_metadata_from_csv(csv_path='/data1/home/sathvik/Documents/FRESH/Metadata/recipes_metadata_FINAL_completed.json', output_path='/data1/home/sathvik/Documents/FRESH/Metadata/Claude_metadata_FINAL.json'):
    """
    Extract metadata from CSV with NER column
    """
    print(f"Loading CSV from {csv_path}...")
    
    # Read CSV in chunks due to size (2.14GB)
    chunk_size = 10000
    recipes = []
    
    total_rows = sum(1 for _ in open(csv_path)) - 1  # Subtract header
    print(f"Total recipes to process: {total_rows:,}")
    
    with tqdm(total=total_rows, desc="Processing recipes") as pbar:
        for chunk in pd.read_csv(csv_path, chunksize=chunk_size, low_memory=False):
            for idx, row in chunk.iterrows():
                try:
                    # Use the NER column as cleaned_ingredients
                    cleaned_ingredients = parse_list_column(row.get('NER', []))
                    
                    # Skip recipes with no ingredients
                    if not cleaned_ingredients:
                        pbar.update(1)
                        continue
                    
                    # Parse ingredients list
                    ingredients = parse_list_column(row.get('ingredients', []))
                    
                    # Parse directions
                    directions_raw = row.get('directions', '')
                    if isinstance(directions_raw, str):
                        # Split by newlines or periods for individual steps
                        directions = [d.strip() for d in re.split(r'[\n.]', directions_raw) if d.strip()]
                    else:
                        directions = parse_list_column(directions_raw)
                    
                    # Build recipe metadata
                    recipe = {
                        'recipe_id': str(row.get('Unnamed: 0', idx)),
                        'title': clean_text(row.get('title', 'Untitled Recipe')),
                        'cleaned_ingredients': cleaned_ingredients,
                        'ingredients': ingredients,
                        'directions': directions,
                        'link': clean_text(row.get('link', '')),
                        'source': clean_text(row.get('source', 'Unknown'))
                    }
                    
                    # Optional fields
                    if 'dish' in row and pd.notna(row['dish']):
                        recipe['dish'] = clean_text(row['dish'])
                    
                    if 'meal_type' in row and pd.notna(row['meal_type']):
                        recipe['meal_type'] = clean_text(row['meal_type'])
                    
                    if 'time_minutes' in row and pd.notna(row['time_minutes']):
                        try:
                            recipe['time_minutes'] = int(float(row['time_minutes']))
                        except:
                            pass
                    
                    if 'diet' in row:
                        diet = parse_list_column(row['diet'])
                        if diet:
                            recipe['diet'] = diet
                    
                    # Create search keywords from title and ingredients
                    keywords = set()
                    if recipe['title']:
                        keywords.update(recipe['title'].lower().split())
                    keywords.update([ing.lower() for ing in cleaned_ingredients])
                    if 'meal_type' in recipe:
                        keywords.add(recipe['meal_type'].lower())
                    if 'diet' in recipe:
                        keywords.update([d.lower() for d in recipe['diet']])
                    
                    recipe['search_keywords'] = sorted(list(keywords))
                    
                    recipes.append(recipe)
                    
                except Exception as e:
                    print(f"\nError processing row {idx}: {str(e)}")
                    continue
                
                pbar.update(1)
    
    print(f"\nSuccessfully processed {len(recipes):,} recipes")
    
    # Save metadata
    print(f"Saving metadata to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(recipes, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Metadata saved: {output_path}")
    print(f"✓ Total recipes: {len(recipes):,}")
    
    # Print sample
    if recipes:
        print("\nSample recipe:")
        print(json.dumps(recipes[0], indent=2))
    
    return recipes

if __name__ == "__main__":
    extract_metadata_from_csv()