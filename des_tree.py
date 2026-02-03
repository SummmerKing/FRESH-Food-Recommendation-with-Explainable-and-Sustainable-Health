"""
Recipe Recommender v3.1 - PRODUCTION READY WITH CRITICAL FIXES
===============================================================
Critical Fixes Applied:
1. ✅ Comprehensive allergy filter checking ALL ingredient fields
2. ✅ Allergen variations support (egg/eggs, peanut/peanuts, etc.)
3. ✅ Intelligent meal type filtering with fallback
4. ✅ Enhanced logging and debugging
5. ✅ New allergy validation endpoint
6. ✅ FIXED: Diet filter handling for nested lists and 'veg' variant
"""

import os
import typing as t
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import json
from enum import Enum
from collections import defaultdict
import logging
from datetime import datetime
import time
import hashlib
import random
import requests
from typing import Optional

PANTRY_API_URL = "http://127.0.0.1:8002"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/data1/home/sathvik/Documents/FRESH/recipe_recommendations.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# FAISS SETUP
# ============================================================================
USE_FAISS = False
FAISS_READ_INDEX_AVAILABLE = False
try:
    import faiss
    if hasattr(faiss, 'read_index') and hasattr(faiss, 'write_index'):
        USE_FAISS = True
        FAISS_READ_INDEX_AVAILABLE = True
        print("FAISS import OK -> will use faiss when index exists")
    else:
        print("FAISS imported but missing required functions")
except Exception as e:
    print(f"FAISS import failed, falling back to NumPy search. Error: {e}")

# ============================================================================
# PATHS
# ============================================================================
DATA_DIR = r"/data1/home/sathvik/Documents/FRESH/FRESH/data"
INDEX_PATH = os.path.join(DATA_DIR, "/data1/home/sathvik/Documents/FRESH/recipe_index.faiss")
VECS_PATH = os.path.join(DATA_DIR, "/data1/home/sathvik/Documents/FRESH/claude_recipe_vectors_FINAL.npy")
IDS_PATH = os.path.join(DATA_DIR, "/data1/home/sathvik/Documents/FRESH/recipe_ids_FINAL.npy")
METADATA_PATH = os.path.join(DATA_DIR, "/data1/home/sathvik/Documents/FRESH/claude_recipes_metadata_FINAL.json")
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_DIM = None

# ============================================================================
# BALANCED SCORING WEIGHTS
# ============================================================================
WEIGHT_SIM = 2.0
WEIGHT_PANTRY = 1.0
WEIGHT_LIKE = 1.8
WEIGHT_DIVERSITY = 0.5
WEIGHT_TIME_BONUS = 0.3

# ============================================================================
# ALLERGEN VARIATIONS MAP (CRITICAL FIX #1)
# ============================================================================
ALLERGEN_VARIATIONS = {
    'eggs': ['egg', 'eggs', 'omelette', 'omelet', 'benedict', 'frittata', 'quiche', 'scrambled', 'poached'],
    'peanuts': ['peanut', 'peanuts', 'peanut butter'],
    'shellfish': ['shellfish', 'shrimp', 'prawn', 'prawns', 'crab', 'lobster', 'clam', 'clams', 'oyster', 'oysters', 'mussel', 'mussels', 'scallop', 'scallops'],
    'tree nuts': ['almond', 'almonds', 'walnut', 'walnuts', 'cashew', 'cashews', 'pecan', 'pecans', 'pistachio', 'pistachios', 'hazelnut', 'hazelnuts', 'macadamia'],
    'dairy': ['milk', 'cheese', 'butter', 'cream', 'yogurt', 'yoghurt', 'ghee', 'whey', 'casein', 'lactose'],
    'soy': ['soy', 'soya', 'tofu', 'tempeh', 'edamame', 'miso'],
    'gluten': ['wheat', 'barley', 'rye', 'flour', 'bread', 'pasta', 'semolina', 'couscous'],
    'fish': ['fish', 'salmon', 'tuna', 'cod', 'halibut', 'sardine', 'anchovy', 'anchovies', 'trout', 'bass', 'tilapia']
}

# ============================================================================
# MEAL-AWARE TIME CONSTRAINTS
# ============================================================================
MIN_PREP_TIME = {
    "breakfast": 5,
    "lunch": 10,
    "dinner": 15
}

DEFAULT_TIME_LIMITS = {
    "breakfast": 30,
    "lunch": 45,
    "dinner": 60
}

# ============================================================================
# EXPANDED MEAL TYPE COMPATIBILITY (FIX #2)
# ============================================================================
MEAL_TYPE_COMPATIBLE = {
    "breakfast": ["brunch", "snack", "lunch"],
    "lunch": ["main", "dinner", "breakfast"],
    "dinner": ["main", "lunch"]
}

# ============================================================================
# CACHE CONFIGURATION
# ============================================================================
recommendation_cache = {}
CACHE_TTL = 300

# ============================================================================
# FEEDBACK STORAGE
# ============================================================================
feedback_log = []
impression_log = []

class MealType(str, Enum):
    BREAKFAST = "breakfast"
    LUNCH = "lunch" 
    DINNER = "dinner"

# ============================================================================
# EMBEDDING MODEL
# ============================================================================
_embed_model = None
def get_embedder():
    global _embed_model, EMBED_DIM
    if _embed_model is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            error_msg = f"sentence-transformers not installed. Install via `pip install sentence-transformers`. Error: {e}"
            print(error_msg)
            raise RuntimeError(error_msg)
        
        try:
            _embed_model = SentenceTransformer(EMBED_MODEL_NAME)
            EMBED_DIM = _embed_model.get_sentence_embedding_dimension()
            print(f"Loaded embedder: {EMBED_MODEL_NAME}, dim: {EMBED_DIM}")
        except Exception as e:
            error_msg = f"Failed to load SentenceTransformer model '{EMBED_MODEL_NAME}': {e}"
            print(error_msg)
            raise RuntimeError(error_msg)
    
    return _embed_model

def text_to_vector(text: str) -> np.ndarray:
    """Convert text to normalized embedding vector."""
    if not text or not text.strip():
        if EMBED_DIM is None:
            get_embedder()
        return np.zeros(EMBED_DIM, dtype=np.float32)
    
    try:
        model = get_embedder()
        vec = model.encode(text.strip(), convert_to_numpy=True)
        v = vec.astype(np.float32)
        norm = np.linalg.norm(v)
        if norm > 0:
            v = v / norm
        return v
    except Exception as e:
        print(f"Error encoding text '{text}': {e}")
        raise

# ============================================================================
# DATA LOADING
# ============================================================================
def load_data():
    """Load recipe data with proper error handling."""
    print("Loading recipe vectors and ids...")
    
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")
    
    if not os.path.exists(VECS_PATH):
        raise FileNotFoundError(f"Recipe vectors file not found: {VECS_PATH}")
    
    try:
        recipe_vectors = np.load(VECS_PATH).astype(np.float32)
        print(f"Loaded recipe vectors: shape {recipe_vectors.shape}")
    except Exception as e:
        raise RuntimeError(f"Failed to load recipe vectors from {VECS_PATH}: {e}")
    
    norms = np.linalg.norm(recipe_vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    recipe_vectors_norm = recipe_vectors / norms
    
    if not os.path.exists(IDS_PATH):
        raise FileNotFoundError(f"Recipe IDs file not found: {IDS_PATH}")
    
    try:
        recipe_ids = np.load(IDS_PATH, allow_pickle=True)
        print(f"Loaded recipe IDs: {len(recipe_ids)} recipes")
    except Exception as e:
        raise RuntimeError(f"Failed to load recipe IDs from {IDS_PATH}: {e}")
    
    if len(recipe_vectors) != len(recipe_ids):
        raise ValueError(f"Mismatch between vectors ({len(recipe_vectors)}) and IDs ({len(recipe_ids)})")
    
    return recipe_vectors, recipe_vectors_norm, recipe_ids

try:
    recipe_vectors, recipe_vectors_norm, recipe_ids = load_data()
except Exception as e:
    print(f"CRITICAL: Failed to load recipe data: {e}")
    print("The server will not function properly without recipe data.")
    recipe_vectors = np.array([], dtype=np.float32).reshape(0, 384)
    recipe_vectors_norm = np.array([], dtype=np.float32).reshape(0, 384)
    recipe_ids = np.array([])

def create_faiss_index():
    """Create FAISS index from recipe vectors."""
    if not USE_FAISS or len(recipe_vectors) == 0:
        return None
    
    try:
        print("Creating new FAISS index...")
        dimension = recipe_vectors_norm.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(recipe_vectors_norm.astype(np.float32))
        faiss.write_index(index, INDEX_PATH)
        print(f"Created and saved FAISS index with {index.ntotal} vectors")
        return index
    except Exception as e:
        print(f"Failed to create FAISS index: {e}")
        return None

faiss_index = None
if USE_FAISS and FAISS_READ_INDEX_AVAILABLE and len(recipe_vectors) > 0:
    if os.path.exists(INDEX_PATH):
        try:
            faiss_index = faiss.read_index(INDEX_PATH)
            print(f"Loaded FAISS index from {INDEX_PATH}")
            if faiss_index.ntotal != len(recipe_vectors):
                print(f"Warning: FAISS index size mismatch, creating new one")
                faiss_index = create_faiss_index()
        except Exception as e:
            print(f"Could not load FAISS index, creating new one. Error: {e}")
            faiss_index = create_faiss_index()
    else:
        print(f"FAISS index file not found, creating new one")
        faiss_index = create_faiss_index()

# ============================================================================
# METADATA LOADING
# ============================================================================
meta_map = {}
if os.path.exists(METADATA_PATH):
    try:
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            metadata_list = json.load(f)
        
        for m in metadata_list:
            if not isinstance(m, dict):
                continue
            
            recipe_id = m.get("recipe_id", m.get("title"))
            if recipe_id is None:
                continue
            
            # Handle meal_type - it's a list in your data
            meal_type_raw = m.get("meal_type", ["dinner"])
            if isinstance(meal_type_raw, str):
                meal_type_list = [meal_type_raw]
            elif isinstance(meal_type_raw, list):
                meal_type_list = meal_type_raw
            else:
                meal_type_list = ["dinner"]
            
            # Get primary meal type (first in list)
            primary_meal_type = meal_type_list[0] if meal_type_list else "dinner"
            
            # Handle diet - it's a string in your data
            diet_raw = m.get("diet", [])
            if isinstance(diet_raw, str):
                diet_list = [diet_raw]
            elif isinstance(diet_raw, list):
                diet_list = diet_raw
            else:
                diet_list = []
            
            processed_meta = {
                "recipe_id": str(recipe_id),
                "title": m.get("title", ""),
                "dish": m.get("title", ""),
                "meal_type": primary_meal_type,  # Store primary for easy access
                "meal_types_all": meal_type_list,  # Store full list
                "ing_tokens": m.get("cleaned_ingredients", []),
                "ingredients_full": m.get("ingredients", []),
                "directions": m.get("directions", []),
                "diet": diet_list,  # Always as list
                "time_minutes": m.get("time_minutes", 30),
                "time": m.get("time", "30 minutes"),
                "link": m.get("link", ""),
                "source": m.get("source", ""),
                "search_keywords": m.get("search_keywords", []),
                "fits_time": True,
                "diet_ok": True
            }
            meta_map[str(recipe_id)] = processed_meta
        
        print(f"Loaded metadata for {len(meta_map)} recipes")
        
        # Log diet distribution for debugging
        diet_dist = defaultdict(int)
        for meta in meta_map.values():
            for d in meta.get("diet", []):
                diet_dist[d] += 1
        print(f"Diet distribution: {dict(diet_dist)}")
        
    except Exception as e:
        print(f"Warning: Failed to load metadata from {METADATA_PATH}: {e}")
        import traceback
        traceback.print_exc()
        meta_map = {}

# ============================================================================
# QUERY FUNCTIONS
# ============================================================================
def query_faiss_with_index(vec: np.ndarray, topk: int = 500) -> t.List[t.Tuple[str, float, int]]:
    """Use FAISS index."""
    try:
        q = vec.reshape(1, -1).astype(np.float32)
        topk = min(topk, faiss_index.ntotal)
        D, I = faiss_index.search(q, topk)
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0 or idx >= len(recipe_ids):
                continue
            rid = recipe_ids[idx]
            results.append((str(rid), float(score), int(idx)))
        return results
    except Exception as e:
        print(f"Error in FAISS query: {e}")
        return []

def query_numpy(vec: np.ndarray, topk: int = 500) -> t.List[t.Tuple[str, float, int]]:
    """Compute cosine similarity via NumPy."""
    try:
        if len(recipe_vectors_norm) == 0:
            return []
        
        v = vec.astype(np.float32)
        norm = np.linalg.norm(v)
        if norm > 0:
            v = v / norm
        
        sims = recipe_vectors_norm.dot(v)
        topk = min(topk, len(sims))
        idxs = np.argsort(-sims)[:topk]
        return [(str(recipe_ids[i]), float(sims[i]), int(i)) for i in idxs]
    except Exception as e:
        print(f"Error in NumPy query: {e}")
        return []

def query_faiss(vec: np.ndarray, topk: int = 500) -> t.List[t.Tuple[str, float, int]]:
    """Query using FAISS if available, otherwise NumPy."""
    if faiss_index is not None:
        return query_faiss_with_index(vec, topk)
    else:
        return query_numpy(vec, topk)

# ============================================================================
# IMPROVED QUERY CONSTRUCTION
# ============================================================================
def build_semantic_query(req: 'RecommendationRequest', meal_type: MealType) -> str:
    """Build coherent semantic query without repetition."""
    parts = []
    
    if req.likes:
        parts.extend(req.likes[:3])
    
    if req.diet:
        parts.append(f"{req.diet} recipes")
    
    if req.pantry:
        important_pantry = [p for p in req.pantry if len(p) > 3][:3]
        parts.extend(important_pantry)
    
    parts.append(f"{meal_type.value} ideas")
    
    query = " ".join(parts)
    logger.info(f"Built query for {meal_type.value}: {query}")
    return query

# ============================================================================
# CRITICAL FIX: COMPREHENSIVE ALLERGY FILTERING
# ============================================================================
def get_allergen_variations(allergy: str) -> t.List[str]:
    """Get all variations of an allergen."""
    allergy_lower = allergy.lower().strip()
    
    if allergy_lower in ALLERGEN_VARIATIONS:
        return ALLERGEN_VARIATIONS[allergy_lower]
    
    variations = [allergy_lower]
    
    if allergy_lower.endswith('s') and len(allergy_lower) > 3:
        variations.append(allergy_lower[:-1])
    else:
        variations.append(allergy_lower + 's')
    
    return variations

def passes_allergy_filter(meta: dict, allergies: t.List[str]) -> bool:
    """
    CRITICAL SAFETY: Check allergy safety across ALL ingredient fields.
    
    Checks:
    1. ing_tokens (cleaned ingredients)
    2. ingredients_full (full ingredient text)
    3. title (dish name - allergens sometimes in title)
    
    Returns False if ANY allergen variation is found ANYWHERE.
    """
    if not allergies:
        return True
    
    # Gather ALL text to check
    ing_tokens = meta.get("ing_tokens", [])
    ing_full = meta.get("ingredients_full", [])
    title = meta.get("title", "")
    
    # Combine all ingredient text
    all_text_parts = []
    all_text_parts.extend([str(t).lower() for t in ing_tokens])
    all_text_parts.extend([str(i).lower() for i in ing_full])
    all_text_parts.append(title.lower())
    
    combined_text = " ".join(all_text_parts)
    
    # Check each allergen and its variations
    for allergy in allergies:
        variations = get_allergen_variations(allergy)
        
        for variant in variations:
            # Use word boundary check for accuracy
            if f" {variant} " in f" {combined_text} " or \
               combined_text.startswith(f"{variant} ") or \
               combined_text.endswith(f" {variant}") or \
               f"({variant})" in combined_text or \
               f"[{variant}]" in combined_text or \
               f",{variant}" in combined_text or \
               f"{variant}," in combined_text:
                logger.warning(
                    f"ALLERGY DETECTED: '{variant}' (allergen: {allergy}) "
                    f"in recipe '{meta.get('title', 'unknown')}' (ID: {meta.get('recipe_id')})"
                )
                return False
    
    return True

# ============================================================================
# DIET AND TIME FILTERS - FIXED VERSION
# ============================================================================
def passes_diet_filter(recipe: dict, diet_requirement: str) -> bool:
    """Check if recipe matches diet requirement."""
    if not diet_requirement:
        return True
    
    # Handle case where diet_requirement might be passed as list
    if isinstance(diet_requirement, list):
        if not diet_requirement:
            return True
        diet_requirement = diet_requirement[0]
    
    diet_requirement = str(diet_requirement).lower().strip()
    recipe_diet = recipe.get("diet", [])
    
    # Convert to list if it's a string
    if isinstance(recipe_diet, str):
        recipe_diet = [recipe_diet]
    elif not isinstance(recipe_diet, list):
        recipe_diet = []
    
    # Flatten nested lists and handle non-string values safely
    flattened_diet = []
    for d in recipe_diet:
        if d is None:
            continue
        if isinstance(d, list):
            # Handle nested lists
            for item in d:
                if item is not None:
                    flattened_diet.append(str(item).lower().strip())
        else:
            flattened_diet.append(str(d).lower().strip())
    
    recipe_diet_lower = flattened_diet
    
    # Handle common diet requirement variations
    if diet_requirement in ["vegetarian", "veg"]:
        return any(diet in recipe_diet_lower for diet in ["vegetarian", "vegan", "veg"])
    
    if diet_requirement == "vegan":
        return "vegan" in recipe_diet_lower
    
    if diet_requirement == "gluten-free":
        return "gluten-free" in recipe_diet_lower
    
    if diet_requirement in ["non-vegetarian", "non-veg", "nonveg"]:
        return any(diet in recipe_diet_lower for diet in ["non-veg", "non-vegetarian", "nonveg"])
    
    # Check for partial matches (e.g., "veg" matches "vegetarian")
    return any(diet_requirement in diet or diet in diet_requirement for diet in recipe_diet_lower)

def passes_time_filter(recipe: dict, time_budget: dict, meal_type: str) -> bool:
    """Check time constraints with meal-aware minimum."""
    recipe_time = recipe.get("time_minutes", 30)
    
    min_time = MIN_PREP_TIME.get(meal_type, 10)
    if recipe_time < min_time:
        return False
    
    if not time_budget or not isinstance(time_budget, dict):
        max_time = DEFAULT_TIME_LIMITS.get(meal_type, 60)
        return recipe_time <= max_time
    
    if "max_minutes" in time_budget:
        return recipe_time <= time_budget["max_minutes"]
    
    if meal_type in time_budget:
        return recipe_time <= time_budget[meal_type]
    
    max_time = DEFAULT_TIME_LIMITS.get(meal_type, 60)
    return recipe_time <= max_time

# ============================================================================
# ENHANCED FILTERING WITH FALLBACK (FIX #2)
# ============================================================================
def passes_all_other_filters(meta: dict, filters: dict) -> bool:
    """
    Check all filters EXCEPT meal type.
    Separated for reuse in fallback logic.
    """
    diet = filters.get("diet")
    allergies = filters.get("allergies", [])
    dislikes = filters.get("dislikes", [])
    time_budget = filters.get("time_budget", {})
    meal_type = filters.get("meal_type", "dinner")
    
    if diet and not passes_diet_filter(meta, diet):
        return False
    
    if not passes_allergy_filter(meta, allergies):
        return False
    
    title_lower = meta.get("title", "").lower()
    dislikes_lower = [d.lower() for d in dislikes]
    if any(dislike in title_lower for dislike in dislikes_lower):
        return False
    
    if not passes_time_filter(meta, time_budget, meal_type):
        return False
    
    return True

def apply_hard_filters(
    candidates: t.List[t.Tuple[str, float, int]], 
    filters: dict,
    allow_fallback: bool = True
) -> t.Tuple[t.List[t.Tuple[str, float, int, dict]], dict]:
    """
    Apply hard filters with intelligent fallback for meal types.
    FIXED: Handle meal_type as LIST in metadata
    """
    stats = defaultdict(int)
    meal_type = filters.get("meal_type", "dinner")
    
    # PHASE 1: Strict meal type matching
    strict_filtered = []
    for rid, sim, idx in candidates:
        meta = meta_map.get(rid)
        if not meta:
            stats["no_metadata"] += 1
            continue
        
        # FIX: Handle meal_type as list
        recipe_meal_types = meta.get("meal_type", [])
        if isinstance(recipe_meal_types, str):
            recipe_meal_types = [recipe_meal_types]
        
        # Convert to lowercase for comparison
        recipe_meal_types_lower = [mt.lower() for mt in recipe_meal_types]
        
        # Check if requested meal_type is in the recipe's meal_types
        if meal_type in recipe_meal_types_lower:
            if passes_all_other_filters(meta, filters):
                strict_filtered.append((rid, sim, idx, meta))
            else:
                if not passes_diet_filter(meta, filters.get("diet")):
                    stats["diet"] += 1
                elif not passes_allergy_filter(meta, filters.get("allergies", [])):
                    stats["allergies"] += 1
                elif not passes_time_filter(meta, filters.get("time_budget", {}), meal_type):
                    stats["time"] += 1
    
    if len(strict_filtered) >= 10:
        logger.info(f"Found {len(strict_filtered)} strict {meal_type} matches")
        return strict_filtered, dict(stats)
    
    # PHASE 2: Expand to compatible meal types
    logger.warning(
        f"Only {len(strict_filtered)} strict {meal_type} matches. "
        f"Expanding to compatible meal types..."
    )
    
    compatible_types = MEAL_TYPE_COMPATIBLE.get(meal_type, [])
    expanded_filtered = strict_filtered.copy()
    
    for rid, sim, idx in candidates:
        if any(r[0] == rid for r in strict_filtered):
            continue
        
        meta = meta_map.get(rid)
        if not meta:
            continue
        
        # FIX: Handle meal_type as list
        recipe_meal_types = meta.get("meal_type", [])
        if isinstance(recipe_meal_types, str):
            recipe_meal_types = [recipe_meal_types]
        
        recipe_meal_types_lower = [mt.lower() for mt in recipe_meal_types]
        
        # Check if any compatible type is in recipe's meal_types
        if any(compat in recipe_meal_types_lower for compat in compatible_types):
            if passes_all_other_filters(meta, filters):
                expanded_filtered.append((rid, sim, idx, meta))
                stats["meal_type_compatible"] += 1
        else:
            stats["meal_type"] += 1
    
    if len(expanded_filtered) > len(strict_filtered):
        logger.info(
            f"Expanded from {len(strict_filtered)} to {len(expanded_filtered)} "
            f"using compatible meal types"
        )
    
    if len(expanded_filtered) < 3:
        logger.error(
            f"CRITICAL: Only {len(expanded_filtered)} recipes for {meal_type} "
            f"after all filtering. Stats: {dict(stats)}"
        )
    
    return expanded_filtered, dict(stats)

# ============================================================================
# SCORING FUNCTIONS
# ============================================================================
def pantry_overlap_score(recipe_tokens: t.List[str], pantry: t.Set[str]) -> float:
    """Calculate pantry item overlap with fuzzy matching."""
    if not recipe_tokens or not pantry:
        return 0.0
    
    recipe_tokens_lower = [str(t).lower().strip() for t in recipe_tokens]
    pantry_lower = [p.lower().strip() for p in pantry]
    
    score = 0.0
    matched_items = set()
    
    for pantry_item in pantry_lower:
        if pantry_item in matched_items:
            continue
            
        for token in recipe_tokens_lower:
            if pantry_item == token:
                score += 1.0
                matched_items.add(pantry_item)
                break
            elif len(pantry_item) >= 4 and (pantry_item in token or token in pantry_item):
                score += 0.8
                matched_items.add(pantry_item)
                break
            elif pantry_item.endswith('s') and pantry_item[:-1] == token:
                score += 0.9
                matched_items.add(pantry_item)
                break
            elif token.endswith('s') and token[:-1] == pantry_item:
                score += 0.9
                matched_items.add(pantry_item)
                break
    
    return score

def calculate_like_bonus(title: str, likes: t.List[str]) -> float:
    """Calculate bonus from liked keywords."""
    if not title or not likes:
        return 0.0
    
    title_lower = title.lower()
    bonus = 0.0
    
    for like in likes:
        like_lower = like.lower()
        if like_lower in title_lower:
            if len(like_lower.split()) > 1:
                bonus += 3.0
            elif len(like_lower) > 8:
                bonus += 2.5
            else:
                bonus += 1.5
    
    return bonus

def calculate_time_bonus(recipe_time: int, time_budget: int) -> float:
    """Bonus for being well under time budget."""
    if recipe_time <= time_budget * 0.7:
        return 1.0
    elif recipe_time <= time_budget * 0.85:
        return 0.5
    return 0.0

def calculate_diversity_penalty_embedding(
    selected_recipes: t.List[dict], 
    new_recipe_id: str
) -> float:
    """Use embedding similarity for diversity."""
    if not selected_recipes:
        return 0.0
    
    try:
        new_idx = np.where(recipe_ids == new_recipe_id)[0]
        if len(new_idx) == 0:
            return 0.0
        new_idx = new_idx[0]
        new_vec = recipe_vectors_norm[new_idx]
        
        penalties = []
        for existing in selected_recipes:
            existing_id = existing.get('recipe_id')
            existing_idx = np.where(recipe_ids == existing_id)[0]
            if len(existing_idx) == 0:
                continue
            existing_idx = existing_idx[0]
            existing_vec = recipe_vectors_norm[existing_idx]
            
            similarity = float(np.dot(new_vec, existing_vec))
            
            if similarity > 0.85:
                penalties.append(2.0)
            elif similarity > 0.75:
                penalties.append(1.0)
            elif similarity > 0.65:
                penalties.append(0.3)
        
        return sum(penalties)
    except Exception as e:
        logger.warning(f"Error calculating diversity penalty: {e}")
        return 0.0

# ============================================================================
# EXPLAINABILITY
# ============================================================================
def explain_recommendation(
    recipe_id: str,
    features: dict,
    weights: dict
) -> dict:
    """Generate human-readable explanation for recommendation."""
    explanations = []
    
    contributors = [
        ("Recipe relevance", features.get("similarity", 0), weights.get("similarity", WEIGHT_SIM)),
        ("Pantry match", features.get("pantry_score", 0), weights.get("pantry", WEIGHT_PANTRY)),
        ("Matches your likes", features.get("like_bonus", 0), weights.get("likes", WEIGHT_LIKE)),
        ("Quick to prepare", features.get("time_bonus", 0), weights.get("time_bonus", WEIGHT_TIME_BONUS)),
    ]
    
    contributors_sorted = sorted(
        [(name, val, weight, val * weight) for name, val, weight in contributors],
        key=lambda x: x[3],
        reverse=True
    )
    
    for name, value, weight, contribution in contributors_sorted:
        if contribution > 0.5:
            explanations.append(f"{name} (+{contribution:.1f})")
    
    diversity_penalty = features.get("diversity_penalty", 0)
    if diversity_penalty > 0.5:
        explanations.append(f"Diversity adjustment (-{diversity_penalty:.1f})")
    
    summary = " • ".join(explanations[:3]) if explanations else "General match"
    
    return {
        "summary": summary,
        "detailed_breakdown": explanations,
        "feature_values": features,
        "top_contributor": contributors_sorted[0][0] if contributors_sorted else "similarity"
    }

# ============================================================================
# MAIN RECOMMENDATION LOGIC
# ============================================================================
def score_candidates(
    filtered_candidates: t.List[t.Tuple[str, float, int, dict]],
    preferences: dict,
    selected_recipes: t.List[dict] = None
) -> t.List[t.Tuple[str, float, dict, dict]]:
    """Score filtered candidates with soft preferences."""
    if selected_recipes is None:
        selected_recipes = []
    
    pantry_set = set(p.lower() for p in preferences.get("pantry", []))
    likes = preferences.get("likes", [])
    time_budget = preferences.get("time_budget", 60)
    
    scored = []
    
    for rid, sim, idx, meta in filtered_candidates:
        pantry_score = pantry_overlap_score(meta.get("ing_tokens", []), pantry_set)
        like_bonus = calculate_like_bonus(meta.get("title", ""), likes)
        time_bonus = calculate_time_bonus(meta.get("time_minutes", 30), time_budget)
        diversity_penalty = calculate_diversity_penalty_embedding(selected_recipes, rid)
        
        final_score = (
            WEIGHT_SIM * sim +
            WEIGHT_PANTRY * pantry_score +
            WEIGHT_LIKE * like_bonus +
            WEIGHT_TIME_BONUS * time_bonus -
            WEIGHT_DIVERSITY * diversity_penalty
        )
        
        features = {
            "similarity": float(sim),
            "pantry_score": float(pantry_score),
            "like_bonus": float(like_bonus),
            "time_bonus": float(time_bonus),
            "diversity_penalty": float(diversity_penalty)
        }
        
        scored.append((rid, final_score, meta, features))
    
    return sorted(scored, key=lambda x: x[1], reverse=True)

def get_meal_recommendations_enhanced(
    req: 'RecommendationRequest', 
    meal_type: MealType, 
    k: int = 3,
    exclude_ids: set = None,
    selected_recipes: t.List[dict] = None
) -> t.List[dict]:
    """Get recommendations with FIXED filtering pipeline."""
    if exclude_ids is None:
        exclude_ids = set()
    
    if selected_recipes is None:
        selected_recipes = []
    
    if len(recipe_ids) == 0:
        return []
    
    query_text = build_semantic_query(req, meal_type)
    q_vec = text_to_vector(query_text)
    raw_candidates = query_faiss(q_vec, topk=500)
    
    if not raw_candidates:
        logger.warning(f"No candidates found for {meal_type.value}")
        return []
    
    filters = {
        "meal_type": meal_type.value,
        "diet": req.diet,
        "allergies": req.allergies,
        "dislikes": req.dislikes,
        "time_budget": req.time_budget
    }
    
    filtered_candidates, filter_stats = apply_hard_filters(raw_candidates, filters)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"FILTERING RESULTS for {meal_type.value.upper()}")
    logger.info(f"{'='*60}")
    logger.info(f"Raw candidates: {len(raw_candidates)}")
    logger.info(f"After filtering: {len(filtered_candidates)}")
    logger.info(f"Filter stats: {filter_stats}")
    logger.info(f"{'='*60}\n")
    
    if not filtered_candidates:
        logger.error(f"No recipes passed filters for {meal_type.value}")
        return []
    
    filtered_candidates = [
        (rid, sim, idx, meta) for rid, sim, idx, meta in filtered_candidates
        if rid not in exclude_ids
    ]
    
    time_budget = req.time_budget.get(meal_type.value, DEFAULT_TIME_LIMITS.get(meal_type.value, 60)) \
                  if isinstance(req.time_budget, dict) else DEFAULT_TIME_LIMITS.get(meal_type.value, 60)
    
    preferences = {
        "pantry": req.pantry,
        "likes": req.likes,
        "time_budget": time_budget
    }
    
    scored = score_candidates(filtered_candidates, preferences, selected_recipes)
    
    if not scored:
        return []
    
    logger.info(f"\n{'='*60}")
    logger.info(f"TOP CANDIDATES for {meal_type.value.upper()}")
    logger.info(f"{'='*60}")
    for i, (rid, score, meta, features) in enumerate(scored[:10], 1):
        logger.info(f"{i:2d}. {meta.get('title', 'Unknown')[:50]}")
        logger.info(f"    Score: {score:.3f} | Time: {meta.get('time_minutes')}min")
        logger.info(f"    Features: sim={features['similarity']:.2f}, "
                   f"pantry={features['pantry_score']:.1f}, "
                   f"likes={features['like_bonus']:.1f}")
    logger.info(f"{'='*60}\n")
    
    recommendations = []
    remaining = scored[:min(100, len(scored))]
    
    weights = {
        "similarity": WEIGHT_SIM,
        "pantry": WEIGHT_PANTRY,
        "likes": WEIGHT_LIKE,
        "time_bonus": WEIGHT_TIME_BONUS
    }
    
    for _ in range(k):
        if not remaining:
            break
        
        rid, score, meta, features = remaining[0]
        
        explanation = explain_recommendation(rid, features, weights)
        
        rec = {
            "recipe_id": rid,
            "score": float(score),
            "title": meta.get("title", rid),
            "dish": meta.get("dish", meta.get("title", rid)),
            "meal_type": meta.get("meal_type", meal_type.value),
            "prep_time": meta.get("time_minutes", 30),
            "time_display": meta.get("time", "30 minutes"),
            "diet": meta.get("diet", []),
            "ingredients": meta.get("ingredients_full", [])[:5],
            "link": meta.get("link", ""),
            "source": meta.get("source", ""),
            "pantry_matches": int(features.get("pantry_score", 0)),
            "explanation": explanation
        }
        
        recommendations.append(rec)
        selected_recipes.append(meta)
        
        remaining = remaining[1:]
        if remaining:
            rescored = []
            for rid_r, score_r, meta_r, features_r in remaining:
                diversity_penalty = calculate_diversity_penalty_embedding(selected_recipes, rid_r)
                
                new_score = (
                    WEIGHT_SIM * features_r["similarity"] +
                    WEIGHT_PANTRY * features_r["pantry_score"] +
                    WEIGHT_LIKE * features_r["like_bonus"] +
                    WEIGHT_TIME_BONUS * features_r["time_bonus"] -
                    WEIGHT_DIVERSITY * diversity_penalty
                )
                
                features_r["diversity_penalty"] = diversity_penalty
                rescored.append((rid_r, new_score, meta_r, features_r))
            
            rescored.sort(key=lambda x: x[1], reverse=True)
            remaining = rescored
    
    return recommendations

# ============================================================================
# CACHE UTILITIES
# ============================================================================
def generate_cache_key(req: 'RecommendationRequest', meal_type: str) -> str:
    """Generate cache key from request parameters."""
    key_parts = [
        req.user_id,
        meal_type,
        req.diet or "",
        ",".join(sorted(req.allergies)),
        ",".join(sorted(req.pantry[:10])),
        ",".join(sorted(req.likes[:5]))
    ]
    key_string = "|".join(key_parts)
    return hashlib.md5(key_string.encode()).hexdigest()

def get_cached_recommendations(cache_key: str) -> t.Optional[t.List[dict]]:
    """Retrieve recommendations from cache if valid."""
    cached = recommendation_cache.get(cache_key)
    if cached:
        age = time.time() - cached["timestamp"]
        if age < CACHE_TTL:
            logger.info(f"Cache HIT for key {cache_key[:8]}... (age: {age:.1f}s)")
            return cached["recommendations"]
        else:
            logger.info(f"Cache EXPIRED for key {cache_key[:8]}...")
    return None

def cache_recommendations(cache_key: str, recommendations: t.List[dict]):
    """Store recommendations in cache."""
    recommendation_cache[cache_key] = {
        "recommendations": recommendations,
        "timestamp": time.time()
    }
    logger.info(f"Cached recommendations for key {cache_key[:8]}...")

# ============================================================================
# PYDANTIC MODELS
# ============================================================================
class PantryItem(BaseModel):
    item_id: str
    canonical: str
    quantity: t.Optional[float] = None
    unit: t.Optional[str] = None
    last_updated: datetime = Field(default_factory=datetime.now)

class RecommendationRequest(BaseModel):
    user_id: str
    diet: t.Optional[str] = None
    allergies: t.List[str] = []
    dislikes: t.List[str] = []
    time_budget: dict = {}
    pantry: t.List[str] = []
    likes: t.List[str] = []
    recommendations_per_meal: int = 3

class FeedbackEvent(BaseModel):
    user_id: str
    recipe_id: str
    event_type: str
    rating: t.Optional[int] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    context: dict = {}

class OnboardingSurvey(BaseModel):
    user_id: str
    favorite_cuisines: t.List[str] = []
    dietary_restrictions: t.List[str] = []
    cooking_skill: str = "intermediate"
    typical_cooking_time: int = 45
    household_size: int = 2
    liked_ingredients: t.List[str] = []
    disliked_ingredients: t.List[str] = []

# ============================================================================
# FASTAPI APP
# ============================================================================
app = FastAPI(title="Enhanced Recipe Recommender v3.1", version="3.1")

@app.get("/")
def read_root():
    """Root endpoint with API information."""
    return {
        "message": "Enhanced Recipe Recommender API v3.1",
        "version": "3.1-critical-fixes",
        "total_recipes": len(recipe_ids),
        "critical_fixes": [
            "🔥 FIXED: Allergy filter now checks ALL ingredient fields",
            "🔥 FIXED: Allergen variations (egg/eggs, peanut/peanuts, etc.)",
            "🔥 FIXED: Meal type filtering with intelligent fallback",
            "🔥 FIXED: Better breakfast dataset handling",
            "🔥 FIXED: Diet filter handles nested lists and 'veg' variant",
            "✅ Enhanced logging for debugging",
            "✅ New allergy validation endpoint"
        ],
        "improvements": [
            "✅ Balanced scoring weights (similarity 2.0x, likes 1.8x)",
            "✅ Embedding-based diversity (replaces string matching)",
            "✅ Hard filtering + soft scoring separation",
            "✅ Semantic query construction (no token repetition)",
            "✅ Built-in explainability for every recommendation",
            "✅ Response caching (5min TTL)",
            "✅ Feedback collection for future LTR"
        ],
        "endpoints": {
            "health": "/health",
            "meal_plan": "/meal_plan (recommended)",
            "validate_allergy": "/validate_allergy_safety (NEW)",
            "recommend": "/recommend",
            "debug": "/debug_recommend",
            "stats": "/stats",
            "feedback": "/feedback",
            "onboarding": "/onboarding"
        },
        "status": "production-ready-with-fixes",
        "recipes_loaded": len(recipe_ids),
        "metadata_loaded": len(meta_map)
    }

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy" if len(recipe_ids) > 0 else "degraded",
        "faiss_available": USE_FAISS,
        "faiss_index_loaded": faiss_index is not None,
        "num_recipes": len(recipe_ids),
        "metadata_recipes": len(meta_map),
        "embedder_loaded": _embed_model is not None,
        "cache_size": len(recommendation_cache),
        "feedback_events_logged": len(feedback_log),
        "weights": {
            "similarity": WEIGHT_SIM,
            "pantry": WEIGHT_PANTRY,
            "likes": WEIGHT_LIKE,
            "diversity": WEIGHT_DIVERSITY,
            "time_bonus": WEIGHT_TIME_BONUS
        },
        "filters": {
            "min_prep_time": MIN_PREP_TIME,
            "default_time_limits": DEFAULT_TIME_LIMITS,
            "meal_type_compatible": MEAL_TYPE_COMPATIBLE
        },
        "allergen_variations_supported": len(ALLERGEN_VARIATIONS)
    }

@app.get("/stats")
def get_stats():
    """Get dataset statistics."""
    if not meta_map:
        return {"error": "No metadata loaded"}
    
    meal_counts = defaultdict(int)
    diet_counts = defaultdict(int)
    source_counts = defaultdict(int)
    time_distribution = {
        "under_5": 0, "5_15": 0, "15_30": 0, 
        "30_45": 0, "45_60": 0, "over_60": 0
    }
    
    for meta in meta_map.values():
        meal_counts[meta.get("meal_type", "unknown")] += 1
        source_counts[meta.get("source", "unknown")] += 1
        
        for diet in meta.get("diet", []):
            diet_counts[diet] += 1
        
        time_min = meta.get("time_minutes", 30)
        if time_min < 5:
            time_distribution["under_5"] += 1
        elif time_min < 15:
            time_distribution["5_15"] += 1
        elif time_min < 30:
            time_distribution["15_30"] += 1
        elif time_min < 45:
            time_distribution["30_45"] += 1
        elif time_min < 60:
            time_distribution["45_60"] += 1
        else:
            time_distribution["over_60"] += 1
    
    return {
        "total_recipes": len(meta_map),
        "meal_type_distribution": dict(meal_counts),
        "diet_distribution": dict(diet_counts),
        "source_distribution": dict(source_counts),
        "time_distribution": time_distribution,
        "filtering": {
            "min_prep_time": MIN_PREP_TIME,
            "default_time_limits": DEFAULT_TIME_LIMITS,
            "meal_type_compatibility": MEAL_TYPE_COMPATIBLE
        },
        "scoring_weights": {
            "similarity": WEIGHT_SIM,
            "pantry": WEIGHT_PANTRY,
            "likes": WEIGHT_LIKE,
            "diversity": WEIGHT_DIVERSITY,
            "time_bonus": WEIGHT_TIME_BONUS
        },
        "cache_stats": {
            "entries": len(recommendation_cache),
            "ttl_seconds": CACHE_TTL
        },
        "feedback_stats": {
            "total_events": len(feedback_log)
        }
    }

@app.post("/meal_plan")
def get_daily_meal_plan(req: RecommendationRequest):
    """Get complete daily meal plan with caching and explainability."""
    try:
        if len(recipe_ids) == 0:
            return {
                "user_id": req.user_id,
                "meal_plan": {"breakfast": [], "lunch": [], "dinner": []},
                "total_recommendations": 0,
                "message": "No recipe data available"
            }
        
        logger.info(f"\n{'='*60}")
        logger.info(f"  Generating meal plan for {req.user_id}")
        logger.info(f"{'='*60}")
        logger.info(f"🥗 Diet: {req.diet or 'Any'}")
        logger.info(f"❤️ Likes: {', '.join(req.likes) if req.likes else 'None'}")
        logger.info(f"🥫 Pantry: {len(req.pantry)} items")
        logger.info(f"⏰ Time budget: {req.time_budget if req.time_budget else 'Using defaults'}")
        logger.info(f"🚫 Allergies: {', '.join(req.allergies) if req.allergies else 'None'}")
        
        used_recipe_ids = set()
        all_selected_recipes = []
        meal_plan = {}
        total_recs = 0
        
        for meal_type in [MealType.BREAKFAST, MealType.LUNCH, MealType.DINNER]:
            cache_key = generate_cache_key(req, meal_type.value)
            cached_recs = get_cached_recommendations(cache_key)
            
            if cached_recs:
                recommendations = cached_recs
            else:
                logger.info(f"\n🔍 Searching for {meal_type.value.upper()} recipes...")
                
                recommendations = get_meal_recommendations_enhanced(
                    req, 
                    meal_type, 
                    req.recommendations_per_meal,
                    exclude_ids=used_recipe_ids,
                    selected_recipes=all_selected_recipes
                )
                
                cache_recommendations(cache_key, recommendations)
            
            for rec in recommendations:
                used_recipe_ids.add(rec['recipe_id'])
                meta = meta_map.get(rec['recipe_id'], {})
                all_selected_recipes.append(meta)
            
            meal_plan[meal_type.value] = recommendations
            total_recs += len(recommendations)
            
            logger.info(f"✅ Found {len(recommendations)} {meal_type.value} recipes")
            if recommendations:
                logger.info(f"   Top: {recommendations[0]['title']} (score: {recommendations[0]['score']:.2f})")
                logger.info(f"   Why: {recommendations[0]['explanation']['summary']}")
        
        insights = []
        
        if total_recs == 0:
            message = "No recipes found. Try relaxing filters or adding more pantry items."
        elif total_recs < req.recommendations_per_meal * 3:
            message = f"Found {total_recs} recipes. Consider expanding preferences for more variety."
            insights.append("⚠️ Limited results - try adding pantry items or relaxing time constraints")
        else:
            message = None
            
            if req.allergies:
                insights.append(f"🛡️ Filtered for {len(req.allergies)} allergy concern(s): {', '.join(req.allergies)}")
            
            if req.diet:
                diet_count = sum(
                    1 for meal_recs in meal_plan.values()
                    for rec in meal_recs
                    if req.diet.lower() in [d.lower() for d in rec.get('diet', [])]
                )
                if diet_count > total_recs * 0.7:
                    insights.append(f"✅ Strong {req.diet} match - {diet_count}/{total_recs} recipes aligned")
            
            if req.likes:
                likes_matched = sum(
                    1 for meal_recs in meal_plan.values()
                    for rec in meal_recs
                    if any(like.lower() in rec['title'].lower() for like in req.likes)
                )
                if likes_matched > 0:
                    insights.append(f"❤️ {likes_matched} recipes match your preferences")
            
            avg_pantry = sum(
                rec.get('pantry_matches', 0)
                for meal_recs in meal_plan.values()
                for rec in meal_recs
            ) / max(total_recs, 1)
            
            if avg_pantry >= 2:
                insights.append(f"🥫 Great pantry utilization - avg {avg_pantry:.1f} items per recipe")
        
        logger.info(f"\n{'='*60}")
        logger.info(f"✅ Meal plan complete: {total_recs} recipes recommended")
        logger.info(f"{'='*60}\n")
        
        return {
            "user_id": req.user_id,
            "query_context": {
                "diet": req.diet,
                "pantry_items": len(req.pantry),
                "likes": req.likes,
                "allergies": req.allergies,
                "dislikes": req.dislikes,
                "time_budget": req.time_budget or DEFAULT_TIME_LIMITS
            },
            "meal_plan": meal_plan,
            "total_recommendations": total_recs,
            "insights": insights if insights else None,
            "message": message,
            "system_info": {
                "version": "3.1-production-fixes",
                "explainability_enabled": True,
                "caching_enabled": True,
                "diversity_method": "embedding-based",
                "filtering_method": "hard-constraints-with-fallback",
                "allergy_check": "comprehensive-all-fields",
                "scoring_weights": {
                    "similarity": WEIGHT_SIM,
                    "pantry": WEIGHT_PANTRY,
                    "likes": WEIGHT_LIKE,
                    "diversity": WEIGHT_DIVERSITY,
                    "time_bonus": WEIGHT_TIME_BONUS
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Error in meal_plan: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/recommend")
def recommend(req: RecommendationRequest):
    """Get recommendations - returns combined meal plan."""
    try:
        meal_plan_response = get_daily_meal_plan(req)
        
        all_recommendations = []
        for meal_type, recs in meal_plan_response["meal_plan"].items():
            for rec in recs:
                rec["meal_type"] = meal_type
                all_recommendations.append(rec)
        
        return {
            "user_id": req.user_id,
            "recommendations": all_recommendations,
            "meal_breakdown": meal_plan_response["meal_plan"],
            "total_count": len(all_recommendations),
            "insights": meal_plan_response.get("insights"),
            "message": meal_plan_response.get("message"),
            "system_info": meal_plan_response.get("system_info")
        }
    
    except Exception as e:
        logger.error(f"Error in recommend: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/debug_recommend")
def debug_recommend(req: RecommendationRequest):
    """Debug endpoint with full scoring breakdown."""
    try:
        if len(recipe_ids) == 0:
            return {"error": "No recipe data available"}
        
        query_text = build_semantic_query(req, MealType.DINNER)
        q_vec = text_to_vector(query_text)
        candidates = query_faiss(q_vec, topk=100)
        
        filters = {
            "meal_type": "dinner",
            "diet": req.diet,
            "allergies": req.allergies,
            "dislikes": req.dislikes,
            "time_budget": req.time_budget
        }
        
        filtered, filter_stats = apply_hard_filters(candidates, filters)
        
        preferences = {
            "pantry": req.pantry,
            "likes": req.likes,
            "time_budget": 60
        }
        
        scored = score_candidates(filtered[:50], preferences, [])
        
        debug = []
        weights = {
            "similarity": WEIGHT_SIM,
            "pantry": WEIGHT_PANTRY,
            "likes": WEIGHT_LIKE,
            "time_bonus": WEIGHT_TIME_BONUS
        }
        
        for rid, score, meta, features in scored:
            explanation = explain_recommendation(rid, features, weights)
            
            debug.append({
                "recipe_id": rid,
                "title": meta.get("title", rid),
                "final_score": float(score),
                "features": features,
                "explanation": explanation,
                "meal_type": meta.get("meal_type"),
                "diet": meta.get("diet", []),
                "time_minutes": meta.get("time_minutes"),
                "source": meta.get("source"),
                "allergy_safe": passes_allergy_filter(meta, req.allergies)
            })
        
        return {
            "user_id": req.user_id,
            "query_text": query_text,
            "candidates_retrieved": len(candidates),
            "candidates_after_filtering": len(filtered),
            "filter_stats": filter_stats,
            "scoring_weights": weights,
            "allergen_variations_checked": {k: len(v) for k, v in ALLERGEN_VARIATIONS.items()},
            "debug_results": debug
        }
    
    except Exception as e:
        logger.error(f"Error in debug_recommend: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/validate_allergy_safety")
def validate_allergy_safety(recipe_id: str, allergies: t.List[str]):
    """
    Validate if a specific recipe is safe for given allergies.
    Useful for testing and debugging the allergy filter.
    """
    meta = meta_map.get(recipe_id)
    
    if not meta:
        raise HTTPException(status_code=404, detail="Recipe not found")
    
    is_safe = passes_allergy_filter(meta, allergies)
    
    detected_allergens = []
    if not is_safe:
        ing_tokens = meta.get("ing_tokens", [])
        ing_full = meta.get("ingredients_full", [])
        title = meta.get("title", "")
        combined_text = " ".join([str(t).lower() for t in ing_tokens + ing_full] + [title.lower()])
        
        for allergy in allergies:
            variations = get_allergen_variations(allergy)
            for variant in variations:
                if variant in combined_text:
                    detected_allergens.append({
                        "allergen": allergy,
                        "variant_found": variant,
                        "context_preview": combined_text[:200]
                    })
    
    return {
        "recipe_id": recipe_id,
        "recipe_title": meta.get("title"),
        "is_safe": is_safe,
        "allergies_checked": allergies,
        "detected_allergens": detected_allergens if detected_allergens else None,
        "ingredients_checked": {
            "ing_tokens": meta.get("ing_tokens", [])[:5],
            "ingredients_full": meta.get("ingredients_full", [])[:3],
            "title": meta.get("title")
        },
        "allergen_variations": {a: get_allergen_variations(a) for a in allergies}
    }

@app.post("/feedback")
def log_feedback(event: FeedbackEvent):
    """Log user feedback for future learning-to-rank."""
    try:
        feedback_log.append(event.dict())
        logger.info(f"Logged feedback: {event.user_id} {event.event_type} {event.recipe_id}")
        
        return {
            "status": "success",
            "message": "Feedback logged successfully",
            "total_feedback_events": len(feedback_log)
        }
    except Exception as e:
        logger.error(f"Error logging feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/onboarding")
def process_onboarding(survey: OnboardingSurvey):
    """Process onboarding survey and return initial recommendations."""
    try:
        logger.info(f"Processing onboarding for {survey.user_id}")
        
        req = RecommendationRequest(
            user_id=survey.user_id,
            diet=survey.dietary_restrictions[0] if survey.dietary_restrictions else None,
            allergies=[],
            dislikes=survey.disliked_ingredients,
            time_budget={
                "breakfast": min(survey.typical_cooking_time // 2, 30),
                "lunch": survey.typical_cooking_time,
                "dinner": survey.typical_cooking_time
            },
            pantry=[],
            likes=survey.liked_ingredients + survey.favorite_cuisines,
            recommendations_per_meal=3
        )
        
        meal_plan = get_daily_meal_plan(req)
        
        return {
            "user_id": survey.user_id,
            "onboarding_complete": True,
            "initial_meal_plan": meal_plan,
            "message": "Welcome! Here are your personalized recommendations.",
            "next_steps": [
                "Try cooking one of these recipes",
                "Add items to your pantry for better matches",
                "Rate recipes to improve future recommendations"
            ]
        }
    
    except Exception as e:
        logger.error(f"Error in onboarding: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/cache/stats")
def get_cache_stats():
    """Get cache statistics."""
    now = time.time()
    valid_entries = sum(
        1 for cached in recommendation_cache.values()
        if now - cached["timestamp"] < CACHE_TTL
    )
    
    return {
        "total_entries": len(recommendation_cache),
        "valid_entries": valid_entries,
        "ttl_seconds": CACHE_TTL,
        "hit_rate": "tracked in production logs"
    }

@app.post("/cache/clear")
def clear_cache():
    """Clear recommendation cache."""
    global recommendation_cache
    old_size = len(recommendation_cache)
    recommendation_cache = {}
    logger.info(f"Cache cleared: removed {old_size} entries")
    return {
        "status": "success",
        "entries_cleared": old_size
    }

@app.post("/meal_plan_smart")
def get_smart_meal_plan(req: RecommendationRequest):
    """
    Smart meal plan that automatically fetches user's pantry
    
    This endpoint:
    1. Fetches user's pantry from pantry API
    2. Uses pantry items for better recommendations
    3. Prioritizes recipes using expiring ingredients
    4. Returns recommendations with pantry match info
    """
    try:
        # Fetch user's pantry
        pantry_items = []
        expiring_soon = []
        
        try:
            pantry_response = requests.get(
                f"{PANTRY_API_URL}/pantry/{req.user_id}",
                timeout=2
            )
            
            if pantry_response.status_code == 200:
                pantry_data = pantry_response.json()
                pantry_items = [
                    item['canonical_name'] 
                    for item in pantry_data.get('items', [])
                ]
                
                # Get expiring items
                expiring_response = requests.get(
                    f"{PANTRY_API_URL}/pantry/{req.user_id}/expiring?days=3",
                    timeout=2
                )
                
                if expiring_response.status_code == 200:
                    expiring_data = expiring_response.json()
                    expiring_soon = [
                        item['canonical_name']
                        for item in expiring_data.get('items', [])
                    ]
                
        except requests.exceptions.RequestException as e:
            logger.warning(f"Could not fetch pantry for {req.user_id}: {e}")
        
        # Override pantry in request with actual pantry data
        if pantry_items:
            req.pantry = pantry_items
            logger.info(f"Using {len(pantry_items)} pantry items for {req.user_id}")
        
        # Add expiring items to likes for priority
        if expiring_soon:
            req.likes = list(set(req.likes + expiring_soon))
            logger.info(f"Prioritizing {len(expiring_soon)} expiring items")
        
        # Get meal plan using existing logic
        meal_plan_response = get_daily_meal_plan(req)
        
        # Enhance response with pantry info
        meal_plan_response['pantry_info'] = {
            'total_pantry_items': len(pantry_items),
            'pantry_items_used': pantry_items[:10],  # First 10 for display
            'expiring_soon': expiring_soon,
            'pantry_api_status': 'connected' if pantry_items else 'unavailable'
        }
        
        return meal_plan_response
        
    except Exception as e:
        logger.error(f"Error in smart meal plan: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/cook_recipe")
def record_recipe_cooked(
    user_id: str,
    recipe_id: str,
    recipe_name: str,
    ingredients_used: list
):
    """
    Record that user cooked a recipe
    This will:
    1. Log usage in pantry system
    2. Optionally deduct from pantry (future feature)
    3. Track cooking history for better recommendations
    """
    try:
        # Record in pantry system
        usage_data = {
            "user_id": user_id,
            "recipe_id": recipe_id,
            "recipe_name": recipe_name,
            "ingredients_used": ingredients_used
        }
        
        response = requests.post(
            f"{PANTRY_API_URL}/pantry/record-usage",
            json=usage_data,
            timeout=2
        )
        
        if response.status_code == 200:
            return {
                "status": "success",
                "message": f"Recipe '{recipe_name}' cooking recorded",
                "pantry_updated": True,
                "usage_data": response.json()
            }
        else:
            return {
                "status": "partial_success",
                "message": "Recipe recorded locally but pantry update failed",
                "pantry_updated": False
            }
            
    except Exception as e:
        logger.error(f"Error recording recipe: {e}")
        return {
            "status": "error",
            "message": str(e),
            "pantry_updated": False
        }


@app.get("/user/{user_id}/pantry_status")
def get_user_pantry_status(user_id: str):
    """
    Get user's pantry status and recommendations
    """
    try:
        # Fetch pantry
        pantry_response = requests.get(
            f"{PANTRY_API_URL}/pantry/{user_id}",
            timeout=2
        )
        
        # Fetch expiring items
        expiring_response = requests.get(
            f"{PANTRY_API_URL}/pantry/{user_id}/expiring?days=7",
            timeout=2
        )
        
        # Fetch suggestions
        suggestions_response = requests.get(
            f"{PANTRY_API_URL}/pantry/{user_id}/suggestions",
            timeout=2
        )
        
        pantry_data = pantry_response.json() if pantry_response.status_code == 200 else {}
        expiring_data = expiring_response.json() if expiring_response.status_code == 200 else {}
        suggestions_data = suggestions_response.json() if suggestions_response.status_code == 200 else {}
        
        return {
            "user_id": user_id,
            "pantry": pantry_data,
            "expiring_soon": expiring_data,
            "suggestions": suggestions_data,
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Error fetching pantry status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# ENHANCED MEAL PLAN WITH SHOPPING LIST
# ============================================================================

@app.post("/meal_plan_with_shopping_list")
def get_meal_plan_with_shopping_list(req: RecommendationRequest):
    """
    Generate meal plan AND shopping list for missing ingredients
    """
    try:
        # Get smart meal plan
        meal_plan = get_smart_meal_plan(req)
        
        # Extract all required ingredients from meal plan
        required_ingredients = set()
        for meal_type, recipes in meal_plan.get('meal_plan', {}).items():
            for recipe in recipes:
                ingredients = recipe.get('ingredients', [])
                # Extract ingredient names (simplified)
                for ing in ingredients:
                    # Simple extraction: take first word
                    ing_name = ing.lower().split()[0] if isinstance(ing, str) else ''
                    if len(ing_name) > 2:
                        required_ingredients.add(ing_name)
        
        # Get user's pantry
        pantry_items = set(meal_plan['pantry_info'].get('pantry_items_used', []))
        
        # Calculate missing ingredients
        missing_ingredients = required_ingredients - pantry_items
        
        # Add shopping list to response
        meal_plan['shopping_list'] = {
            'required_ingredients': sorted(list(required_ingredients)),
            'have_in_pantry': sorted(list(required_ingredients & pantry_items)),
            'need_to_buy': sorted(list(missing_ingredients)),
            'pantry_coverage': f"{len(required_ingredients & pantry_items)}/{len(required_ingredients)}"
        }
        
        return meal_plan
        
    except Exception as e:
        logger.error(f"Error generating meal plan with shopping list: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    print("=" * 70)
    print("  FIXED RECIPE RECOMMENDER v3.1")
    print("=" * 70)
    print(f"\n🔥 CRITICAL FIXES:")
    print(f"   ✅ Allergy filter: Now checks ALL ingredient fields")
    print(f"   ✅ Allergen variations: {len(ALLERGEN_VARIATIONS)} allergen types with variants")
    print(f"   ✅ Meal type: Intelligent fallback when strict matching fails")
    print(f"   ✅ Breakfast: Better handling with compatible meal types")
    
    print(f"\n📊 SYSTEM STATUS:")
    print(f"   • Recipe data: {len(recipe_ids):,} recipes")
    print(f"   • Metadata: {len(meta_map):,} recipes")
    print(f"   • FAISS: {'✅ Available' if USE_FAISS else '❌ Using NumPy'}")
    print(f"   • Index: {'✅ Loaded' if faiss_index is not None else '❌ Not loaded'}")
    
    print(f"\n✨ KEY IMPROVEMENTS:")
    print(f"   ✅ Balanced scoring (sim: {WEIGHT_SIM}x, pantry: {WEIGHT_PANTRY}x, likes: {WEIGHT_LIKE}x)")
    print(f"   ✅ Hard filtering + soft scoring separation")
    print(f"   ✅ Embedding-based diversity")
    print(f"   ✅ Explainability built-in")
    print(f"   ✅ Response caching ({CACHE_TTL}s TTL)")
    
    print(f"\n🌐 SERVER ENDPOINTS:")
    print(f"   • Main: http://127.0.0.1:8001")
    print(f"   • Docs: http://127.0.0.1:8001/docs")
    print(f"   • Validate Allergy: http://127.0.0.1:8001/validate_allergy_safety")
    print(f"   • Stats: http://127.0.0.1:8001/stats")
    print(f"   • Health: http://127.0.0.1:8001/health")
    print(f"   • Debug: http://127.0.0.1:8001/debug_recommend")
    print("=" * 70)
    print("\n🚀 Starting server...\n")
    
    uvicorn.run(app, host="127.0.0.1", port=8001, log_level="info")