import os
import typing as t
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import json
from enum import Enum
from collections import defaultdict
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/data1/home/sathvik/Documents/FRESH/recipe_recommendations.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

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

DATA_DIR = r"/data1/home/sathvik/Documents/FRESH/FRESH/data"
INDEX_PATH = os.path.join(DATA_DIR, "/data1/home/sathvik/Documents/FRESH/recipe_index.faiss")
VECS_PATH = os.path.join(DATA_DIR, "/data1/home/sathvik/Documents/FRESH/vectors/recipe_vectors_FINAL.npy")
IDS_PATH = os.path.join(DATA_DIR, "/data1/home/sathvik/Documents/FRESH/vectors/recipe_ids_FINAL.npy")
METADATA_PATH = os.path.join(DATA_DIR, "/data1/home/sathvik/Documents/FRESH/Metadata/recipes_metadata_FINAL_completed.json")
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_DIM = None

# FIXED: Enhanced scoring weights with stricter meal type matching
WEIGHT_SIM = 1.0
WEIGHT_PANTRY = 0.8  # Increased from 0.6
WEIGHT_LIKE = 1.5  # Increased from 1.2
WEIGHT_MEAL_TYPE = 4.0  # Increased from 2.5 - CRITICAL FIX
WEIGHT_DIVERSITY = 1.0  # Increased from 0.8
WEIGHT_ALLERGY_PENALTY = 5.0  # Increased from 3.0
WEIGHT_TIGHTENING = 1.0  # Increased from 0.5
WEIGHT_TIME_PENALTY = 3.0  # NEW: Penalize time violations

# Time constraints - STRICTER
DEFAULT_TIME_LIMITS = {
    "breakfast": 30,
    "lunch": 45,
    "dinner": 60
}

# MINIMUM prep time threshold - filters unrealistic recipes
MIN_PREP_TIME = 10  # NEW: Filter recipes under 10 minutes as unrealistic

class MealType(str, Enum):
    BREAKFAST = "breakfast"
    LUNCH = "lunch" 
    DINNER = "dinner"

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
            
            processed_meta = {
                "recipe_id": str(recipe_id),
                "title": m.get("title", ""),
                "dish": m.get("title", ""),
                "meal_type": m.get("meal_type", "dinner"),
                "ing_tokens": m.get("cleaned_ingredients", m.get("ner", [])),
                "ingredients_full": m.get("ingredients", []),
                "directions": m.get("directions", []),
                "diet": m.get("diet", []),
                "time_minutes": m.get("time_minutes", 30),
                "time": m.get("time", "30 minutes"),
                "link": m.get("link", ""),
                "source": m.get("source", ""),
                "fits_time": True,
                "diet_ok": True
            }
            meta_map[str(recipe_id)] = processed_meta
        
        print(f"Loaded metadata for {len(meta_map)} recipes")
        
    except Exception as e:
        print(f"Warning: Failed to load metadata from {METADATA_PATH}: {e}")
        meta_map = {}

if not meta_map and len(recipe_ids) > 0:
    meta_map = {
        str(recipe_ids[i]): {
            "recipe_id": str(recipe_ids[i]),
            "title": str(recipe_ids[i]), 
            "dish": str(recipe_ids[i]),
            "ing_tokens": [], 
            "fits_time": True, 
            "diet_ok": True,
            "meal_type": "dinner",
            "diet": []
        } for i in range(len(recipe_ids))
    }

def query_faiss_with_index(vec: np.ndarray, topk: int = 200) -> t.List[t.Tuple[str, float, int]]:
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

def query_numpy(vec: np.ndarray, topk: int = 200) -> t.List[t.Tuple[str, float, int]]:
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

def query_faiss(vec: np.ndarray, topk: int = 200) -> t.List[t.Tuple[str, float, int]]:
    """Query using FAISS if available, otherwise NumPy."""
    if faiss_index is not None:
        return query_faiss_with_index(vec, topk)
    else:
        return query_numpy(vec, topk)

def get_recipe_signature(meta: dict) -> str:
    """Get a signature for recipe similarity."""
    title_lower = meta.get("title", "").lower()
    
    key_words = ["paneer", "palak", "biryani", "curry", "roti", "dosa", "idli", 
                 "sambar", "dal", "rice", "paratha", "naan", "tikka", "korma",
                 "chapati", "vada", "uttapam", "upma"]
    
    signature_parts = []
    for word in key_words:
        if word in title_lower:
            signature_parts.append(word)
    
    return "-".join(sorted(signature_parts)) if signature_parts else title_lower[:20]

def calculate_diversity_penalty(selected_recipes: t.List[dict], new_recipe: dict) -> float:
    """ENHANCED: Calculate penalty for recipe similarity with stricter matching."""
    if not selected_recipes:
        return 0.0
    
    new_sig = get_recipe_signature(new_recipe)
    new_title = new_recipe.get("title", "").lower()
    
    penalty = 0.0
    for existing in selected_recipes:
        existing_sig = get_recipe_signature(existing)
        existing_title = existing.get("title", "").lower()
        
        # Exact signature match - heavy penalty
        if new_sig == existing_sig and new_sig:
            penalty += 3.0  # Increased from 2.0
        
        # Very similar titles - medium penalty
        new_words = set(new_title.split())
        existing_words = set(existing_title.split())
        common_words = new_words & existing_words
        if len(common_words) >= 3:
            penalty += 1.5
        
        # Partial signature overlap
        elif new_sig and existing_sig:
            new_parts = set(new_sig.split("-"))
            existing_parts = set(existing_sig.split("-"))
            overlap = len(new_parts & existing_parts)
            if overlap > 0:
                penalty += 0.7 * overlap  # Increased from 0.5
    
    return penalty

def calculate_allergy_penalty(ingredients: t.List[str], allergies: t.List[str]) -> float:
    """STRICT: Calculate penalty for potential allergy concerns."""
    if not allergies:
        return 0.0
    
    ingredients_text = " ".join(str(ing).lower() for ing in ingredients)
    penalty = 0.0
    
    for allergy in allergies:
        allergy_lower = allergy.lower().strip()
        
        # Direct match - severe penalty
        if allergy_lower in ingredients_text:
            penalty += 10.0  # Increased from 5.0
        
        # Partial match - medium penalty
        elif len(allergy_lower) > 3:
            words = ingredients_text.split()
            for word in words:
                if allergy_lower in word or word in allergy_lower:
                    penalty += 5.0  # Increased from 2.0
                    break
    
    return penalty

def calculate_time_penalty(recipe_time: int, time_budget: int, meal_type: str) -> float:
    """NEW: Penalize recipes that exceed time budget."""
    if recipe_time <= time_budget:
        return 0.0
    
    # Calculate how much over budget
    overage = recipe_time - time_budget
    
    # Progressive penalty
    if overage <= 5:
        return 1.0  # Slight overage
    elif overage <= 10:
        return 2.5  # Moderate overage
    else:
        return 5.0  # Significant overage

def pantry_overlap_score(recipe_tokens: t.List[str], pantry: t.Set[str]) -> int:
    """ENHANCED: Calculate pantry item overlap with better matching."""
    if not recipe_tokens or not pantry:
        return 0
    
    recipe_tokens_lower = [str(t).lower().strip() for t in recipe_tokens]
    pantry_lower = [p.lower().strip() for p in pantry]
    
    score = 0
    matched_items = set()
    
    for pantry_item in pantry_lower:
        if pantry_item in matched_items:
            continue
            
        for token in recipe_tokens_lower:
            # Exact match or substring match
            if len(pantry_item) >= 3 and (pantry_item in token or token in pantry_item):
                score += 1
                matched_items.add(pantry_item)
                break
            # Special case: plural/singular variations
            if pantry_item.endswith('s') and pantry_item[:-1] in token:
                score += 1
                matched_items.add(pantry_item)
                break
    
    return score

def like_bonus_from_title(title: str, likes: t.List[str]) -> float:
    """ENHANCED: Calculate bonus from liked keywords with better weighting."""
    if not title or not likes:
        return 0.0
    
    title_lower = title.lower()
    bonus = 0.0
    
    for like in likes:
        like_lower = like.lower()
        if like_lower in title_lower:
            # Multi-word phrases get highest bonus
            if len(like_lower.split()) > 1:
                bonus += 4.0  # Increased from 3.0
            # Long single words get high bonus
            elif len(like_lower) > 8:
                bonus += 3.0  # Increased from 3.0
            # Short words get moderate bonus
            else:
                bonus += 2.0  # Increased from 1.5
    
    return bonus

def meal_type_bonus(recipe_meal_type: str, target_meal_type: str) -> float:
    """FIXED: Much stricter meal type matching with heavy penalties for mismatches."""
    if not recipe_meal_type or not target_meal_type:
        return 0.0
    
    recipe_type = recipe_meal_type.lower().strip()
    target_type = target_meal_type.lower().strip()
    
    # Exact match - strong bonus
    if recipe_type == target_type:
        return 2.0  # Increased from 1.0
    
    # BREAKFAST rules - very strict
    if target_type == "breakfast":
        if recipe_type in ["brunch", "snack"]:
            return 0.3  # Reduced from 0.5
        if recipe_type in ["dinner", "main", "lunch"]:
            return -2.0  # Heavy penalty instead of -0.5
        return -1.0  # Default penalty for non-breakfast
    
    # LUNCH rules - moderately strict
    if target_type == "lunch":
        if recipe_type in ["dinner", "main"]:
            return 0.5  # Keep moderate compatibility
        if recipe_type == "breakfast":
            return -1.0  # Penalty for breakfast at lunch
        if recipe_type == "snack":
            return 0.2
        return -0.5  # Default penalty
    
    # DINNER rules - more flexible but still enforce
    if target_type == "dinner":
        if recipe_type in ["lunch", "main"]:
            return 0.8  # Increased from 0.6
        if recipe_type == "breakfast":
            return -0.5  # Penalty for breakfast at dinner
        if recipe_type == "snack":
            return 0.0
        return -0.3  # Slight penalty
    
    return -0.5  # Default penalty for unclear types

def calculate_tightening_bonus(meta: dict, diet_requirement: str, target_meal_type: str) -> float:
    """ENHANCED: Bonus when both diet AND meal type match perfectly."""
    if not diet_requirement or not target_meal_type:
        return 0.0
    
    diet_match = passes_diet_filter(meta, diet_requirement)
    meal_type_match = meta.get("meal_type", "").lower() == target_meal_type.lower()
    
    # Both match - strong bonus
    if diet_match and meal_type_match:
        return 2.0  # Increased from 1.0
    
    # Only diet matches
    if diet_match:
        return 0.5
    
    return 0.0

def final_score(
    similarity: float, 
    pantry_overlap: int, 
    like_bonus: float, 
    meal_type_bonus_val: float = 0.0,
    diversity_penalty: float = 0.0,
    allergy_penalty: float = 0.0,  
    tightening_bonus: float = 0.0,
    time_penalty: float = 0.0  # NEW
) -> float:
    """ENHANCED: Calculate final weighted score with all factors."""
    base_score = (WEIGHT_SIM * similarity + 
                  WEIGHT_PANTRY * pantry_overlap + 
                  WEIGHT_LIKE * like_bonus + 
                  WEIGHT_MEAL_TYPE * meal_type_bonus_val +
                  WEIGHT_TIGHTENING * tightening_bonus)
    
    # Apply penalties
    penalties = (WEIGHT_DIVERSITY * diversity_penalty + 
                WEIGHT_ALLERGY_PENALTY * allergy_penalty +
                WEIGHT_TIME_PENALTY * time_penalty)
    
    final = base_score - penalties
    
    return final

def passes_diet_filter(recipe: dict, diet_requirement: str) -> bool:
    """Check if recipe matches diet requirement."""
    if not diet_requirement:
        return True
    
    diet_requirement = diet_requirement.lower().strip()
    recipe_diet = recipe.get("diet", [])
    
    if isinstance(recipe_diet, str):
        recipe_diet = [recipe_diet]
    elif not isinstance(recipe_diet, list):
        recipe_diet = []
    
    recipe_diet_lower = [str(d).lower().strip() for d in recipe_diet]
    
    if diet_requirement == "vegetarian":
        return any(diet in recipe_diet_lower for diet in ["vegetarian", "vegan"])
    
    if diet_requirement == "vegan":
        return "vegan" in recipe_diet_lower
    
    if diet_requirement == "gluten-free":
        return "gluten-free" in recipe_diet_lower
    
    return diet_requirement in recipe_diet_lower

def passes_allergy_filter(ingredients: t.List[str], allergies: t.List[str]) -> bool:
    """STRICT: Check allergy safety - hard filter."""
    if not allergies:
        return True
    
    ingredients_text = " ".join(str(ing).lower() for ing in ingredients)
    return not any(allergy.lower() in ingredients_text for allergy in allergies)

def passes_time_filter(recipe: dict, time_budget: dict, meal_type: str) -> bool:
    """FIXED: Stricter time constraints with minimum time check."""
    recipe_time = recipe.get("time_minutes", 30)
    
    # NEW: Filter unrealistic prep times
    if recipe_time < MIN_PREP_TIME:
        logger.debug(f"Filtered recipe {recipe.get('title')} - unrealistic prep time: {recipe_time} min")
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

def log_top_recipes(user_id: str, meal_type: str, scored_recipes: t.List[t.Tuple[str, float, dict]], 
                   weights: dict, params: dict):
    """Log top-10 recipes with scoring breakdown."""
    logger.info(f"\n{'='*80}")
    logger.info(f"TOP-10 RECIPES | User: {user_id} | Meal: {meal_type}")
    logger.info(f"{'='*80}")
    logger.info(f"Weights: {weights}")
    logger.info(f"Params: Diet={params.get('diet')}, Allergies={params.get('allergies')}, "
               f"Likes={params.get('likes')}")
    logger.info(f"{'-'*80}")
    
    for idx, (rid, score, meta) in enumerate(scored_recipes[:10], 1):
        logger.info(f"{idx:2d}. {meta.get('title', 'Unknown')[:50]}")
        logger.info(f"    Score: {score:.3f} | Meal: {meta.get('meal_type')} | "
                   f"Time: {meta.get('time_minutes')}min")
        logger.info(f"    Recipe ID: {rid}")
    
    logger.info(f"{'='*80}\n")

def get_meal_recommendations_enhanced(
    req: 'RecommendationRequest', 
    meal_type: MealType, 
    k: int = 3,
    exclude_ids: set = None,
    selected_recipes: t.List[dict] = None
) -> t.List[dict]:
    """FIXED: Enhanced recommendations with proper filtering and scoring."""
    if exclude_ids is None:
        exclude_ids = set()
    
    if selected_recipes is None:
        selected_recipes = []
    
    if len(recipe_ids) == 0:
        return []
    
    # Build enhanced query with meal type emphasis
    query_parts = [meal_type.value, meal_type.value]  # Double weight on meal type
    
    # Add likes with high weight
    if req.likes:
        for like in req.likes[:5]:
            query_parts.extend([like, like])  # Double each like
    
    # Add diet
    if req.diet:
        query_parts.append(req.diet)
    
    # Add pantry items
    if req.pantry and len(query_parts) < 10:
        query_parts.extend(req.pantry[:4])
    
    query_text = " ".join(query_parts)
    logger.info(f"Query for {meal_type.value}: {query_text}")
    
    # Get candidates
    q_vec = text_to_vector(query_text)
    raw_candidates = query_faiss(q_vec, topk=500)  # Increased from 300
    
    if not raw_candidates:
        return []
    
    # Prepare filters
    pantry_set = set(p.lower() for p in req.pantry) if req.pantry else set()
    allergies_list = req.allergies if req.allergies else []
    dislikes_lower = [d.lower() for d in req.dislikes] if req.dislikes else []
    
    # Get time budget
    if isinstance(req.time_budget, dict) and meal_type.value in req.time_budget:
        time_budget = req.time_budget[meal_type.value]
    else:
        time_budget = DEFAULT_TIME_LIMITS.get(meal_type.value, 60)
    
    scored = []
    filtered_count = {"diet": 0, "allergy": 0, "time": 0, "dislike": 0, "min_time": 0}
    
    for rid, sim, idx in raw_candidates:
        if rid in exclude_ids:
            continue
        
        meta = meta_map.get(rid)
        if not meta:
            continue
        
        # Apply strict filters
        if req.diet and not passes_diet_filter(meta, req.diet):
            filtered_count["diet"] += 1
            continue
        
        ing_tokens = meta.get("ing_tokens", [])
        
        if not passes_allergy_filter(ing_tokens, allergies_list):
            filtered_count["allergy"] += 1
            continue
        
        title_lower = meta.get("title", "").lower()
        if any(dislike in title_lower for dislike in dislikes_lower):
            filtered_count["dislike"] += 1
            continue
        
        recipe_time = meta.get("time_minutes", 30)
        
        # Filter unrealistic times
        if recipe_time < MIN_PREP_TIME:
            filtered_count["min_time"] += 1
            continue
        
        # Strict time filter
        if not passes_time_filter(meta, req.time_budget, meal_type.value):
            filtered_count["time"] += 1
            continue
        
        # Calculate all scoring components
        pantry_score = pantry_overlap_score(ing_tokens, pantry_set)
        like_bonus = like_bonus_from_title(meta.get("title", ""), req.likes)
        meal_bonus = meal_type_bonus(meta.get("meal_type", ""), meal_type.value)
        diversity_penalty = calculate_diversity_penalty(selected_recipes, meta)
        allergy_penalty = calculate_allergy_penalty(ing_tokens, allergies_list)
        tightening_bonus = calculate_tightening_bonus(meta, req.diet, meal_type.value)
        time_penalty = calculate_time_penalty(recipe_time, time_budget, meal_type.value)
        
        final_score_val = final_score(
            similarity=sim,
            pantry_overlap=pantry_score,
            like_bonus=like_bonus,
            meal_type_bonus_val=meal_bonus,
            diversity_penalty=diversity_penalty,
            allergy_penalty=allergy_penalty,
            tightening_bonus=tightening_bonus,
            time_penalty=time_penalty
        )
        
        scored.append((rid, final_score_val, meta))
    
    logger.info(f"Filtered stats for {meal_type.value}: {filtered_count}")
    
    if not scored:
        logger.warning(f"No recipes found for {meal_type.value} after filtering")
        return []
    
    # Sort by score
    scored.sort(key=lambda x: x[1], reverse=True)
    
    # Log top recipes
    log_params = {
        'diet': req.diet,
        'allergies': allergies_list,
        'likes': req.likes
    }
    log_weights = {
        'similarity': WEIGHT_SIM,
        'pantry': WEIGHT_PANTRY,
        'likes': WEIGHT_LIKE,
        'meal_type': WEIGHT_MEAL_TYPE,
        'diversity': WEIGHT_DIVERSITY,
        'allergy_penalty': WEIGHT_ALLERGY_PENALTY,
        'tightening': WEIGHT_TIGHTENING,
        'time_penalty': WEIGHT_TIME_PENALTY
    }
    log_top_recipes(req.user_id, meal_type.value, scored, log_weights, log_params)
    
    # Select recommendations with iterative diversity checking
    recommendations = []
    remaining_candidates = scored[:min(100, len(scored))]
    
    for _ in range(k):
        if not remaining_candidates:
            break
        
        rid, score, meta = remaining_candidates[0]
        
        # Build recommendation object
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
            "pantry_matches": pantry_overlap_score(meta.get("ing_tokens", []), pantry_set)
        }
        
        recommendations.append(rec)
        selected_recipes.append(meta)
        
        # Rescore remaining candidates with updated diversity
        remaining_candidates = remaining_candidates[1:]
        if remaining_candidates and selected_recipes:
            rescored = []
            for rid_r, score_r, meta_r in remaining_candidates:
                pantry_score = pantry_overlap_score(meta_r.get("ing_tokens", []), pantry_set)
                like_bonus = like_bonus_from_title(meta_r.get("title", ""), req.likes)
                meal_bonus = meal_type_bonus(meta_r.get("meal_type", ""), meal_type.value)
                diversity_penalty = calculate_diversity_penalty(selected_recipes, meta_r)
                allergy_penalty = calculate_allergy_penalty(meta_r.get("ing_tokens", []), allergies_list)
                tightening_bonus = calculate_tightening_bonus(meta_r, req.diet, meal_type.value)
                time_penalty = calculate_time_penalty(meta_r.get("time_minutes", 30), time_budget, meal_type.value)
                
                orig_sim = next((s for r, s, m in scored if r == rid_r), 0.5)
                
                new_score = final_score(
                    similarity=orig_sim,
                    pantry_overlap=pantry_score,
                    like_bonus=like_bonus,
                    meal_type_bonus_val=meal_bonus,
                    diversity_penalty=diversity_penalty,
                    allergy_penalty=allergy_penalty,
                    tightening_bonus=tightening_bonus,
                    time_penalty=time_penalty
                )
                
                rescored.append((rid_r, new_score, meta_r))
            
            rescored.sort(key=lambda x: x[1], reverse=True)
            remaining_candidates = rescored
    
    return recommendations


app = FastAPI(title="Fixed Recipe Recommender v2.4")

class RecommendationRequest(BaseModel):
    user_id: str
    diet: t.Optional[str] = None
    allergies: t.List[str] = []
    dislikes: t.List[str] = []
    time_budget: dict = {}
    pantry: t.List[str] = []
    likes: t.List[str] = []
    recommendations_per_meal: int = 3

@app.get("/")
def read_root():
    """Root endpoint with API information."""
    return {
        "message": "Fixed Recipe Recommender API v2.4",
        "version": "2.4-fixed",
        "total_recipes": len(recipe_ids),
        "critical_fixes": [
            "✅ Meal type matching heavily weighted (4.0x)",
            "✅ Time budget strictly enforced",
            "✅ Minimum prep time filter (10+ minutes)",
            "✅ Enhanced diversity penalties",
            "✅ Stronger likes boost (1.5x)",
            "✅ Time penalty for budget violations"
        ],
        "improvements": [
            "🔧 Stricter breakfast/lunch/dinner separation",
            "🔧 No more unrealistic 2-minute recipes",
            "🔧 Better pantry matching with plural handling",
            "🔧 Enhanced allergy penalties (5-10x)",
            "🔧 Meal type mismatches get -2.0 penalty"
        ],
        "endpoints": {
            "health": "/health",
            "meal_plan": "/meal_plan",
            "recommend": "/recommend",
            "debug": "/debug_recommend",
            "stats": "/stats"
        },
        "status": "running",
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
        "weights": {
            "similarity": WEIGHT_SIM,
            "pantry": WEIGHT_PANTRY,
            "likes": WEIGHT_LIKE,
            "meal_type": WEIGHT_MEAL_TYPE,
            "diversity": WEIGHT_DIVERSITY,
            "allergy_penalty": WEIGHT_ALLERGY_PENALTY,
            "tightening_bonus": WEIGHT_TIGHTENING,
            "time_penalty": WEIGHT_TIME_PENALTY
        },
        "filters": {
            "min_prep_time": MIN_PREP_TIME,
            "default_time_limits": DEFAULT_TIME_LIMITS
        }
    }

@app.get("/stats")
def get_stats():
    """Get dataset statistics."""
    if not meta_map:
        return {"error": "No metadata loaded"}
    
    meal_counts = defaultdict(int)
    diet_counts = defaultdict(int)
    source_counts = defaultdict(int)
    time_distribution = {"under_10": 0, "10_30": 0, "30_60": 0, "over_60": 0}
    
    for meta in meta_map.values():
        meal_counts[meta.get("meal_type", "unknown")] += 1
        source_counts[meta.get("source", "unknown")] += 1
        
        for diet in meta.get("diet", []):
            diet_counts[diet] += 1
        
        time_min = meta.get("time_minutes", 30)
        if time_min < 10:
            time_distribution["under_10"] += 1
        elif time_min <= 30:
            time_distribution["10_30"] += 1
        elif time_min <= 60:
            time_distribution["30_60"] += 1
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
            "unrealistic_recipes_filtered": time_distribution["under_10"]
        },
        "scoring_weights": {
            "similarity": WEIGHT_SIM,
            "pantry": WEIGHT_PANTRY,
            "likes": WEIGHT_LIKE,
            "meal_type": WEIGHT_MEAL_TYPE,
            "diversity": WEIGHT_DIVERSITY,
            "allergy_penalty": WEIGHT_ALLERGY_PENALTY,
            "tightening_bonus": WEIGHT_TIGHTENING,
            "time_penalty": WEIGHT_TIME_PENALTY
        }
    }

@app.post("/meal_plan")
def get_daily_meal_plan(req: RecommendationRequest):
    """Get complete daily meal plan with fixed scoring and filtering."""
    try:
        if len(recipe_ids) == 0:
            return {
                "user_id": req.user_id,
                "meal_plan": {"breakfast": [], "lunch": [], "dinner": []},
                "total_recommendations": 0,
                "message": "No recipe data available"
            }
        
        logger.info(f"\n{'='*60}")
        logger.info(f"  Generating FIXED meal plan for {req.user_id}")
        logger.info(f"{'='*60}")
        logger.info(f"🥗 Diet: {req.diet or 'Any'}")
        logger.info(f"❤️ Likes: {', '.join(req.likes) if req.likes else 'None specified'}")
        logger.info(f"🥫 Pantry: {len(req.pantry)} items")
        logger.info(f"⏰ Time budget: {req.time_budget if req.time_budget else 'Using defaults'}")
        logger.info(f"⚠️ Allergies: {', '.join(req.allergies) if req.allergies else 'None'}")
        logger.info(f"👎 Dislikes: {', '.join(req.dislikes) if req.dislikes else 'None'}")
        
        used_recipe_ids = set()
        all_selected_recipes = []
        
        meal_plan = {}
        total_recs = 0
        
        for meal_type in [MealType.BREAKFAST, MealType.LUNCH, MealType.DINNER]:
            logger.info(f"\n🔍 Searching for {meal_type.value.upper()} recipes...")
            
            recommendations = get_meal_recommendations_enhanced(
                req, 
                meal_type, 
                req.recommendations_per_meal,
                exclude_ids=used_recipe_ids,
                selected_recipes=all_selected_recipes
            )
            
            for rec in recommendations:
                used_recipe_ids.add(rec['recipe_id'])
                meta = meta_map.get(rec['recipe_id'], {})
                all_selected_recipes.append(meta)
            
            meal_plan[meal_type.value] = recommendations
            total_recs += len(recommendations)
            
            logger.info(f"✅ Found {len(recommendations)} {meal_type.value} recipes")
            if recommendations:
                logger.info(f"   Top: {recommendations[0]['title']} (score: {recommendations[0]['score']:.2f})")
        
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
                tightened_count = sum(
                    1 for meal_recs in meal_plan.values()
                    for rec in meal_recs
                    if req.diet.lower() in [d.lower() for d in rec.get('diet', [])]
                )
                if tightened_count > total_recs * 0.8:
                    insights.append(f"✅ Strong {req.diet} match - {tightened_count}/{total_recs} recipes aligned")
            
            if req.likes:
                likes_matched = sum(
                    1 for meal_recs in meal_plan.values()
                    for rec in meal_recs
                    if any(like.lower() in rec['title'].lower() for like in req.likes)
                )
                if likes_matched > 0:
                    insights.append(f"❤️ Preference boost applied - {likes_matched} recipes match your likes")
            
            meal_type_accuracy = sum(
                1 for meal_type, meal_recs in meal_plan.items()
                for rec in meal_recs
                if rec['meal_type'] == meal_type
            )
            if meal_type_accuracy > total_recs * 0.8:
                insights.append(f"🎯 Meal type accuracy: {meal_type_accuracy}/{total_recs} correctly categorized")
            
            time_compliant = sum(
                1 for meal_type, meal_recs in meal_plan.items()
                for rec in meal_recs
                if rec['prep_time'] <= req.time_budget.get(meal_type, DEFAULT_TIME_LIMITS.get(meal_type, 60))
            )
            insights.append(f"⏰ Time compliant: {time_compliant}/{total_recs} within budget")
        
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
            "scoring_info": {
                "version": "2.4-fixed",
                "allergy_penalty_applied": len(req.allergies) > 0,
                "likes_boost_applied": len(req.likes) > 0,
                "diet_meal_tightening": req.diet is not None,
                "time_filtering": "strict",
                "min_prep_time": MIN_PREP_TIME,
                "weights_used": {
                    "similarity": WEIGHT_SIM,
                    "pantry": WEIGHT_PANTRY,
                    "likes": WEIGHT_LIKE,
                    "meal_type": WEIGHT_MEAL_TYPE,
                    "diversity": WEIGHT_DIVERSITY,
                    "allergy_penalty": WEIGHT_ALLERGY_PENALTY,
                    "tightening_bonus": WEIGHT_TIGHTENING,
                    "time_penalty": WEIGHT_TIME_PENALTY
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Error in meal_plan: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/debug_recommend")
def debug_recommend(req: RecommendationRequest):
    """Debug endpoint to see scoring breakdown."""
    try:
        if len(recipe_ids) == 0:
            return {"user_id": req.user_id, "error": "No recipe data available", "debug": []}
        
        query_text = " ".join(req.likes) or req.diet or "vegetarian"
        q_vec = text_to_vector(query_text)
        candidates = query_faiss(q_vec, topk=100)
        
        pantry_set = set(p.lower() for p in req.pantry) if req.pantry else set()
        allergies_list = req.allergies if req.allergies else []
        
        debug = []
        for rid, sim, idx in candidates[:50]:
            meta = meta_map.get(rid, {})
            
            passes_diet = passes_diet_filter(meta, req.diet) if req.diet else True
            passes_time = passes_time_filter(meta, req.time_budget, "dinner")
            passes_allergy = passes_allergy_filter(meta.get("ing_tokens", []), allergies_list)
            passes_min_time = meta.get("time_minutes", 30) >= MIN_PREP_TIME
            
            ing_tokens = meta.get("ing_tokens", [])
            pantry_score = pantry_overlap_score(ing_tokens, pantry_set)
            like_bonus = like_bonus_from_title(meta.get("title", ""), req.likes)
            meal_bonus = meal_type_bonus(meta.get("meal_type", ""), "dinner")
            allergy_penalty = calculate_allergy_penalty(ing_tokens, allergies_list)
            tightening_bonus = calculate_tightening_bonus(meta, req.diet, "dinner")
            time_penalty = calculate_time_penalty(meta.get("time_minutes", 30), 60, "dinner")
            
            final = final_score(sim, pantry_score, like_bonus, meal_bonus, 0.0, 
                              allergy_penalty, tightening_bonus, time_penalty)
            
            debug.append({
                "recipe_id": rid,
                "title": meta.get("title", rid),
                "similarity_score": float(sim),
                "pantry_overlap": pantry_score,
                "like_bonus": float(like_bonus),
                "meal_bonus": float(meal_bonus),
                "allergy_penalty": float(allergy_penalty),
                "tightening_bonus": float(tightening_bonus),
                "time_penalty": float(time_penalty),
                "final_score": float(final),
                "meal_type": meta.get("meal_type", "unknown"),
                "diet": meta.get("diet", []),
                "passes_diet_filter": passes_diet,
                "passes_time_filter": passes_time,
                "passes_allergy_filter": passes_allergy,
                "passes_min_time_filter": passes_min_time,
                "time_minutes": meta.get("time_minutes", 0),
                "source": meta.get("source", "")
            })
        
        debug.sort(key=lambda x: x['final_score'], reverse=True)
        
        return {
            "user_id": req.user_id,
            "query_text": query_text,
            "filters_applied": {
                "diet": req.diet,
                "allergies": allergies_list,
                "likes": req.likes,
                "time_budget": req.time_budget or DEFAULT_TIME_LIMITS,
                "min_prep_time": MIN_PREP_TIME
            },
            "weights": {
                "similarity": WEIGHT_SIM,
                "pantry": WEIGHT_PANTRY,
                "likes": WEIGHT_LIKE,
                "meal_type": WEIGHT_MEAL_TYPE,
                "diversity": WEIGHT_DIVERSITY,
                "allergy_penalty": WEIGHT_ALLERGY_PENALTY,
                "tightening_bonus": WEIGHT_TIGHTENING,
                "time_penalty": WEIGHT_TIME_PENALTY
            },
            "debug": debug
        }
    
    except Exception as e:
        logger.error(f"Error in debug_recommend: {e}")
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
            "query_text": " ".join(req.likes) or req.diet or "indian food",
            "recommendations": all_recommendations,
            "meal_breakdown": meal_plan_response["meal_plan"],
            "total_count": len(all_recommendations),
            "insights": meal_plan_response.get("insights"),
            "message": meal_plan_response.get("message"),
            "scoring_info": meal_plan_response.get("scoring_info")
        }
    
    except Exception as e:
        logger.error(f"Error in recommend: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    print("=" * 70)
    print("  FIXED RECIPE RECOMMENDER v2.4")
    print("=" * 70)
    print(f"\n📊 SYSTEM STATUS:")
    print(f"   • Recipe data: {len(recipe_ids):,} recipes")
    print(f"   • Metadata: {len(meta_map):,} recipes")
    print(f"   • FAISS available: {'✅ Yes' if USE_FAISS else '❌ No'}")
    print(f"   • FAISS index: {'✅ Loaded' if faiss_index is not None else '❌ Not loaded'}")
    
    print(f"\n🔧 CRITICAL FIXES IN v2.4:")
    print(f"   ✅ Meal type weight: {WEIGHT_MEAL_TYPE}x (was 2.5x)")
    print(f"   ✅ Time budget: Strictly enforced")
    print(f"   ✅ Min prep time: {MIN_PREP_TIME} minutes")
    print(f"   ✅ Likes boost: {WEIGHT_LIKE}x (was 1.2x)")
    print(f"   ✅ Time penalty: {WEIGHT_TIME_PENALTY}x for violations")
    print(f"   ✅ Allergy penalty: {WEIGHT_ALLERGY_PENALTY}x (was 3.0x)")
    
    print(f"\n⚙️ SCORING CONFIGURATION:")
    print(f"   • Similarity: {WEIGHT_SIM}")
    print(f"   • Pantry: {WEIGHT_PANTRY}")
    print(f"   • Likes: {WEIGHT_LIKE}")
    print(f"   • Meal type: {WEIGHT_MEAL_TYPE} ⭐")
    print(f"   • Diversity penalty: {WEIGHT_DIVERSITY}")
    print(f"   • Allergy penalty: {WEIGHT_ALLERGY_PENALTY}")
    print(f"   • Tightening bonus: {WEIGHT_TIGHTENING}")
    print(f"   • Time penalty: {WEIGHT_TIME_PENALTY}")
    
    print(f"\n🌐 SERVER ENDPOINTS:")
    print(f"   • Main: http://127.0.0.1:8000")
    print(f"   • Docs: http://127.0.0.1:8000/docs")
    print(f"   • Stats: http://127.0.0.1:8000/stats")
    print(f"   • Health: http://127.0.0.1:8000/health")
    print(f"   • Debug: http://127.0.0.1:8000/debug_recommend")
    print("=" * 70)
    print("\n🚀 Starting server...\n")
    
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")