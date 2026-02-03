import os
import typing as t
import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
import faiss
from sentence_transformers import SentenceTransformer
import json
import time
import re
import random 
import pandas as pd
from supabase import create_client, Client 

# --- CONFIGURATION ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Supabase Credentials
SUPABASE_URL = "https://wrwbqawmwqcknqlntpfb.supabase.co"
SUPABASE_KEY = "sb_publishable_tEak2s2lQyaMgBu4fQmG6Q_IjkcWLZU"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Paths
DATA_DIR = r"/data1/home/sathvik/Documents/FRESH"
VECS_PATH = os.path.join(DATA_DIR, "recipe_embeddings_2m.pt")
METADATA_PATH = os.path.join(DATA_DIR, "recipe_metadata_2m.pkl")
RANKER_PATH = "fresh_model_epoch_5.pth"
CLASSIFIER_PATH = "meal_classifier.pth" 

# ==============================================================================
# 🧠 1. MODEL ARCHITECTURE (Included here to fix NameError)
# ==============================================================================
class FeatureInteractionLayer(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        attn_out, _ = self.multihead_attn(x, x, x)
        return self.norm(x + attn_out)

class FRESH_Network(nn.Module):
    def __init__(self, embedding_dim=384):
        super(FRESH_Network, self).__init__()
        self.taste_encoder = nn.Sequential(
            nn.Linear(embedding_dim * 2, 256),
            nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, 128)
        )
        self.constraint_encoder = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, 128)
        )
        self.wide_encoder = nn.Sequential(nn.Linear(2, 16), nn.ReLU())
        self.interaction = FeatureInteractionLayer(dim=128, num_heads=4)
        self.head = nn.Sequential(
            nn.Linear(128 * 2 + 16 + 1, 64), 
            nn.LayerNorm(64), nn.GELU(),
            nn.Linear(64, 1), nn.Sigmoid() 
        )

    def forward(self, recipe_vecs, user_vecs, constraint_vecs, explicit_feats, align_score):
        taste_input = torch.cat([recipe_vecs, user_vecs], dim=1) 
        taste_emb = self.taste_encoder(taste_input)
        constraint_emb = self.constraint_encoder(constraint_vecs)
        seq_input = torch.stack([taste_emb, constraint_emb], dim=1)
        interacted = self.interaction(seq_input)
        flat_features = interacted.view(interacted.size(0), -1)
        wide_out = self.wide_encoder(explicit_feats)
        final_input = torch.cat([flat_features, wide_out, align_score], dim=1)
        return self.head(final_input)

# --- DEFINE MEAL CLASSIFIER ARCHITECTURE ---
class MealClassifier(nn.Module):
    def __init__(self, input_dim=384, num_classes=4):
        super(MealClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
            nn.Sigmoid() 
        )
    def forward(self, x): return self.network(x)

# --- LOAD DATA ---
print("Loading Data...")
try:
    # 1. Load Vectors (Handle both .pt and .npy)
    if VECS_PATH.endswith('.pt'):
        recipe_vectors = torch.load(VECS_PATH, map_location='cpu').numpy().astype(np.float32)
    else:
        recipe_vectors = np.load(VECS_PATH)
        if hasattr(recipe_vectors, 'files'):
            recipe_vectors = recipe_vectors[recipe_vectors.files[0]]
        recipe_vectors = recipe_vectors.astype(np.float32)

    # 2. GENERATE IDs (Fixing the IDS_PATH error)
    # Since we are using the new 2M dataset, IDs are just indices 0..N
    recipe_ids = np.arange(recipe_vectors.shape[0])
    
    # 3. Normalize for FAISS
    norms = np.linalg.norm(recipe_vectors, axis=1, keepdims=True)
    recipe_vectors_norm = recipe_vectors / np.maximum(norms, 1e-12)
    
    # 4. Build FAISS Index
    d = recipe_vectors.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(recipe_vectors_norm)
    print(f"Data Loaded. {len(recipe_ids)} recipes.")

except Exception as e:
    print(f"CRITICAL ERROR LOADING DATA: {e}")
    recipe_vectors = [] 
    index = None

# --- LOAD METADATA ---
meta_map = {}
if os.path.exists(METADATA_PATH):
    try:
        # Load Pickle (New Format)
        if METADATA_PATH.endswith('.pkl'):
            df = pd.read_pickle(METADATA_PATH)
            # Use Index as ID
            meta_map = df.T.to_dict()
            # Ensure keys are strings
            meta_map = {str(k): v for k, v in meta_map.items()}
        # Load JSON (Old Format)
        else:
            with open(METADATA_PATH, 'r') as f:
                data = json.load(f)
                for m in data:
                    meta_map[str(m.get('recipe_id'))] = m
    except Exception as e:
        print(f"Error loading metadata: {e}")

# --- SAFETY LISTS ---
NON_VEG_KEYWORDS = {
    "chicken", "beef", "pork", "steak", "fish", "salmon", "shrimp", 
    "crab", "bacon", "ham", "sausage", "meat", "lamb", "tuna", "anchovy", 
    "gelatin", "turkey", "duck", "goose", "clam", "oyster", "mussel", "egg", "eggs"
}
NON_VEGAN_KEYWORDS = NON_VEG_KEYWORDS.union({
    "milk", "butter", "cream", "cheese", "honey", "yogurt", "ghee", "paneer", 
    "whey", "casein", "lactose"
})

# --- LOAD MODELS ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Ranking Model
fresh_model = FRESH_Network()
try:
    fresh_model.load_state_dict(torch.load(RANKER_PATH, map_location=device))
    fresh_model.to(device).eval()
    print("✅ RANKER LOADED SUCCESSFULLY")
except: 
    print("⚠️ RANKER MISSING (Using random weights)")
    fresh_model.to(device).eval()

# 2. Meal Classifier Model
meal_model = MealClassifier()
try:
    meal_model.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=device))
    meal_model.to(device).eval()
    print("✅ MEAL CLASSIFIER LOADED SUCCESSFULLY")
except: print("⚠️ MEAL CLASSIFIER MISSING")

embedder = SentenceTransformer('all-MiniLM-L6-v2')
def text_to_vector(text): return embedder.encode(text)

# --- SEMANTIC ANCHORS ---
macro_vectors = {
    "Protein": embedder.encode("meat fish chicken beef tofu lentils beans eggs protein muscle"),
    "Carbs": embedder.encode("rice pasta bread potato sugar grains flour wheat carbs starch"),
    "Fats": embedder.encode("oil butter cream cheese nuts avocado fat fry rich grease")
}
indian_anchor = embedder.encode("authentic indian cuisine curry masala spices tandoori south indian north indian biryani dal roti")

# --- DATABASE HELPERS ---
def get_user_profile(user_id: str):
    try:
        r = supabase.table("users").select("*").eq("id", user_id).execute()
        return r.data[0] if r.data else {}
    except: return {}

def get_user_pantry(user_id: str):
    try:
        r = supabase.table("pantry_items").select("ingredient_name").eq("user_id", user_id).execute()
        return [i['ingredient_name'] for i in r.data]
    except: return []

def get_user_history(user_id: str):
    try:
        r = supabase.table("interactions").select("recipe_id").eq("user_id", user_id).limit(50).execute()
        return r.data
    except: return []

# --- ONLINE LEARNING HELPER ---
def update_user_taste_vector(user_id, recipe_id, action_type="cook"):
    alpha = 0.15
    if action_type == "cook": alpha = 0.20
    elif action_type == "like": alpha = 0.45
    elif action_type == "dislike": alpha = -0.25
    
    try:
        r_indices = np.where(recipe_ids == int(recipe_id))[0] # Cast to int for new ID system
        if len(r_indices) == 0: return 
        r_vec = recipe_vectors[r_indices[0]]
        
        user_data = supabase.table("users").select("taste_vector").eq("id", user_id).execute()
        current_vec_data = user_data.data[0].get("taste_vector")
        
        if current_vec_data and len(current_vec_data) > 0:
            user_vec = np.array(current_vec_data, dtype=np.float32)
            if action_type == "dislike":
                new_vec = user_vec - (alpha * r_vec) 
            else:
                new_vec = (1 - alpha) * user_vec + (alpha) * r_vec 
        else:
            new_vec = r_vec
            
        norm = np.linalg.norm(new_vec)
        if norm > 0: new_vec = new_vec / norm

        supabase.table("users").update({"taste_vector": new_vec.tolist()}).eq("id", user_id).execute()
        logger.info(f"✅ User {user_id} taste vector updated ({action_type}).")
    except Exception as e:
        logger.error(f"Online Learning Failed: {e}")

# --- SMART MATCHING HELPERS ---
def get_missing_ingredients(recipe_ings, pantry_items):
    """Calculates missing ingredients for the 'Buy List'"""
    missing = []
    pantry_set = set(p.lower().strip() for p in pantry_items)
    
    # Common staples to ignore in "missing list"
    staples = {"salt", "sugar", "water", "oil", "ice", "pepper", "ghee", "turmeric", "chili powder", "mustard seeds"}
    
    # recipe_ings might be a list or a string in the new metadata
    if isinstance(recipe_ings, str):
        ing_list = recipe_ings.split(',')
    else:
        ing_list = recipe_ings

    for ing in ing_list:
        ing_clean = ing.lower().strip()
        # Check if pantry item exists inside recipe ingredient string
        if not any(p in ing_clean for p in pantry_set) and ing_clean not in staples: 
            missing.append(ing.title()) # Format nicely
    return missing[:5] # Limit to 5 items

def fuzzy_pantry_match(ing_tokens, pantry_items, title=""):
    matches = 0
    pantry_set = set(p.lower().strip() for p in pantry_items)
    # Check format
    if isinstance(ing_tokens, list):
        text = " ".join(ing_tokens + [title]).lower()
    else:
        text = (str(ing_tokens) + " " + title).lower()
        
    for p in pantry_set:
        if p in text: matches += 1
    return min(matches, 5)

# --- SCORING ENGINE ---
def score_candidates_neural(filtered_candidates, preferences):
    if not filtered_candidates: return []
    
    # Batch Containers
    recipe_vecs_batch = []
    user_vecs_batch = []
    constraint_vecs_batch = []
    explicit_feats_batch = []
    align_scores_batch = []
    candidate_metadata = []
    
    user_bmi = preferences.get("bmi", 22.0)
    query_keywords = preferences.get("query_keywords", [])
    has_pantry = len(preferences.get("pantry", [])) > 0
    
    # Prepare Inputs
    constraint_vec = preferences.get("constraint_vec", None)
    if constraint_vec is None:
        batch_constraint_vec = np.zeros(384, dtype=np.float32)
    else:
        batch_constraint_vec = constraint_vec.astype(np.float32)

    user_vec = preferences.get("user_vec", None)
    if user_vec is None:
        batch_user_vec = batch_constraint_vec 
    else:
        batch_user_vec = user_vec.astype(np.float32)
    
    for rid, sim, idx, meta in filtered_candidates:
        vec = recipe_vectors[idx]
        ing_tokens = meta.get("ingredients", []) # Changed from cleaned_ingredients
        title = meta.get("title", "")
        
        # A. Pantry Score
        pantry_score = min(fuzzy_pantry_match(ing_tokens, preferences.get("pantry", []), title) / 5.0, 1.0)
        
        # B. Health Score
        prot = max(0.01, float(np.dot(vec, macro_vectors["Protein"])))
        carb = max(0.01, float(np.dot(vec, macro_vectors["Carbs"])))
        fat =  max(0.01, float(np.dot(vec, macro_vectors["Fats"])))
        total = prot + carb + fat
        
        p_ratio = prot / total
        c_ratio = carb / total
        f_ratio = fat / total
        
        health_score = 1.0
        if user_bmi >= 25.0: 
            if p_ratio > 0.4: health_score = 1.2 
            elif (c_ratio + f_ratio) > 0.7: health_score = 0.6 
        elif user_bmi <= 18.5:
            if (c_ratio + f_ratio) > 0.6: health_score = 1.2
            
        # C. Alignment Score
        alignment = 0.0
        if constraint_vec is not None:
             alignment = float(np.dot(vec, constraint_vec))
        
        # Batching
        recipe_vecs_batch.append(vec)
        user_vecs_batch.append(batch_user_vec)
        constraint_vecs_batch.append(batch_constraint_vec)
        explicit_feats_batch.append([pantry_score, health_score])
        align_scores_batch.append([alignment])
        
        candidate_metadata.append({
            "rid": rid, "meta": meta, 
            "macros": {"p": p_ratio, "c": c_ratio, "f": f_ratio},
            "p_val": pantry_score, "h_val": health_score
        })

    if not recipe_vecs_batch: return []
    
    # Tensor Conversion
    X_recipe = torch.tensor(np.array(recipe_vecs_batch), dtype=torch.float32).to(device)
    X_user = torch.tensor(np.array(user_vecs_batch), dtype=torch.float32).to(device)
    X_const = torch.tensor(np.array(constraint_vecs_batch), dtype=torch.float32).to(device)
    X_explicit = torch.tensor(np.array(explicit_feats_batch), dtype=torch.float32).to(device)
    X_align = torch.tensor(np.array(align_scores_batch), dtype=torch.float32).to(device)
    
    # Neural Inference
    with torch.no_grad():
        neural_scores = fresh_model(X_recipe, X_user, X_const, X_explicit, X_align).cpu().numpy().flatten()
        
    results = []
    for i, item in enumerate(candidate_metadata):
        rid = item['rid']
        meta = item['meta']
        macros = item['macros']
        
        neural_raw = float(neural_scores[i])
        p_val = item['p_val']
        alignment = float(align_scores_batch[i][0])
        
        # Scoring Logic
        w_n = neural_raw * 1.0 
        w_p = p_val * (0.5 if has_pantry else 0.0) 
        w_a = 0.0
        if alignment > 0.15:
            w_a = alignment * 3.0 
        
        raw_score = w_n + w_p + w_a 
        if alignment < 0.05: raw_score *= 0.1

        ui_score = 0.60 + (raw_score * 0.40)
        if alignment > 0.2: ui_score += 0.15
        if p_val > 0.6: ui_score += 0.10
        final_score = min(ui_score, 0.99)
        
        matched_nutrients = []
        if macros['p'] > 0.40: matched_nutrients.append("High Protein")
        if macros['c'] < 0.20: matched_nutrients.append("Low Carb")
        
        txt = (meta.get('title', '') + " " + str(meta.get('ingredients', []))).lower()
        for k in query_keywords:
            if k.lower() in txt: 
                final_score += 0.05 
                matched_nutrients.append(k.title())
        
        final_score = min(final_score, 0.99)
        matched_nutrients = list(set(matched_nutrients))

        # IMPORTANT: Calculate Missing Ingredients Here
        missing_ings = get_missing_ingredients(meta.get("ingredients", []), preferences.get("pantry", []))

        reason = "Recommended"
        if p_val > 0.6: reason = "Great Pantry Match"
        elif len(missing_ings) > 0: reason = f"Shopping needed ({len(missing_ings)} items)"
        
        final_score += random.uniform(-0.02, 0.02)
        
        results.append({
            "recipe_id": rid, 
            "title": meta.get("title"), 
            "ingredients": meta.get("ingredients", []),
            "missing_ingredients": missing_ings, # <--- Return this to API
            "link": f"https://google.com/search?q={meta['title']}",
            "nutrition": {
                "Protein": round(macros['p'] * 100, 1),
                "Carbs": round(macros['c'] * 100, 1),
                "Fats": round(macros['f'] * 100, 1)
            },
            "score": final_score, 
            "reason": reason, 
            "matched_nutrients": matched_nutrients,
            "match_details": {"neural_confidence": f"{neural_raw:.1%}", "pantry_match_level": f"{p_val:.1%}"}
        })
    return results

# --- API ---
app = FastAPI()

class MealPlanRequest(BaseModel):
    user_id: str; likes: list; pantry: list = None; diet: str = None
    time_budget: int = 60; num_recs: int = 3; regenerate: bool = False
    indian_only: bool = False; bmi: float = 22.0; query_keywords: list = [] 

class CookEvent(BaseModel):
    user_id: str; recipe_id: str; rating: int = 5; interaction_type: str = "cook"; recipe_title: str = ""

@app.get("/")
def root(): return {"status": "online", "message": "FRESH AI Connected"}

@app.post("/log_cooking")
def log_cooking(event: CookEvent):
    try:
        supabase.table("interactions").insert({
            "user_id": event.user_id,
            "recipe_id": event.recipe_id,
            "interaction_type": event.interaction_type,
            "recipe_title": event.recipe_title
        }).execute()
        update_user_taste_vector(event.user_id, event.recipe_id, event.interaction_type)
        return {"status": "Logged & Learned"}
    except Exception as e:
        return {"status": "Error", "details": str(e)}

@app.post("/generate_meal_plan")
def generate_meal_plan(req: MealPlanRequest):
    meal_map_indices = {"breakfast": 0, "lunch": 1, "dinner": 2, "snack": 3}
    full_plan = {}
    
    # 1. Fetch & Merge Data
    u_prof = get_user_profile(req.user_id)
    u_pantry = list(set((req.pantry or []) + get_user_pantry(req.user_id)))
    u_diet = req.diet or (u_prof.get("dietary_constraints", [None])[0])
    u_bmi = u_prof.get("bmi_data", {}).get("bmi", req.bmi)
    blocked = set(x['recipe_id'] for x in get_user_history(req.user_id))
    
    # Constraint Vector
    constraint_vec = None
    kw_str = ""
    if req.query_keywords:
        kw_str = " ".join(req.query_keywords)
        constraint_vec = text_to_vector(kw_str) 

    # 2. Retrieval Loop
    for meal_name in ["breakfast", "lunch", "dinner"]:
        budget = req.time_budget
        if meal_name == 'breakfast': budget = max(15, req.time_budget * 0.5)
        
        # --- QUERY EXPANSION ---
        query_base = f"{meal_name} " + " ".join(req.likes + random.sample(u_pantry, min(3, len(u_pantry))))
        if "south indian" in kw_str.lower(): query_base += " idli dosa sambar vada uttapam chettinad kerala"
        elif "north indian" in kw_str.lower(): query_base += " roti naan paneer dal makhani punjabi paratha"
        elif req.indian_only: query_base += " authentic indian cuisine curry spicy"
        
        vec_context = text_to_vector(query_base)
        
        # Weights
        if constraint_vec is not None:
            w_query = 0.90
            w_history = 0.10
        else:
            w_query = 0.50
            w_history = 0.50

        if u_prof.get("taste_vector"):
            learned_vec = np.array(u_prof.get("taste_vector"), dtype=np.float32)
            v_q = vec_context / (np.linalg.norm(vec_context) + 1e-9)
            v_h = learned_vec / (np.linalg.norm(learned_vec) + 1e-9)
            vec_context = (v_q * w_query) + (v_h * w_history)

        final_search_vec = vec_context
        if constraint_vec is not None:
             final_search_vec = (vec_context * 0.5) + (constraint_vec * 0.5)

        D, I = index.search(final_search_vec.reshape(1,-1), 1500) 
        
        candidates = []
        seen = set()
        
        valid_indices = [idx for idx in I[0] if idx >= 0]
        if not valid_indices: continue
        
        candidate_vecs = torch.tensor(recipe_vectors[valid_indices]).to(device)
        with torch.no_grad():
            meal_probs = meal_model(candidate_vecs)
            
        target_idx = meal_map_indices[meal_name]
        
        for i, real_idx in enumerate(valid_indices):
            rid = str(recipe_ids[real_idx])
            if rid in blocked: continue
            meta = meta_map.get(rid, {})
            title = meta.get('title', '').strip()
            if title in seen: continue
            
            # --- 🛡️ HARD FILTERING ---
            full_text = (str(title) + " " + str(meta.get('ingredients', []))).lower()
            
            # 1. Diet Check
            if u_diet in ['Vegetarian', 'Veg']:
                if any(x in full_text for x in NON_VEG_KEYWORDS): continue
            elif u_diet == 'Vegan':
                if any(x in full_text for x in NON_VEGAN_KEYWORDS): continue
            
            # 2. Negative Constraints
            if "no onion" in kw_str.lower() and "onion" in full_text: continue
            if "no garlic" in kw_str.lower() and "garlic" in full_text: continue
            
            # 3. Allergy Check
            user_allergies = req.model_dump().get('profile', {}).get('allergies', [])
            if user_allergies:
                if any(alg.lower() in full_text for alg in user_allergies): continue

            prob = float(meal_probs[i][target_idx])
            threshold = 0.3
            if meal_name == 'breakfast':
                if req.indian_only:
                    snack_prob = float(meal_probs[i][3]) 
                    prob = max(prob, snack_prob) 
                    threshold = 0.25 
            if meal_name in str(meta.get('meal_type','')).lower(): prob += 0.3
            if prob < threshold: continue
            
            r_time = meta.get("time_minutes", 0)
            if meal_name != 'breakfast' and r_time > 0 and r_time > (budget + 15): continue
            
            seen.add(title)
            candidates.append((rid, 0.0, real_idx, meta))
            
        if candidates:
            ctx = req.model_dump()
            ctx['pantry'] = u_pantry
            ctx['bmi'] = u_bmi
            ctx['constraint_vec'] = constraint_vec 
            ctx['user_vec'] = vec_context 
            
            scored = score_candidates_neural(candidates, ctx)
            scored.sort(key=lambda x: x['score'], reverse=True)
            
            # --- 80/20 SELECTION LOGIC ---
            high_pantry = []
            discovery = []
            
            for rec in scored:
                p_str = rec['match_details']['pantry_match_level'].strip('%')
                p_val = float(p_str)
                if p_val >= 50.0:
                    high_pantry.append(rec)
                else:
                    discovery.append(rec)
            
            final_list = []
            sel_titles = []
            
            # STRATEGY: 
            # If Auto Mode (no query) -> Prioritize Pantry (100% Pantry)
            # If Smart Mode (query exists) -> Allow Discovery (80/20)
            if not req.query_keywords and not req.likes:
                target_pantry = req.num_recs
                target_discovery = 0
            else:
                target_discovery = max(1, int(req.num_recs * 0.2)) 
                target_pantry = req.num_recs - target_discovery
            
            def add_unique(pool, quota):
                count = 0
                for rec in pool:
                    if count >= quota: break
                    t1 = set(rec['title'].lower().split())
                    if any(len(t1 & set(t2.lower().split()))/len(t1 | set(t2.lower().split())) > 0.6 for t2 in sel_titles):
                        continue
                    final_list.append(rec)
                    sel_titles.append(rec['title'])
                    count += 1
                return count

            add_unique(high_pantry, target_pantry)
            add_unique(discovery, target_discovery)
            
            # Fill remaining slots with whatever is best
            needed = req.num_recs - len(final_list)
            if needed > 0:
                remaining = [r for r in high_pantry if r not in final_list] + [r for r in discovery if r not in final_list]
                add_unique(remaining, needed)
            
            final_list.sort(key=lambda x: x['score'], reverse=True)
            full_plan[meal_name] = final_list
        else: full_plan[meal_name] = []

    return {"user_id": req.user_id, "meal_plan": full_plan}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)