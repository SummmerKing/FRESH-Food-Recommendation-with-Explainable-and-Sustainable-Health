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
import itertools

# ✅ Correct Import
from nutrition_agent import nutrition_analysis_agent


def calculate_exact_shapley(model, inputs, feature_names):
    """
    Computes exact Shapley values by permuting feature groups.
    N=5 features results in 32 forward passes per recipe.
    """
    n = len(inputs)
    all_combinations = list(itertools.product([0, 1], repeat=n))
    
    null_inputs = [torch.zeros_like(x) for x in inputs]
    with torch.no_grad():
        baseline_score = model(*null_inputs).item()
        
    shap_values = {name: 0.0 for name in feature_names}
    
    for i, feature_name in enumerate(feature_names):
        marginal_contributions = 0
        count = 0
        for combo in all_combinations:
            if combo[i] == 1: continue 
            
            coalition_without = list(combo)
            coalition_with = list(combo); coalition_with[i] = 1
            
            inputs_without = [inputs[j] if coalition_without[j]==1 else null_inputs[j] for j in range(n)]
            inputs_with    = [inputs[j] if coalition_with[j]==1    else null_inputs[j] for j in range(n)]
            
            with torch.no_grad():
                score_without = model(*inputs_without).item()
                score_with    = model(*inputs_with).item()
                
            marginal_contributions += (score_with - score_without)
            count += 1
            
        shap_values[feature_name] = marginal_contributions / count
        
    return shap_values, baseline_score

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
# 🧠 MODEL ARCHITECTURE
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
    if VECS_PATH.endswith('.pt'):
        recipe_vectors = torch.load(VECS_PATH, map_location='cpu').numpy().astype(np.float32)
    else:
        recipe_vectors = np.load(VECS_PATH)
        if hasattr(recipe_vectors, 'files'):
            recipe_vectors = recipe_vectors[recipe_vectors.files[0]]
        recipe_vectors = recipe_vectors.astype(np.float32)

    recipe_ids = np.arange(recipe_vectors.shape[0])
    norms = np.linalg.norm(recipe_vectors, axis=1, keepdims=True)
    recipe_vectors_norm = recipe_vectors / np.maximum(norms, 1e-12)
    
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
        if METADATA_PATH.endswith('.pkl'):
            df = pd.read_pickle(METADATA_PATH)
            meta_map = df.T.to_dict()
            meta_map = {str(k): v for k, v in meta_map.items()}
        else:
            with open(METADATA_PATH, 'r') as f:
                data = json.load(f)
                for m in data:
                    meta_map[str(m.get('recipe_id'))] = m
    except Exception as e:
        print(f"Error loading metadata: {e}")

# --- SAFETY LISTS ---
NON_VEG_KEYWORDS = {
    "chicken", "beef", "pork", "steak", "fish", "salmon", "shrimp", "crab", "bacon", 
    "ham", "sausage", "meat", "lamb", "tuna", "anchovy", "gelatin", "turkey", "duck", "goose", "egg", "eggs"
}
NON_VEGAN_KEYWORDS = NON_VEG_KEYWORDS.union({"milk", "butter", "cream", "cheese", "honey", "yogurt", "ghee", "paneer", "whey", "casein", "lactose"})

# --- LOAD MODELS ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

fresh_model = FRESH_Network()
try:
    fresh_model.load_state_dict(torch.load(RANKER_PATH, map_location=device))
    fresh_model.to(device).eval()
    print("✅ RANKER LOADED SUCCESSFULLY")
except: 
    print("⚠️ RANKER MISSING (Using random weights)")
    fresh_model.to(device).eval()

meal_model = MealClassifier()
try:
    meal_model.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=device))
    meal_model.to(device).eval()
    print("✅ MEAL CLASSIFIER LOADED SUCCESSFULLY")
except: print("⚠️ MEAL CLASSIFIER MISSING")

embedder = SentenceTransformer('all-MiniLM-L6-v2')
def text_to_vector(text): return embedder.encode(text)

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

def update_user_taste_vector(user_id, recipe_id, action_type="cook"):
    alpha = 0.15
    if action_type == "cook": alpha = 0.10
    elif action_type == "like": alpha = 0.20
    elif action_type == "dislike": alpha = -0.10
    try:
        r_indices = np.where(recipe_ids == int(recipe_id))[0]
        if len(r_indices) == 0: return 
        r_vec = recipe_vectors[r_indices[0]]
        user_data = supabase.table("users").select("taste_vector").eq("id", user_id).execute()
        current_vec_data = user_data.data[0].get("taste_vector")
        if current_vec_data and len(current_vec_data) > 0:
            user_vec = np.array(current_vec_data, dtype=np.float32)
            if action_type == "dislike": new_vec = user_vec - (alpha * r_vec) 
            else: new_vec = (1 - alpha) * user_vec + (alpha) * r_vec 
        else: new_vec = r_vec
        norm = np.linalg.norm(new_vec)
        if norm > 0: new_vec = new_vec / norm
        supabase.table("users").update({"taste_vector": new_vec.tolist()}).eq("id", user_id).execute()
        logger.info(f"✅ User {user_id} taste vector updated ({action_type}).")
    except Exception as e: logger.error(f"Online Learning Failed: {e}")

def get_missing_ingredients(recipe_ings, pantry_items):
    missing = []
    pantry_set = set(p.lower().strip() for p in pantry_items)
    staples = {"salt", "sugar", "water", "oil", "ice", "pepper", "ghee", "turmeric", "chili powder", "mustard seeds"}
    if isinstance(recipe_ings, str): ing_list = recipe_ings.split(',')
    else: ing_list = recipe_ings
    for ing in ing_list:
        ing_clean = ing.lower().strip()
        if not any(p in ing_clean for p in pantry_set) and ing_clean not in staples: 
            missing.append(ing.title())
    return missing[:5]

def fuzzy_pantry_match(ing_tokens, pantry_items, title=""):
    matches = 0
    pantry_set = set(p.lower().strip() for p in pantry_items)
    if isinstance(ing_tokens, list): text = " ".join(ing_tokens + [title]).lower()
    else: text = (str(ing_tokens) + " " + title).lower()
    for p in pantry_set:
        if p in text: matches += 1
    return min(matches, 5)

def score_candidates_neural(filtered_candidates, preferences, rag_explanation=""):
    """
    Scores recipes using internal model weights and external Shapley attribution.
    Fixed: Variable scoping, None-Type safety, and relative batch comparisons.
    """
    if not filtered_candidates:
        return []
    
    # 1. SETUP GLOBAL CONTEXT
    user_bmi = preferences.get("bmi", 22.0)
    query_keywords = preferences.get("query_keywords", [])
    user_pantry = preferences.get("pantry", [])
    has_pantry = len(user_pantry) > 0
    
    # 2. VECTOR SAFETY CHECK
    raw_constraint = preferences.get("constraint_vec")
    constraint_vec = np.array(raw_constraint, dtype=np.float32) if raw_constraint is not None else np.zeros(384, dtype=np.float32)

    raw_user = preferences.get("user_vec")
    user_vec = np.array(raw_user, dtype=np.float32) if raw_user is not None else np.zeros(384, dtype=np.float32)
    
    results = []

    # 3. PRE-CALCULATE BATCH MAXES (For relative reasoning)
    # We do this first so the AI can say "Highest in this set"
    all_pantry_scores = []
    all_protein_ratios = []
    
    for _, _, _, meta in filtered_candidates:
        p_match = fuzzy_pantry_match(meta.get("ingredients", []), user_pantry, meta.get("title", "")) / 5.0
        all_pantry_scores.append(p_match)
        
        # Quick Macro Check for batch stats
        rid_vec = recipe_vectors[int(np.where(recipe_ids == int(meta.get('recipe_id', 0)))[0][0])] if 'recipe_id' in meta else np.zeros(384)
        prot_val = max(0.01, float(np.dot(rid_vec, macro_vectors["Protein"])))
        carb_val = max(0.01, float(np.dot(rid_vec, macro_vectors["Carbs"])))
        fat_val = max(0.01, float(np.dot(rid_vec, macro_vectors["Fats"])))
        all_protein_ratios.append(prot_val / (prot_val + carb_val + fat_val + 1e-9))

    batch_max_pantry = max(all_pantry_scores) if all_pantry_scores else 0
    batch_max_protein = max(all_protein_ratios) if all_protein_ratios else 0

    # 4. INDIVIDUAL RECIPE SCORING LOOP
    for rid, sim, idx, meta in filtered_candidates:
        vec = recipe_vectors[idx]
        ing_tokens = meta.get("ingredients", [])
        title = meta.get("title", "")
        
        # Define current p_val explicitly to avoid NameError
        current_p_val = min(fuzzy_pantry_match(ing_tokens, user_pantry, title) / 5.0, 1.0)
        
        # Macros
        prot = max(0.01, float(np.dot(vec, macro_vectors["Protein"])))
        carb = max(0.01, float(np.dot(vec, macro_vectors["Carbs"])))
        fat =  max(0.01, float(np.dot(vec, macro_vectors["Fats"])))
        total_macros = prot + carb + fat
        p_ratio = prot / total_macros; c_ratio = carb / total_macros; f_ratio = fat / total_macros
        
        h_score = 1.0
        if user_bmi >= 25.0 and p_ratio > 0.4: h_score = 1.2
        
        alignment_val = float(np.dot(vec, constraint_vec))

        # 5. TENSOR CONVERSION
        t_recipe = torch.tensor(vec, dtype=torch.float32).unsqueeze(0).to(device)
        t_user   = torch.tensor(user_vec, dtype=torch.float32).unsqueeze(0).to(device)
        t_const  = torch.tensor(constraint_vec, dtype=torch.float32).unsqueeze(0).to(device)
        t_expl   = torch.tensor([current_p_val, h_score], dtype=torch.float32).unsqueeze(0).to(device)
        t_align  = torch.tensor([alignment_val], dtype=torch.float32).unsqueeze(0).to(device)
        
        # 6. SHAPLEY ATTRIBUTION
        feature_groups = ["Taste (History)", "User Context", "Medical (RAG)", "Pantry/Health", "Alignment"]
        inputs = [t_recipe, t_user, t_const, t_expl, t_align]
        
        shap_vals, base_score = calculate_exact_shapley(fresh_model, inputs, feature_groups)
        
        # Aggregating for UI
        total_shap_taste = shap_vals["Taste (History)"] + shap_vals["User Context"]
        total_shap_pantry = shap_vals["Pantry/Health"]
        total_shap_health = shap_vals["Medical (RAG)"] + shap_vals["Alignment"]
        
        final_neural_score = base_score + sum(shap_vals.values())
        
        # 7. DYNAMIC NARRATIVE
        factors = {"Taste Preference": total_shap_taste, "Pantry Efficiency": total_shap_pantry, "Health Alignment": total_shap_health}
        dominant = max(factors, key=factors.get)
        
        narrative = ""
        if dominant == "Health Alignment" and rag_explanation:
            narrative = f"I chose this primarily because it aligns with the medical insight: **{rag_explanation}**."
        elif dominant == "Pantry Efficiency":
            narrative = f"This is a high-efficiency pick; it uses **{current_p_val:.0%} of your ingredients**, which is top-tier for this set."
        else:
            narrative = f"This matches your 'Taste Signature'—the neural network identified patterns similar to your favorite historical meals."

        # 8. ROADMAP TRACE
        trace = [
            f"1. **Retrieval:** Captured via semantic vector search ({sim:.2f} proximity).",
            f"2. **Attribution:** SHAP analysis shows {dominant} contributed +{factors[dominant]:.2f} to the confidence.",
            f"3. **Pantry Check:** Matched {current_p_val:.0%} of available stocks."
        ]

        # 9. FINAL SCORE ASSEMBLY
        ui_score = 0.60 + (final_neural_score * 0.40)
        
        matched_nutrients = []
        if p_ratio > batch_max_protein * 0.9: matched_nutrients.append("High Protein")
        
        results.append({
            "recipe_id": rid, 
            "title": title, 
            "ingredients": ing_tokens, 
            "missing_ingredients": get_missing_ingredients(ing_tokens, user_pantry),
            "link": f"https://google.com/search?q={title}",
            "nutrition": {"Protein": round(p_ratio*100), "Carbs": round(c_ratio*100), "Fats": round(f_ratio*100)},
            "score": min(ui_score, 0.99), 
            "reason": dominant, 
            "matched_nutrients": list(set(matched_nutrients)),
            "explanation_text": narrative, 
            "decision_trace": trace,
            "attribution": { 
                "Taste": max(0, round(total_shap_taste*100, 1)),
                "Pantry": max(0, round(total_shap_pantry*100, 1)), 
                "Health": max(0, round(total_shap_health*100, 1))
            },
            "match_details": {
                "neural_confidence": f"{final_neural_score:.1%}", 
                "pantry_match_level": f"{current_p_val:.1%}"
            }
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
    
    u_prof = get_user_profile(req.user_id)
    u_pantry = list(set((req.pantry or []) + get_user_pantry(req.user_id)))
    u_diet = req.diet or (u_prof.get("dietary_constraints", [None])[0]) or "Non-Veg"
    u_bmi = u_prof.get("bmi_data", {}).get("bmi", req.bmi)
    blocked = set(x['recipe_id'] for x in get_user_history(req.user_id))
    
    # ✅ RAG Logic: Default is None (Empty) for Auto Mode
    medical_keywords = []
    medical_reasoning = "" 
    
    if req.query_keywords and len(req.query_keywords) > 0:
        logger.info(f"🧬 Smart Mode Detected: Analyzing '{req.query_keywords}'")
        try:
            rag_insight = nutrition_analysis_agent(
                rag_context_snippets=[], 
                user_query=" ".join(req.query_keywords),
                user_profile={'bmi': u_bmi, 'diet': u_diet, 'allergies': req.query_keywords}
            )
            medical_keywords = rag_insight.get("recommended_keywords", [])
            medical_reasoning = rag_insight.get("insight", "Nutritionally balanced.")
        except Exception as e:
            logger.error(f"RAG Agent Failed: {e}")
            medical_reasoning = "Standard Optimization (RAG Unavailable)"

    constraint_vec = None
    kw_str = ""
    all_kws = (req.query_keywords or []) + medical_keywords

    if all_kws:
        kw_str = " ".join(all_kws)
        constraint_vec = text_to_vector(kw_str) 

    for meal_name in ["breakfast", "lunch", "dinner"]:
        budget = req.time_budget
        if meal_name == 'breakfast': budget = max(15, req.time_budget * 0.5)
        
        query_base = f"{meal_name} " + " ".join(req.likes + random.sample(u_pantry, min(3, len(u_pantry))))
        if "south indian" in kw_str.lower(): query_base += " idli dosa sambar vada uttapam chettinad kerala"
        elif "north indian" in kw_str.lower(): query_base += " roti naan paneer dal makhani punjabi paratha"
        elif req.indian_only: query_base += " authentic indian cuisine curry spicy"
        
        vec_context = text_to_vector(query_base)
        
        if constraint_vec is not None: w_query = 0.95; w_history = 0.05
        else: w_query = 0.70; w_history = 0.30

        if u_prof.get("taste_vector"):
            learned_vec = np.array(u_prof.get("taste_vector"), dtype=np.float32)
            v_q = vec_context / (np.linalg.norm(vec_context) + 1e-9)
            v_h = learned_vec / (np.linalg.norm(learned_vec) + 1e-9)
            vec_context = (v_q * w_query) + (v_h * w_history)

        final_search_vec = vec_context
        if constraint_vec is not None:
             final_search_vec = (vec_context * 0.5) + (constraint_vec * 0.5)

        D, I = index.search(final_search_vec.reshape(1,-1), 1000) 
        
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
            
            full_text = (str(title) + " " + str(meta.get('ingredients', []))).lower()
            if u_diet in ['Vegetarian', 'Veg'] and any(x in full_text for x in NON_VEG_KEYWORDS): continue
            elif u_diet == 'Vegan' and any(x in full_text for x in NON_VEGAN_KEYWORDS): continue
            if "no onion" in kw_str.lower() and "onion" in full_text: continue
            if "no garlic" in kw_str.lower() and "garlic" in full_text: continue
            
            user_allergies = req.model_dump().get('profile', {}).get('allergies', [])
            if user_allergies and any(alg.lower() in full_text for alg in user_allergies): continue

            prob = float(meal_probs[i][target_idx])
            threshold = 0.3
            if meal_name == 'breakfast':
                if req.indian_only:
                    snack_prob = float(meal_probs[i][3]); prob = max(prob, snack_prob); threshold = 0.25 
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
            
            # --- Pass Reasoning ---
            scored = score_candidates_neural(candidates, ctx, rag_explanation=medical_reasoning)
            scored.sort(key=lambda x: x['score'], reverse=True)
            
            high_pantry = []
            discovery = []
            for rec in scored:
                p_val = float(rec['match_details']['pantry_match_level'].strip('%'))
                if p_val >= 50.0: high_pantry.append(rec)
                else: discovery.append(rec)
            
            final_list = []
            sel_titles = []
            target_discovery = max(1, int(req.num_recs * 0.2)) 
            target_pantry = req.num_recs - target_discovery
            
            def add_unique(pool, quota):
                count = 0
                for rec in pool:
                    if count >= quota: break
                    t1 = set(rec['title'].lower().split())
                    if any(len(t1 & set(t2.lower().split()))/len(t1 | set(t2.lower().split())) > 0.6 for t2 in sel_titles): continue
                    final_list.append(rec); sel_titles.append(rec['title']); count += 1
                return count

            add_unique(high_pantry, target_pantry)
            add_unique(discovery, target_discovery)
            
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