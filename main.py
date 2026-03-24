"""
main.py  —  FRESH v3.2  (egg/eggplant false-positive fix + eggs plural fix)
=============================================================================
Fixes vs. v3.1:
  1. full_text construction now joins ingredient lists properly instead of
     using str(list), which caused "egg" to match inside "eggplant".
     Both Stage-2 pre-filter AND score_candidates_neural are fixed.
  2. "eggs" (plural) added to NON_VEGAN_KEYWORDS so "2 eggs, beaten" is caught.
  3. Allergy stemming now also checks the plural form explicitly, so
     "peanuts" stored in the list as a token is caught independently of
     the rstrip("s") path.
  4. Diagnostic logger.info call added for peanut/allergy debugging
     (remove after confirming CSR=1.0).
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI
from pydantic import BaseModel
import logging
import faiss
from sentence_transformers import SentenceTransformer
import json
import random
import pandas as pd
from supabase import create_client, Client
import itertools

from nutrition_agent import nutrition_analysis_agent

# ==============================================================================
# LAGRANGIAN SYMBOLIC CONSTRAINT PENALTY (SCP)
# ==============================================================================
class LagrangianSCP:
    def __init__(self, tau=0.98, rho=0.05, lambda_init=5.0, lambda_max=25.0):
        self.tau = tau
        self.rho = rho
        self.lambda_max = lambda_max
        self.lambdas = {"diet": lambda_init, "allergy": lambda_init, "avoidance": lambda_init}

    def penalty(self, violation_type):
        return -self.lambdas.get(violation_type, self.lambdas["diet"])

    def update(self, violations):
        for k, v in violations.items():
            if k in self.lambdas:
                g_k = float(v) - (1.0 - self.tau)
                self.lambdas[k] = max(0.0, min(self.lambda_max,
                                               self.lambdas[k] + self.rho * g_k))

    def state(self):
        return {k: round(v, 3) for k, v in self.lambdas.items()}


lagrangian_scp = LagrangianSCP()


# ==============================================================================
# ENTROPY-ADAPTIVE ONLINE LEARNING RATE
# ==============================================================================
def compute_adaptive_alpha(user_vec, action_type):
    abs_v = np.abs(user_vec)
    p = abs_v / (abs_v.sum() + 1e-9)
    log_n = np.log(len(p) + 1e-9)
    H_norm = float(-np.sum(p * np.log(p + 1e-9)) / log_n)
    bounds = {"cook": (0.05, 0.20), "like": (0.10, 0.30), "dislike": (0.05, 0.15)}
    alpha_min, alpha_max = bounds.get(action_type, (0.05, 0.20))
    return alpha_min + (alpha_max - alpha_min) * H_norm


# ==============================================================================
# MULTI-EVIDENCE ATTENTIVE RAG AGGREGATION  (paper Eq. 3-4)
# ==============================================================================
_EMBED_DIM = 384
_SQRT_D    = float(np.sqrt(_EMBED_DIM))


def multi_evidence_aggregate(recipe_vec, chunk_vecs):
    if not chunk_vecs:
        return np.zeros(_EMBED_DIM, dtype=np.float32)
    scores = np.array([float(np.dot(recipe_vec, cv)) for cv in chunk_vecs]) / _SQRT_D
    scores -= scores.max()
    exp_s  = np.exp(scores)
    alphas = exp_s / (exp_s.sum() + 1e-9)
    aggregated = sum(float(a) * cv for a, cv in zip(alphas, chunk_vecs))
    norm = np.linalg.norm(aggregated)
    return (aggregated / norm).astype(np.float32) if norm > 1e-9 else np.zeros(_EMBED_DIM, dtype=np.float32)


# ==============================================================================
# SHAPLEY COUNTERFACTUAL RECOURSE
# ==============================================================================
def generate_counterfactuals(combo_scores, feature_names):
    n = len(feature_names)
    base = combo_scores.get(tuple([0] * n), 0.0)
    results = []
    for i, name in enumerate(feature_names):
        only_i = [0] * n
        only_i[i] = 1
        gain = combo_scores.get(tuple(only_i), base) - base
        results.append((name, gain))
    return sorted(results, key=lambda x: x[1], reverse=True)


def calculate_exact_shapley(model, inputs, feature_names):
    n = len(inputs)
    all_combinations = list(itertools.product([0, 1], repeat=n))
    null_inputs = [torch.zeros_like(x) for x in inputs]
    batch_per_input = []
    for j in range(n):
        slices = [inputs[j] if combo[j] == 1 else null_inputs[j]
                  for combo in all_combinations]
        batch_per_input.append(torch.cat(slices, dim=0))
    with torch.no_grad():
        all_scores = model(*batch_per_input).squeeze(-1)
    combo_scores   = {combo: all_scores[i].item() for i, combo in enumerate(all_combinations)}
    baseline_score = combo_scores[tuple([0] * n)]
    full_score     = combo_scores[tuple([1] * n)]
    shap_values = {}
    for i, name in enumerate(feature_names):
        marginal = 0.0
        count    = 0
        for combo in all_combinations:
            if combo[i] == 1:
                continue
            combo_with    = list(combo)
            combo_with[i] = 1
            marginal += combo_scores[tuple(combo_with)] - combo_scores[combo]
            count    += 1
        shap_values[name] = marginal / count
    shap_sum = sum(shap_values.values())
    expected = full_score - baseline_score
    if abs(shap_sum - expected) > 1e-4:
        logger.warning(f"Shapley efficiency violated: sum={shap_sum:.6f}, expected={expected:.6f}")
    return shap_values, baseline_score, combo_scores


# ==============================================================================
# LLM NARRATIVE
# ==============================================================================
def generate_llm_narrative(shap_vals, meta, rag_explanation, current_p_val):
    top_factor = max(shap_vals, key=shap_vals.get) if shap_vals else "Taste"
    prompt = (
        f"Explain why recipe '{meta.get('title')}' is a good choice based on:\n"
        f"- Taste Match: {shap_vals.get('Taste', 0):.2f}\n"
        f"- Pantry Match ({current_p_val:.0%}): {shap_vals.get('Pantry', 0):.2f}\n"
        f"- Health Alignment: {shap_vals.get('Health', 0):.2f}\n"
        f"- Medical Context: {rag_explanation or 'General goals'}\n\n"
        "Tone: Professional Chef. Concise (1 sentence). "
        "Do NOT mention 'SHAP', 'scores', or 'math'."
    )
    try:
        from Fresh_metrics import llm_judge_relevance
        response = llm_judge_relevance(prompt, "Chef Analysis")
        if isinstance(response, dict):
            return response.get("explanation") or response.get("content") or str(response)
        return str(response).strip()
    except Exception as e:
        logger.error(f"LLM Narrative Error: {e}")
        return f"This is a strong {top_factor.lower()} match for your current goals."


# ==============================================================================
# CONFIGURATION
# ==============================================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SUPABASE_URL  = "https://wrwbqawmwqcknqlntpfb.supabase.co"
SUPABASE_KEY  = "sb_publishable_tEak2s2lQyaMgBu4fQmG6Q_IjkcWLZU"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

DATA_DIR        = r"/data1/home/sathvik/Documents/FRESH"
VECS_PATH       = os.path.join(DATA_DIR, "recipe_embeddings_2m.pt")
METADATA_PATH   = os.path.join(DATA_DIR, "recipe_metadata_2m.pkl")
RANKER_PATH     = "fresh_model_epoch_5.pth"
CLASSIFIER_PATH = "meal_classifier.pth"


# ==============================================================================
# MODEL ARCHITECTURE
# ==============================================================================
class FeatureInteractionLayer(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        attn_out, _ = self.multihead_attn(x, x, x)
        return self.norm(x + attn_out)


class FRESH_Network(nn.Module):
    def __init__(self, embedding_dim=384):
        super().__init__()
        self.taste_encoder = nn.Sequential(
            nn.Linear(embedding_dim * 2, 256),
            nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, 128),
        )
        self.constraint_encoder = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, 128),
        )
        self.wide_encoder = nn.Sequential(nn.Linear(2, 16), nn.ReLU())
        self.interaction   = FeatureInteractionLayer(dim=128, num_heads=4)
        self.head = nn.Sequential(
            nn.Linear(128 * 2 + 16 + 1, 64),
            nn.LayerNorm(64), nn.GELU(),
            nn.Linear(64, 1), nn.Sigmoid(),
        )

    def forward(self, recipe_vecs, user_vecs, constraint_vecs, explicit_feats, align_score):
        taste_input    = torch.cat([recipe_vecs, user_vecs], dim=1)
        taste_emb      = self.taste_encoder(taste_input)
        constraint_emb = self.constraint_encoder(constraint_vecs)
        seq_input      = torch.stack([taste_emb, constraint_emb], dim=1)
        interacted     = self.interaction(seq_input)
        flat_features  = interacted.view(interacted.size(0), -1)
        wide_out       = self.wide_encoder(explicit_feats)
        final_input    = torch.cat([flat_features, wide_out, align_score], dim=1)
        return self.head(final_input)


class MealClassifier(nn.Module):
    def __init__(self, input_dim=384, num_classes=4):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, num_classes), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.network(x)


# ==============================================================================
# LOAD DATA
# ==============================================================================
print("Loading recipe data ...")
try:
    if VECS_PATH.endswith(".pt"):
        recipe_vectors = torch.load(VECS_PATH, map_location="cpu").numpy().astype(np.float32)
    else:
        recipe_vectors = np.load(VECS_PATH)
        if hasattr(recipe_vectors, "files"):
            recipe_vectors = recipe_vectors[recipe_vectors.files[0]]
        recipe_vectors = recipe_vectors.astype(np.float32)
    recipe_ids = np.arange(recipe_vectors.shape[0])
    norms = np.linalg.norm(recipe_vectors, axis=1, keepdims=True)
    recipe_vectors_norm = recipe_vectors / np.maximum(norms, 1e-12)
    d = recipe_vectors.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(recipe_vectors_norm)
    print(f"Data loaded: {len(recipe_ids):,} recipes  (dim={d})")
except Exception as e:
    print(f"CRITICAL ERROR LOADING DATA: {e}")
    recipe_vectors = []
    index = None

meta_map = {}
if os.path.exists(METADATA_PATH):
    try:
        if METADATA_PATH.endswith(".pkl"):
            df = pd.read_pickle(METADATA_PATH)
            meta_map = {str(k): v for k, v in df.T.to_dict().items()}
        else:
            with open(METADATA_PATH, "r") as f:
                data = json.load(f)
                for m in data:
                    meta_map[str(m.get("recipe_id"))] = m
        print(f"Metadata loaded: {len(meta_map):,} entries")
    except Exception as e:
        print(f"Error loading metadata: {e}")


# ==============================================================================
# HELPER: build a safe full_text string from title + ingredients
# ==============================================================================
def _build_full_text(title: str, ingredients_raw) -> str:
    """
    FIX v3.2: Previously used str(list) which produced bracket/quote artefacts
    like "['eggplant', 'onion']" — causing "egg" to match inside "eggplant".

    FIX v3.3: Plant-based compound ingredients (e.g. "coconut milk") whose
    sub-tokens appear in NON_VEGAN_KEYWORDS are normalised to a single safe
    token (e.g. "coconut_milk") before keyword scanning, preventing false
    vegan violations.

    Now we join list items with spaces so each token is isolated. A word-boundary
    check (wrapped in spaces) is then used for single-character-sensitive keywords
    like "egg" vs "eggplant".
    """
    if isinstance(ingredients_raw, list):
        ings_str = " ".join(str(x) for x in ingredients_raw)
    else:
        ings_str = str(ingredients_raw)
    combined = f"{title.lower()} {ings_str.lower()}"
    # Normalise plant-based compounds before keyword scanning
    for compound, safe_token in VEGAN_SAFE_COMPOUNDS.items():
        combined = combined.replace(compound, safe_token)
    # Surround with spaces so boundary checks like " egg " work at string edges too
    return f" {combined} "


def _contains_whole_word(text: str, word: str) -> bool:
    """
    Check that `word` appears as a whole token in `text`.
    `text` must already be lowercased and space-padded (output of _build_full_text).
    Handles both ' egg ' and ' eggs ' without needing separate entries.
    """
    # Check exact word and common plural/singular variants
    return (f" {word} " in text
            or f" {word}s " in text
            or f" {word}," in text
            or f" {word})" in text
            or f"({word} " in text)


def _has_any_keyword(full_text: str, keyword_set: set) -> bool:
    """Whole-word match for any keyword in the set against padded full_text."""
    return any(_contains_whole_word(full_text, kw) for kw in keyword_set)


# ==============================================================================
# SAFETY KEYWORD SETS  (Stage 2 hard pre-filter)
# ==============================================================================
NON_VEG_KEYWORDS = {
    # Common English meat/seafood terms
    "chicken", "beef", "pork", "fish", "salmon", "shrimp", "crab",
    "bacon", "ham", "sausage", "meat", "lamb", "tuna", "anchovy",
    "gelatin", "turkey", "duck", "goose", "prawn",
    "lobster", "squid", "octopus", "mussel", "clam",
    # FIX v3.3: Indian subcontinent terms absent from original set —
    # caused "Shammi Kabab (Mutton Kabab)" to slip past the Vegetarian filter
    "mutton", "keema", "mince", "kheema",
    # Other common non-veg terms
    "seafood", "sardine", "herring", "mackerel", "tilapia",
}

# FIX v3.2: added "egg" and "eggs" explicitly so both singular and plural
# are caught. "egg" alone previously matched "eggplant" via str(list) artefact;
# the new _build_full_text + _contains_whole_word approach prevents that.
NON_VEGAN_KEYWORDS = NON_VEG_KEYWORDS | {
    "milk", "butter", "cream", "cheese", "honey", "yogurt", "ghee",
    "paneer", "whey", "casein", "lactose", "egg", "eggs",
}

# FIX v3.3: plant-based compound ingredients whose sub-tokens appear in
# NON_VEGAN_KEYWORDS (e.g. "coconut milk" contains "milk").
# These are normalised to a single safe token before keyword scanning.
VEGAN_SAFE_COMPOUNDS = {
    "coconut milk":   "coconut_milk",
    "almond milk":    "almond_milk",
    "oat milk":       "oat_milk",
    "soy milk":       "soy_milk",
    "rice milk":      "rice_milk",
    "cashew milk":    "cashew_milk",
    "coconut cream":  "coconut_cream",
    "coconut butter": "coconut_butter",
    "peanut butter":  "peanut_butter",
    "nut butter":     "nut_butter",
}


# ==============================================================================
# LOAD MODELS
# ==============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

fresh_model = FRESH_Network()
try:
    fresh_model.load_state_dict(torch.load(RANKER_PATH, map_location=device))
    print("RANKER LOADED SUCCESSFULLY")
except Exception:
    print("RANKER MISSING -- using random weights")
fresh_model.to(device).eval()

meal_model = MealClassifier()
try:
    meal_model.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=device))
    print("MEAL CLASSIFIER LOADED SUCCESSFULLY")
except Exception:
    print("MEAL CLASSIFIER MISSING")
meal_model.to(device).eval()

embedder = SentenceTransformer("all-MiniLM-L6-v2")


def text_to_vector(text):
    return embedder.encode(text)


macro_vectors = {
    "Protein": embedder.encode("meat fish chicken beef tofu lentils beans eggs protein muscle"),
    "Carbs":   embedder.encode("rice pasta bread potato sugar grains flour wheat carbs starch"),
    "Fats":    embedder.encode("oil butter cream cheese nuts avocado fat fry rich grease"),
}
indian_anchor = embedder.encode(
    "authentic indian cuisine curry masala spices tandoori south indian "
    "north indian biryani dal roti"
)


# ==============================================================================
# DATABASE HELPERS
# ==============================================================================
def get_user_profile(user_id):
    try:
        r = supabase.table("users").select("*").eq("id", user_id).execute()
        return r.data[0] if r.data else {}
    except Exception:
        return {}


def get_user_pantry(user_id):
    try:
        r = supabase.table("pantry_items").select("ingredient_name").eq("user_id", user_id).execute()
        return [i["ingredient_name"] for i in r.data]
    except Exception:
        return []


def get_user_history(user_id):
    try:
        r = supabase.table("interactions").select("recipe_id").eq("user_id", user_id).limit(50).execute()
        return r.data
    except Exception:
        return []


def update_user_taste_vector(user_id, recipe_id, action_type="cook"):
    try:
        r_indices = np.where(recipe_ids == int(recipe_id))[0]
        if len(r_indices) == 0:
            return
        r_vec     = recipe_vectors[r_indices[0]]
        user_data = supabase.table("users").select("taste_vector").eq("id", user_id).execute()
        current   = user_data.data[0].get("taste_vector")
        if current and len(current) > 0:
            user_vec_arr = np.array(current, dtype=np.float32)
            alpha        = compute_adaptive_alpha(user_vec_arr, action_type)
            if action_type == "dislike":
                new_vec = user_vec_arr - (alpha * r_vec)
            else:
                new_vec = (1 - alpha) * user_vec_arr + alpha * r_vec
        else:
            new_vec = r_vec
            alpha   = 0.10
        norm = np.linalg.norm(new_vec)
        if norm > 0:
            new_vec = new_vec / norm
        supabase.table("users").update({"taste_vector": new_vec.tolist()}).eq("id", user_id).execute()
        logger.info(f"Taste vector updated for {user_id} ({action_type}, alpha={alpha:.3f})")
    except Exception as e:
        logger.error(f"Online Learning Failed: {e}")


def get_missing_ingredients(recipe_ings, pantry_items):
    missing    = []
    pantry_set = {p.lower().strip() for p in pantry_items}
    staples    = {"salt", "sugar", "water", "oil", "ice", "pepper", "ghee",
                  "turmeric", "chili powder", "mustard seeds"}
    ing_list   = recipe_ings.split(",") if isinstance(recipe_ings, str) else recipe_ings
    for ing in ing_list:
        ing_clean = ing.lower().strip()
        if not any(p in ing_clean for p in pantry_set) and ing_clean not in staples:
            missing.append(ing.title())
    return missing[:5]


def fuzzy_pantry_match(ing_tokens, pantry_items, title=""):
    pantry_set = {p.lower().strip() for p in pantry_items}
    text       = (" ".join(ing_tokens + [title]) if isinstance(ing_tokens, list)
                  else str(ing_tokens) + " " + title).lower()
    return min(sum(1 for p in pantry_set if p in text), 5)


# ==============================================================================
# ALLERGY CHECK HELPER
# ==============================================================================
def _allergy_present(full_text: str, allergen: str) -> bool:
    """
    FIX v3.2: Check both the raw allergen and its stemmed form (rstrip 's'),
    plus explicit plural, so 'peanut' catches 'peanuts' and vice versa.
    Uses whole-word matching to avoid false positives.
    `full_text` must be space-padded (output of _build_full_text).
    """
    allergen_lower  = allergen.lower()
    allergen_stem   = allergen_lower.rstrip("s")       # "peanuts" → "peanut"
    allergen_plural = allergen_stem + "s"              # "peanut"  → "peanuts"

    # Diagnostic log — remove after confirming CSR=1.0
    for form in {allergen_lower, allergen_stem, allergen_plural}:
        if _contains_whole_word(full_text, form):
            logger.debug(f"ALLERGY HIT: '{form}' found in: {full_text[:120]}")
            return True
    return False


# ==============================================================================
# NEURAL SCORING ENGINE
# ==============================================================================
def score_candidates_neural(filtered_candidates, preferences, rag_explanation="", explain=True):
    if not filtered_candidates:
        return []

    user_bmi       = preferences.get("bmi", 22.0)
    query_keywords = preferences.get("query_keywords", [])
    user_pantry    = preferences.get("pantry", [])
    user_allergies = preferences.get("allergies", [])

    ablate_scp = preferences.get("ablate_scp", False)
    ablate_med = preferences.get("ablate_medical", False)
    ablate_pan = preferences.get("ablate_pantry", False)

    raw_constraint = preferences.get("constraint_vec")
    constraint_vec = (np.array(raw_constraint, dtype=np.float32)
                      if raw_constraint is not None
                      else np.zeros(_EMBED_DIM, dtype=np.float32))

    raw_user  = preferences.get("user_vec")
    user_vec  = (np.array(raw_user, dtype=np.float32)
                 if raw_user is not None
                 else np.zeros(_EMBED_DIM, dtype=np.float32))

    medical_chunk_vecs = preferences.get("medical_chunk_vecs", [])
    results = []

    all_pantry, all_prot = [], []
    for _, _, idx, meta in filtered_candidates:
        p = fuzzy_pantry_match(
            meta.get("ingredients", []), user_pantry, meta.get("title", "")) / 5.0
        all_pantry.append(p)
        v  = recipe_vectors[idx]
        pr = max(0.01, float(np.dot(v, macro_vectors["Protein"])))
        total = (pr
                 + max(0.01, float(np.dot(v, macro_vectors["Carbs"])))
                 + max(0.01, float(np.dot(v, macro_vectors["Fats"]))))
        all_prot.append(pr / total)
    batch_max_protein = max(all_prot) if all_prot else 0

    for rid, sim, idx, meta in filtered_candidates:
        vec        = recipe_vectors[idx]
        ing_tokens = meta.get("ingredients", [])
        title      = meta.get("title", "")

        current_p_val = min(fuzzy_pantry_match(ing_tokens, user_pantry, title) / 5.0, 1.0)
        if ablate_pan:
            current_p_val = 0.5

        prot  = max(0.01, float(np.dot(vec, macro_vectors["Protein"])))
        carb  = max(0.01, float(np.dot(vec, macro_vectors["Carbs"])))
        fat   = max(0.01, float(np.dot(vec, macro_vectors["Fats"])))
        total_m  = prot + carb + fat
        p_ratio  = prot / total_m
        c_ratio  = carb / total_m
        f_ratio  = fat  / total_m

        h_score       = 1.2 if (user_bmi >= 25.0 and p_ratio > 0.4) else 1.0
        alignment_val = float(np.dot(vec, constraint_vec))

        if ablate_med:
            h_score       = 1.0
            alignment_val = 0.0

        # ── LAGRANGIAN SCP ────────────────────────────────────────────────
        scp_penalty         = 0.0
        violations_detected = {"diet": 0, "allergy": 0, "avoidance": 0}

        if not ablate_scp:
            # FIX v3.2: use _build_full_text (space-joined, space-padded)
            full_text = _build_full_text(title, ing_tokens)
            diet      = preferences.get("diet", "Non-Veg")

            # Diet constraint — whole-word match prevents eggplant false positive
            if diet in ("Vegetarian", "Veg") and _has_any_keyword(full_text, NON_VEG_KEYWORDS):
                scp_penalty += lagrangian_scp.penalty("diet")
                violations_detected["diet"] = 1
            elif diet == "Vegan" and _has_any_keyword(full_text, NON_VEGAN_KEYWORDS):
                scp_penalty += lagrangian_scp.penalty("diet")
                violations_detected["diet"] = 1

            # Avoidance from query keywords
            for kw in query_keywords:
                if kw.lower() == "no onion" and _contains_whole_word(full_text, "onion"):
                    scp_penalty += lagrangian_scp.penalty("avoidance")
                    violations_detected["avoidance"] = 1
                if "diabetic" in kw.lower() and _contains_whole_word(full_text, "sugar"):
                    scp_penalty += lagrangian_scp.penalty("allergy")
                    violations_detected["allergy"] = 1

            # FIX v3.2: allergy check uses _allergy_present (whole-word + plural)
            if user_allergies:
                for allergen in user_allergies:
                    if _allergy_present(full_text, allergen):
                        scp_penalty += lagrangian_scp.penalty("allergy")
                        violations_detected["allergy"] = 1
                        break

        lagrangian_scp.update(violations_detected)

        # ── MULTI-EVIDENCE ATTENTION ──────────────────────────────────────
        if medical_chunk_vecs and not ablate_med:
            effective_constraint = multi_evidence_aggregate(vec, medical_chunk_vecs)
            effective_alignment  = float(np.dot(vec, effective_constraint))
        else:
            effective_constraint = constraint_vec
            effective_alignment  = alignment_val

        # ── TENSOR CONVERSION ─────────────────────────────────────────────
        t_recipe = torch.tensor(vec,                  dtype=torch.float32).unsqueeze(0).to(device)
        t_user   = torch.tensor(user_vec,             dtype=torch.float32).unsqueeze(0).to(device)
        t_const  = torch.tensor(effective_constraint, dtype=torch.float32).unsqueeze(0).to(device)
        t_expl   = torch.tensor([current_p_val, h_score], dtype=torch.float32).unsqueeze(0).to(device)
        t_align  = torch.tensor([effective_alignment],    dtype=torch.float32).unsqueeze(0).to(device)

        final_shap      = {"Taste": 0.0, "Pantry": 0.0, "Health": 0.0}
        trace           = []
        human_narrative = ""
        base_score      = 0.5

        if not explain:
            with torch.no_grad():
                raw_score = fresh_model(t_recipe, t_user, t_const, t_expl, t_align).item()
            final_neural_score = raw_score + scp_penalty
            human_narrative    = "Optimization Mode (Fast)"
            top_factor         = "Neural Score"
        else:
            feature_groups = ["Taste", "User", "Medical", "Stats", "Align"]
            inputs = [t_recipe, t_user, t_const, t_expl, t_align]
            shap_vals, base_score, combo_scores = calculate_exact_shapley(
                fresh_model, inputs, feature_groups)
            final_shap = {
                "Taste":  shap_vals["Taste"] + shap_vals["User"],
                "Pantry": shap_vals["Stats"],
                "Health": shap_vals["Medical"] + shap_vals["Align"],
            }
            final_neural_score = base_score + sum(shap_vals.values()) + scp_penalty
            top_factor         = max(final_shap, key=final_shap.get)
            human_narrative    = generate_llm_narrative(
                final_shap, meta, rag_explanation, current_p_val)
            counterfacts = generate_counterfactuals(combo_scores, feature_groups)
            cf_labels = {
                "Taste":   "broader ingredient variety",
                "User":    "more cooking history",
                "Medical": "medical/dietary context",
                "Stats":   "pantry coverage",
                "Align":   "dietary alignment",
            }
            top_cf_name, top_cf_gain = counterfacts[0]
            missing_ings = get_missing_ingredients(ing_tokens, user_pantry)
            trace = [
                f"1. **Baseline:** Confidence at {base_score:.2f}",
                f"2. **Driver:** {top_factor} (+{final_shap[top_factor]:.2f})",
                (f"3. **Counterfactual:** Improving **{cf_labels.get(top_cf_name, top_cf_name)}** "
                 f"alone adds +{top_cf_gain:.3f} to confidence"),
                (f"4. **Pantry Tip:** Adding *{missing_ings[0]}* could unlock a full pantry match"
                 if missing_ings else "4. **Pantry:** Fully matched - no shopping needed"),
            ]

        total_abs = sum(abs(v) for v in final_shap.values())
        denom     = total_abs if total_abs > 0 else 1e-9
        normalized_attribution = {
            k: round((abs(v) / denom) * 100, 1) for k, v in final_shap.items()
        }

        ui_score = 0.60 + (final_neural_score * 0.40)
        matched_nutrients = []
        txt = (title + " " + str(ing_tokens)).lower()
        for k in query_keywords:
            if k.lower() in txt:
                ui_score += 0.05
                matched_nutrients.append(k.title())
        if p_ratio > batch_max_protein * 0.9:
            matched_nutrients.append("High Protein")
        final_ui_score = max(0.01, min(ui_score, 0.99))

        results.append({
            "recipe_id":           rid,
            "title":               title,
            "ingredients":         ing_tokens,
            "missing_ingredients": get_missing_ingredients(ing_tokens, user_pantry),
            "link":                f"https://google.com/search?q={title}",
            "nutrition": {
                "Protein": round(p_ratio * 100),
                "Carbs":   round(c_ratio * 100),
                "Fats":    round(f_ratio * 100),
            },
            "score":             final_ui_score,
            "reason":            top_factor,
            "matched_nutrients": list(set(matched_nutrients)),
            "explanation_text":  human_narrative,
            "decision_trace":    trace,
            "attribution":       normalized_attribution,
            "match_details": {
                "neural_confidence": f"{final_neural_score:.1%}",
                "pantry_match_level": f"{current_p_val:.1%}",
            },
        })

    return results


# ==============================================================================
# FASTAPI APPLICATION
# ==============================================================================
app = FastAPI()


class MealPlanRequest(BaseModel):
    user_id:        str
    likes:          list  = []
    pantry:         list  = None
    diet:           str   = None
    time_budget:    int   = 60
    num_recs:       int   = 3
    regenerate:     bool  = False
    indian_only:    bool  = False
    bmi:            float = 22.0
    query_keywords: list  = []
    explain:        bool  = True
    allergies:      list  = []
    # Research ablation flags
    ablate_scp:     bool  = False
    ablate_medical: bool  = False
    ablate_pantry:  bool  = False


class CookEvent(BaseModel):
    user_id:          str
    recipe_id:        str
    rating:           int = 5
    interaction_type: str = "cook"
    recipe_title:     str = ""


@app.get("/")
def root():
    return {"status": "online", "message": "FRESH AI Connected"}


@app.post("/log_cooking")
def log_cooking(event: CookEvent):
    try:
        supabase.table("interactions").insert({
            "user_id":          event.user_id,
            "recipe_id":        event.recipe_id,
            "interaction_type": event.interaction_type,
            "recipe_title":     event.recipe_title,
        }).execute()
        update_user_taste_vector(event.user_id, event.recipe_id, event.interaction_type)
        return {"status": "Logged & Learned"}
    except Exception as e:
        return {"status": "Error", "details": str(e)}


@app.post("/generate_meal_plan")
def generate_meal_plan(req: MealPlanRequest):
    meal_map_indices = {"breakfast": 0, "lunch": 1, "dinner": 2, "snack": 3}
    full_plan = {}

    # ── User context ──────────────────────────────────────────────────────────
    u_prof = get_user_profile(req.user_id)
    u_diet = req.diet or (
        u_prof.get("dietary_constraints", ["Non-Veg"])[0]
        if isinstance(u_prof, dict) else "Non-Veg"
    )
    u_bmi = req.bmi if req.bmi != 22.0 else (
        u_prof.get("bmi_data", {}).get("bmi", 22.0)
        if isinstance(u_prof, dict) else 22.0
    )
    db_pantry = get_user_pantry(req.user_id)
    u_pantry  = list(set((req.pantry or []) + (db_pantry if isinstance(db_pantry, list) else [])))
    blocked   = {x["recipe_id"] for x in get_user_history(req.user_id)}
    u_allergies = req.allergies or []

    # ── RAG / Medical grounding ───────────────────────────────────────────────
    medical_reasoning  = ""
    medical_keywords   = []
    medical_chunk_vecs = []

    if req.query_keywords:
        logger.info(f"Smart Mode: RAG for {req.query_keywords}")
        try:
            rag_insight = nutrition_analysis_agent(
                user_query=" ".join(req.query_keywords),
                user_profile={"bmi": u_bmi, "diet": u_diet, "allergies": u_allergies},
            )
            medical_keywords   = rag_insight.get("recommended_keywords", [])
            medical_reasoning  = rag_insight.get("insight", "Nutritionally balanced.")
            medical_chunk_vecs = rag_insight.get("retrieved_chunk_vecs", [])
            if not medical_chunk_vecs:
                raw_chunks = rag_insight.get("retrieved_chunks", [])
                if raw_chunks:
                    logger.warning("retrieved_chunk_vecs missing -- re-encoding raw chunks")
                    medical_chunk_vecs = [text_to_vector(c) for c in raw_chunks if c]
            logger.info(f"Multi-evidence: {len(medical_chunk_vecs)} chunk vectors ready")
        except Exception as e:
            logger.error(f"Agent Execution Failed: {e}")
            medical_reasoning = "Standard Optimization"

    # ── Constraint vector ─────────────────────────────────────────────────────
    constraint_vec = None
    kw_str         = ""
    all_kws        = (req.query_keywords or []) + medical_keywords
    if all_kws:
        kw_str         = " ".join(all_kws)
        constraint_vec = text_to_vector(kw_str)

    # ── Per-meal generation loop ──────────────────────────────────────────────
    for meal_name in ("breakfast", "lunch", "dinner"):
        budget = req.time_budget
        if meal_name == "breakfast":
            budget = max(15, req.time_budget * 0.5)

        pantry_sample = random.sample(u_pantry, min(3, len(u_pantry)))
        query_base    = f"{meal_name} " + " ".join(req.likes + pantry_sample)
        if   "south indian" in kw_str.lower():
            query_base += " idli dosa sambar vada uttapam chettinad kerala"
        elif "north indian" in kw_str.lower():
            query_base += " roti naan paneer dal makhani punjabi paratha"
        elif req.indian_only:
            query_base += " authentic indian cuisine curry spicy"

        vec_context = text_to_vector(query_base)

        w_query   = 0.95 if constraint_vec is not None else 0.70
        w_history = 1.0 - w_query
        if u_prof.get("taste_vector"):
            learned_vec = np.array(u_prof["taste_vector"], dtype=np.float32)
            v_q = vec_context / (np.linalg.norm(vec_context) + 1e-9)
            v_h = learned_vec  / (np.linalg.norm(learned_vec)  + 1e-9)
            vec_context = v_q * w_query + v_h * w_history

        final_search_vec = (
            (vec_context * 0.5) + (constraint_vec * 0.5)
            if constraint_vec is not None
            else vec_context
        )

        D, I = index.search(final_search_vec.reshape(1, -1), 1000)
        candidates    = []
        seen          = set()
        valid_indices = [idx for idx in I[0] if idx >= 0]
        if not valid_indices:
            full_plan[meal_name] = []
            continue

        candidate_vecs = torch.tensor(recipe_vectors[valid_indices]).to(device)
        with torch.no_grad():
            meal_probs = meal_model(candidate_vecs)
        target_idx = meal_map_indices[meal_name]

        for i, real_idx in enumerate(valid_indices):
            rid  = str(recipe_ids[real_idx])
            if rid in blocked:
                continue
            meta  = meta_map.get(rid, {})
            title = meta.get("title", "").strip()
            if title in seen:
                continue

            # FIX v3.2: use _build_full_text throughout (space-joined, padded)
            full_text = _build_full_text(title, meta.get("ingredients", []))

            # ── Stage 2 hard pre-filter ───────────────────────────────────
            if not req.ablate_scp:
                # Diet filter — whole-word match
                if u_diet in ("Vegetarian", "Veg") and _has_any_keyword(full_text, NON_VEG_KEYWORDS):
                    continue
                elif u_diet == "Vegan" and _has_any_keyword(full_text, NON_VEGAN_KEYWORDS):
                    continue

                # Avoidance from query
                if "no onion" in kw_str.lower() and _contains_whole_word(full_text, "onion"):
                    continue
                if "no garlic" in kw_str.lower() and _contains_whole_word(full_text, "garlic"):
                    continue

                # FIX v3.2: allergy pre-filter — whole-word + plural aware
                if u_allergies:
                    if any(_allergy_present(full_text, a) for a in u_allergies):
                        continue

            # Meal-type gate
            prob      = float(meal_probs[i][target_idx])
            threshold = 0.3
            if meal_name == "breakfast" and req.indian_only:
                snack_prob = float(meal_probs[i][3])
                prob       = max(prob, snack_prob)
                threshold  = 0.25
            if meal_name in str(meta.get("meal_type", "")).lower():
                prob += 0.3
            if prob < threshold:
                continue

            # Time budget gate
            r_time = meta.get("time_minutes", 0)
            if meal_name != "breakfast" and r_time > 0 and r_time > budget + 15:
                continue

            seen.add(title)
            candidates.append((rid, 0.0, real_idx, meta))

        if not candidates:
            full_plan[meal_name] = []
            continue

        # ── Build scoring context ─────────────────────────────────────────
        ctx = req.model_dump()
        ctx["pantry"]             = u_pantry
        ctx["bmi"]                = u_bmi
        ctx["diet"]               = u_diet
        ctx["allergies"]          = u_allergies
        ctx["constraint_vec"]     = constraint_vec
        ctx["user_vec"]           = vec_context
        ctx["medical_chunk_vecs"] = medical_chunk_vecs

        # 2-Pass Scoring
        fast_scored = score_candidates_neural(candidates, ctx, rag_explanation="", explain=False)
        fast_scored.sort(key=lambda x: x["score"], reverse=True)

        shortlist_ids = {r["recipe_id"] for r in fast_scored[: req.num_recs * 4]}
        shortlist     = [
            (rid, sim, idx, meta)
            for rid, sim, idx, meta in candidates
            if rid in shortlist_ids
        ]

        scored = score_candidates_neural(
            shortlist, ctx, rag_explanation=medical_reasoning, explain=req.explain)
        scored.sort(key=lambda x: x["score"], reverse=True)

        # Diversity-constrained pool split
        high_pantry, discovery = [], []
        for rec in scored:
            p_val = float(rec["match_details"]["pantry_match_level"].strip("%"))
            (high_pantry if p_val >= 50.0 else discovery).append(rec)

        final_list  = []
        sel_titles  = []
        target_disc   = max(1, int(req.num_recs * 0.2))
        target_pantry = req.num_recs - target_disc

        def add_unique(pool, quota):
            count = 0
            for rec in pool:
                if count >= quota:
                    break
                t1 = set(rec["title"].lower().split())
                if any(
                    len(t1 & set(t2.lower().split())) /
                    max(len(t1 | set(t2.lower().split())), 1) > 0.6
                    for t2 in sel_titles
                ):
                    continue
                final_list.append(rec)
                sel_titles.append(rec["title"])
                count += 1
            return count

        add_unique(high_pantry, target_pantry)
        add_unique(discovery,   target_disc)
        needed = req.num_recs - len(final_list)
        if needed > 0:
            remaining = [r for r in high_pantry + discovery if r not in final_list]
            add_unique(remaining, needed)

        final_list.sort(key=lambda x: x["score"], reverse=True)
        full_plan[meal_name] = final_list

    return {"user_id": req.user_id, "meal_plan": full_plan}


# ==============================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)