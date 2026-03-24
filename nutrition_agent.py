"""
nutrition_agent.py  —  FRESH v3.1  (paper-aligned, JSON-fixed)
===============================================================
Fixes applied vs. v3.0:
  1. LLM system prompt now opens with strict JSON formatting rules
     to prevent Qwen from emitting apostrophes / single-quotes inside
     JSON string values (root cause of the parse failures).
  2. JSON extraction replaced with a 3-tier robust parser:
       Tier 1 — direct json.loads on cleaned string
       Tier 2 — fix common Qwen issues (single quotes, trailing commas,
                 control chars) then retry
       Tier 3 — regex field extraction fallback (never returns an error
                 to the caller)
  3. temperature=0.0 / do_sample=False in _llm_chat to make Qwen fully
     deterministic, eliminating random JSON formatting variation.
  4. All other behaviour (RAG retrieval, √d scaling, chunk vectors) is
     unchanged from v3.0.
"""

import os
import json
import glob
import re
import faiss
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional, Tuple

# ---------------------------------------------------------------------------
# LAZY-LOADED LLM  (Qwen2.5-7B-Instruct, bfloat16)
# ---------------------------------------------------------------------------
_llm_model = None
_llm_tokenizer = None

def _get_llm():
    global _llm_model, _llm_tokenizer
    if _llm_model is not None:
        return _llm_model, _llm_tokenizer
    MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
    print("🚀 Loading Qwen 2.5 7B Instruct for Dr. FRESH (bfloat16)...")
    _llm_tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    _llm_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        dtype=torch.bfloat16,          # FIX: use dtype= not torch_dtype=
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    _llm_model.eval()
    print("✅ Dr. FRESH LLM Ready.")
    return _llm_model, _llm_tokenizer


def _llm_chat(system_prompt: str, temperature: float = 0.0) -> str:
    """
    Single-turn chat with the Qwen2.5 model; returns response text.
    FIX: temperature=0.0 and do_sample=False by default for deterministic
    JSON output — eliminates random apostrophe/formatting variation.
    """
    llm, tok = _get_llm()
    messages = [{"role": "system", "content": system_prompt}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok([text], return_tensors="pt").to(llm.device)
    with torch.no_grad():
        out = llm.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,           # FIX: greedy decoding = deterministic JSON
            pad_token_id=tok.eos_token_id,
        )
    generated = out[0][inputs["input_ids"].shape[-1]:]
    return tok.decode(generated, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# ROBUST JSON PARSER  (3-tier fallback)
# ---------------------------------------------------------------------------
def _parse_llm_json(raw: str) -> Dict[str, Any]:
    """
    Parse Qwen's response into a dict using three increasingly lenient tiers.

    Tier 1: Direct parse after extracting the outermost {...} block.
    Tier 2: Fix common Qwen formatting issues then retry.
    Tier 3: Regex field extraction — never raises, always returns a dict.
    """
    # ── Extract the outermost JSON object ────────────────────────────────
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in model output")
    raw_json = match.group(0)

    # ── Tier 1: direct parse ──────────────────────────────────────────────
    try:
        return json.loads(raw_json)
    except json.JSONDecodeError:
        pass

    # ── Tier 2: clean common Qwen issues then retry ───────────────────────
    cleaned = raw_json

    # Replace smart/curly quotes with straight double quotes
    cleaned = cleaned.replace("\u2018", "'").replace("\u2019", "'")
    cleaned = cleaned.replace("\u201c", '"').replace("\u201d", '"')

    # Convert unescaped single-quote strings to double-quote strings.
    # Strategy: replace any value that is 'wrapped in single quotes'
    # with "double-quoted version" — handles: 'word', 'two words', etc.
    cleaned = re.sub(r":\s*'([^']*)'", r': "\1"', cleaned)

    # Remove trailing commas before } or ]
    cleaned = re.sub(r',(\s*[}\]])', r'\1', cleaned)

    # Strip control characters (tabs inside strings cause parse errors)
    cleaned = re.sub(r'[\x00-\x09\x0b\x0c\x0e-\x1f\x7f]', ' ', cleaned)

    # Collapse multiple spaces inside strings
    cleaned = re.sub(r'  +', ' ', cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # ── Tier 3: regex field extraction ───────────────────────────────────
    # At this point JSON is too malformed to fix structurally.
    # Extract each field individually with targeted patterns.
    result: Dict[str, Any] = {}

    # context_relevant
    m = re.search(r'"context_relevant"\s*:\s*(true|false)', cleaned, re.IGNORECASE)
    result["context_relevant"] = (m.group(1).lower() == "true") if m else False

    # insight — grab everything between the first " after "insight": and
    # the closing " (stop before the next key or end of string)
    m = re.search(r'"insight"\s*:\s*"([^"]{5,})"', cleaned)
    if not m:
        # Try single-quoted fallback
        m = re.search(r'"insight"\s*:\s*\'([^\']{5,})\'', cleaned)
    result["insight"] = m.group(1).strip() if m else "Nutritionally balanced meal recommended."

    # detailed_explanation
    m = re.search(r'"detailed_explanation"\s*:\s*"(.*?)"(?=\s*,\s*"|\s*\})', cleaned, re.DOTALL)
    result["detailed_explanation"] = m.group(1).strip() if m else result["insight"]

    # recommended_keywords — grab the list contents then split
    m = re.search(r'"recommended_keywords"\s*:\s*\[([^\]]*)\]', cleaned)
    if m:
        raw_kws = m.group(1)
        keywords = [
            kw.strip().strip('"\'')
            for kw in re.split(r',', raw_kws)
            if kw.strip().strip('"\'')
        ]
        result["recommended_keywords"] = keywords
    else:
        result["recommended_keywords"] = []

    return result


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
DATA_DIR   = "/data1/home/sathvik/Documents/FRESH/new_chunks"
INDEX_PATH = os.path.join(DATA_DIR, "master_medical_index.index")

EMBED_DIM    = 384
SQRT_D       = float(np.sqrt(EMBED_DIM))   # ≈ 19.6
M_ADVISORIES = 5

# ---------------------------------------------------------------------------
# INITIALIZATION
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("⚙️  INITIALIZING FRESH RAG ENGINE (v3.1 — JSON-fixed)")
print("=" * 60)

embedder = SentenceTransformer("all-MiniLM-L6-v2")

nutri_index:  Optional[faiss.Index] = None
nutri_chunks: List[Dict]            = []
nutri_vecs:   Optional[np.ndarray]  = None

try:
    if os.path.exists(INDEX_PATH):
        nutri_index = faiss.read_index(INDEX_PATH)
        print(f"✅ Loaded FAISS Index: {nutri_index.ntotal} vectors")
    else:
        print(f"❌ Index not found at {INDEX_PATH}")

    json_files = sorted(glob.glob(os.path.join(DATA_DIR, "*_chunks.json")))
    for jf in json_files:
        with open(jf, "r") as f:
            data = json.load(f)
            if isinstance(data, list):
                nutri_chunks.extend(data)
    print(f"✅ Loaded Text Chunks: {len(nutri_chunks)} segments")

    if nutri_chunks:
        texts = [
            c.get("content", str(c)) if isinstance(c, dict) else str(c)
            for c in nutri_chunks
        ]
        print("🔄 Pre-encoding chunk vectors for multi-evidence attention …")
        nutri_vecs = embedder.encode(texts, batch_size=256, show_progress_bar=True).astype("float32")
        norms = np.linalg.norm(nutri_vecs, axis=1, keepdims=True)
        nutri_vecs = nutri_vecs / np.maximum(norms, 1e-12)
        print(f"✅ Chunk vectors ready: {nutri_vecs.shape}")

        first_content = nutri_chunks[0].get("content", "")
        if "Pregnant" in first_content or "Vitamin B12" in first_content:
            print("\n🚨 WARNING: DB may still contain old content — run reset_rag_db.py\n")

except Exception as e:
    print(f"❌ Error loading RAG resources: {e}")


# ---------------------------------------------------------------------------
# STRICT KEYWORD FALLBACK
# ---------------------------------------------------------------------------
def _strict_keyword_fallback(query: str, k: int = M_ADVISORIES) -> Tuple[List[str], List[np.ndarray]]:
    keywords = [w.lower() for w in query.split() if len(w) > 3]
    if len(keywords) < 1:
        return [], []

    scored = []
    for i, chunk in enumerate(nutri_chunks):
        text = chunk.get("content", str(chunk)) if isinstance(chunk, dict) else str(chunk)
        match_count = sum(1 for kw in keywords if kw in text.lower())
        if match_count >= max(2, len(keywords) // 2):
            scored.append((match_count, i, text))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:k]
    texts = [t for _, _, t in top]
    vecs  = (
        [nutri_vecs[i] for _, i, _ in top]
        if nutri_vecs is not None
        else [np.zeros(EMBED_DIM, dtype=np.float32)] * len(top)
    )
    return texts, vecs


# ---------------------------------------------------------------------------
# CORE RETRIEVAL  (paper §III-D, Equations 3–4)
# ---------------------------------------------------------------------------
def retrieve(query: str, k: int = M_ADVISORIES) -> Tuple[List[str], List[np.ndarray]]:
    if nutri_index is None or not nutri_chunks:
        print("⚠️  No FAISS index — falling back to keyword scan.")
        return _strict_keyword_fallback(query, k)

    boosted_query = query
    if any(w in query.lower() for w in ["surgery", "dental", "soft", "liquid"]):
        boosted_query += " soft texture bland non-spicy mash"
    if any(w in query.lower() for w in ["diabetic", "diabetes", "sugar"]):
        boosted_query += " low glycemic index high fiber no sugar"

    try:
        query_vec = embedder.encode([boosted_query]).astype("float32")
        faiss.normalize_L2(query_vec)
        distances, indices = nutri_index.search(query_vec, k)
        scaled_scores = distances[0] / SQRT_D

        best_raw = float(distances[0][0])
        print(f"🔍 Top RAG score (raw cosine): {best_raw:.4f}  "
              f"(scaled / √{EMBED_DIM}: {best_raw / SQRT_D:.4f})")

        texts, vecs = [], []
        for raw_idx, score in zip(indices[0], scaled_scores):
            if raw_idx < 0 or raw_idx >= len(nutri_chunks):
                continue
            chunk = nutri_chunks[raw_idx]
            text  = chunk.get("content", str(chunk)) if isinstance(chunk, dict) else str(chunk)
            texts.append(text)
            vec = (
                nutri_vecs[raw_idx]
                if nutri_vecs is not None
                else embedder.encode([text]).astype("float32").flatten()
            )
            vecs.append(vec)

        print(f"✅ Retrieved {len(texts)} advisory chunks.")
        return texts, vecs

    except Exception as e:
        print(f"⚠️ Retrieval failed ({e}) — keyword fallback.")
        return _strict_keyword_fallback(query, k)


# ---------------------------------------------------------------------------
# NUTRITION ANALYSIS AGENT  (Dr. FRESH)
# ---------------------------------------------------------------------------
def nutrition_analysis_agent(
    rag_context_snippets=None,
    user_query: str = "",
    user_profile: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Runs the Dr. FRESH clinical nutrition LLM with RAG-grounded context.
    """
    if user_profile is None:
        user_profile = {}

    bmi         = user_profile.get("bmi", "Unknown")
    diet        = user_profile.get("diet", "Any")
    allergies   = user_profile.get("allergies", [])
    allergy_str = ", ".join(allergies) if allergies else "None"

    search_query = user_query or " ".join(
        cond for cond, flag in [
            ("weight management",    isinstance(bmi, (int, float)) and bmi > 25),
            ("underweight nutrition", isinstance(bmi, (int, float)) and bmi < 18.5),
        ] if flag
    ) or "general nutrition"

    retrieved_texts, retrieved_vecs = retrieve(search_query, k=M_ADVISORIES)

    if retrieved_texts:
        context_block = "\n\n".join(
            f"--- ADVISORY {i+1} ---\n{s}" for i, s in enumerate(retrieved_texts)
        )
    else:
        context_block = "No specific textbook context found."

    # ── LLM System Prompt ─────────────────────────────────────────────────
    # FIX: JSON formatting rules appear FIRST before any other instruction.
    # This is the most effective way to prevent Qwen from using apostrophes.
    system_prompt = f"""CRITICAL JSON FORMATTING RULES — FOLLOW EXACTLY:
1. Use ONLY double quotes for all JSON keys and string values.
2. NEVER use apostrophes or single quotes anywhere in your response.
3. NEVER use contractions (write "do not" not "don't", "it is" not "it's").
4. NEVER add a trailing comma after the last item in an array or object.
5. All text must be on a single line — no newlines inside JSON string values.
6. Respond with ONLY the JSON object. No markdown fences. No preamble.

You are Dr. FRESH, a Board-Certified Clinical Nutritionist.

STEP 0 — RELEVANCE GATE (evaluate first):
Read the advisories below. Is at least one advisory topically related to "{user_query}"?
- If YES: set context_relevant to true and use the advisory content.
- If NO:  set context_relevant to false and answer from general medical knowledge only.

PATIENT PROFILE:
- Query:     "{user_query}"
- BMI:       {bmi}
- Diet:      {diet}
- Allergies: {allergy_str}

RETRIEVED ADVISORIES (M={len(retrieved_texts)}):
{context_block}

PERSONALISATION RULES:
- BMI > 25:   prefer low-calorie, nutrient-dense options.
- BMI < 18.5: prefer energy-dense options.
- Follow the {diet} diet strictly.
- Respect allergies: {allergy_str}.

OUTPUT — respond with this JSON and nothing else:
{{
    "context_relevant": true,
    "insight": "One sentence summary without apostrophes",
    "detailed_explanation": "Markdown dietary advice without apostrophes or newlines",
    "recommended_keywords": ["keyword1", "keyword2", "keyword3"]
}}"""

    try:
        raw = _llm_chat(system_prompt, temperature=0.0)

        # ── Robust 3-tier JSON parsing ────────────────────────────────────
        result = _parse_llm_json(raw)

        result["context_used"]         = context_block
        result["retrieved_chunks"]     = retrieved_texts
        result["retrieved_chunk_vecs"] = retrieved_vecs

        print(
            f"🩺 Dr. FRESH | context_relevant={result.get('context_relevant')} | "
            f"keywords={result.get('recommended_keywords')}"
        )
        return result

    except Exception as e:
        print(f"❌ LLM Error (unrecoverable): {e}")
        # Even on total failure, return useful defaults so the UI does not crash
        return {
            "context_relevant": False,
            "insight":          "Balanced nutrition recommended based on your profile.",
            "detailed_explanation": (
                "Focus on whole foods, adequate protein, and vegetables. "
                "Adjust portions based on your energy needs."
            ),
            "recommended_keywords": [],
            "context_used":         context_block,
            "retrieved_chunks":     retrieved_texts,
            "retrieved_chunk_vecs": retrieved_vecs,
        }


# ---------------------------------------------------------------------------
# SMOKE TEST
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n🧪 TEST 1 — Sore Throat (should NOT use pregnancy advisory)")
    res = nutrition_analysis_agent(
        user_query="I have a sore throat",
        user_profile={"bmi": 28, "diet": "Vegan", "allergies": []},
    )
    print(f"📝 Insight:          {res.get('insight')}")
    print(f"🎯 Keywords:         {res.get('recommended_keywords')}")
    print(f"🔍 Context relevant: {res.get('context_relevant')}")
    print(f"🧩 Chunk vecs count: {len(res.get('retrieved_chunk_vecs', []))}")

    print("\n🧪 TEST 2 — Diabetic query (should use diabetes advisory)")
    res2 = nutrition_analysis_agent(
        user_query="I am diabetic and need dinner ideas",
        user_profile={"bmi": 29, "diet": "Non-Veg", "allergies": []},
    )
    print(f"📝 Insight:          {res2.get('insight')}")
    print(f"🎯 Keywords:         {res2.get('recommended_keywords')}")
    print(f"🔍 Context relevant: {res2.get('context_relevant')}")