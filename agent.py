"""
query_agent.py  —  FRESH v3.0  (paper-aligned)
===============================================
Role in the paper (Fig. 1):
    This is the Logic Agent (A_logic / Vector Encoder) that sits at the
    front of the pipeline.  It translates a vague natural-language user
    query into concrete recipe-search keywords and request parameters
    that the FAISS retrieval stage (Stage 1) can use.

Changes vs. original:
  • Fixed syntax error: stray double-quotes inside the api_key string.
  • Prompt now explicitly incorporates the Nutritionist insight from
    nutrition_analysis_agent (passed in as `nutrition_insight`), so
    the two agents are properly composed as shown in the architecture.
  • Added `constraint_hints` output field so main.py can forward
    constraint-relevant terms to the Lagrangian SCP penalty layer.
  • Added `indian_only` output field for the Indian-mode toggle.
  • model switched to "llama-3.3-70b-versatile" (fast + JSON-reliable).
"""

import os
import json
from groq import Groq

# ---------------------------------------------------------------------------
# CLIENT SETUP
# Preferred: export GROQ_API_KEY="gsk_..." in your shell.
# Fallback   (local dev only — never commit to version control):
# ---------------------------------------------------------------------------
api_key = os.environ.get("GROQ_API_KEY", "")   # ← set via env var

# Only use the hard-coded key when the env var is missing (dev fallback).
# Replace the placeholder below with your actual key for local testing.
if not api_key:
    api_key = "gsk_REPLACE_WITH_YOUR_KEY"       # ← replace, never commit

client = Groq(api_key=api_key)


# ---------------------------------------------------------------------------
# QUERY / LOGIC AGENT
# ---------------------------------------------------------------------------
def query_agent(
    user_query: str,
    user_profile: dict,
    pantry_items: list,
    nutrition_insight: str,
) -> dict:
    """
    Logic Agent (A_logic) — translates the user's natural-language request
    plus the Nutritionist's insight into concrete search parameters for the
    FRESH neural ranking pipeline.

    Parameters
    ----------
    user_query        : Raw text from the chat input box.
    user_profile      : dict with keys 'diet', 'bmi', 'allergies', etc.
    pantry_items      : List of ingredient strings from the user's pantry.
    nutrition_insight : 1-sentence insight from nutrition_analysis_agent.

    Returns
    -------
    dict with keys:
        likes             – list[str]  keyword seeds for FAISS query vector
        time_budget       – int        minutes available for cooking
        meal_type         – str        "breakfast" | "lunch" | "dinner"
        constraint_hints  – list[str]  terms forwarded to Lagrangian SCP
        indian_only       – bool       activate Indian-cuisine anchor vector
        reasoning         – str        agent's self-explanation (debug)
    """

    # Limit pantry to top-50 to stay within token budget
    pantry_str = ", ".join(pantry_items[:50]) if pantry_items else "None"

    diet      = user_profile.get("diet", "Any")
    bmi       = user_profile.get("bmi", "Normal")
    allergies = user_profile.get("allergies", [])
    allergy_str = ", ".join(allergies) if allergies else "None"

    system_prompt = f"""
You are the FRESH AI Logic Agent (Query Translation Unit).

Your role is to bridge the user's request and the Nutritionist's advice
into a precise set of recipe-search parameters.

═══════════════════════════════════════════════════════════════
INPUT CONTEXT
═══════════════════════════════════════════════════════════════
User Query      : "{user_query}"
Diet Type       : {diet}
BMI / Goal      : {bmi}
Allergies       : {allergy_str}
Pantry (top-50) : {pantry_str}
Nutritionist    : "{nutrition_insight}"

═══════════════════════════════════════════════════════════════
TRANSLATION RULES
═══════════════════════════════════════════════════════════════
1. KEYWORD EXPANSION
   • Pull 3–6 ingredient / cuisine / macro keywords from the Nutritionist
     insight (e.g. "high protein" → ["lentils", "paneer", "tofu"]).
   • Bias toward pantry items the user already owns.
   • NEVER include any allergen from the allergy list above.

2. TIME BUDGET
   • "quick" / "fast" / "breakfast" → 20–30 mins.
   • "lunch" → 30–45 mins.
   • "dinner" → 45–60 mins.
   • Use 45 as default when unclear.

3. MEAL TYPE
   • Detect from query or time-of-day cues.  Default: "dinner".

4. CONSTRAINT HINTS
   • List any medical / dietary constraint terms that should be forwarded
     to the penalty layer (e.g. "no sugar", "low glycemic", "no onion").

5. INDIAN MODE
   • Set indian_only = true if the query or nutritionist advice mentions
     Indian cuisine, regional Indian terms, or curry / dal / roti / dosa.

═══════════════════════════════════════════════════════════════
OUTPUT — respond ONLY with valid JSON, no markdown fences:
═══════════════════════════════════════════════════════════════
{{
    "likes":            ["keyword1", "keyword2"],
    "time_budget":      <int minutes>,
    "meal_type":        "breakfast" | "lunch" | "dinner",
    "constraint_hints": ["no sugar", ...],
    "indian_only":      true | false,
    "reasoning":        "<1-sentence explanation>"
}}
"""

    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": "Generate search parameters now."},
            ],
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        return json.loads(completion.choices[0].message.content)

    except Exception as e:
        print(f"❌ Query Agent Failed: {e}")
        # Safe fallback — pipeline continues without crashing
        return {
            "likes":            [user_query],
            "time_budget":      45,
            "meal_type":        "dinner",
            "constraint_hints": [],
            "indian_only":      False,
            "reasoning":        f"Fallback mode ({e})",
        }


# ---------------------------------------------------------------------------
# SMOKE TEST
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    result = query_agent(
        user_query="I want something light for dinner, I have a cold",
        user_profile={"diet": "Vegetarian", "bmi": 24.5, "allergies": ["Nuts"]},
        pantry_items=["tomato", "ginger", "garlic", "lemon", "honey", "turmeric"],
        nutrition_insight=(
            "Warm broths and vitamin C-rich foods support immune recovery "
            "and soothe throat inflammation."
        ),
    )
    print(json.dumps(result, indent=2))