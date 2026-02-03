import os
import json
import faiss
import numpy as np
from groq import Groq
from sentence_transformers import SentenceTransformer

# --- SETUP ---
embedder = SentenceTransformer('all-MiniLM-L6-v2')
# Ideally use os.environ.get("GROQ_API_KEY") for security, but hardcoding works for testing
client = Groq(api_key=os.environ.get("GROQ_API_KEY", "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"))

# PATHS (Ensure these match your actual file locations)
NUTRITION_INDEX_PATH = "/data1/home/sathvik/Documents/FRESH/target_chapters.index"
NUTRITION_META_PATH = "/data1/home/sathvik/Documents/FRESH/target_chapters_chunks.json"

# LOAD RESOURCES GLOBALLY
print("Loading Nutrition RAG Memory...")
try:
    nutri_index = faiss.read_index(NUTRITION_INDEX_PATH)
    with open(NUTRITION_META_PATH, 'r') as f:
        nutri_chunks = json.load(f)
    print("✅ Nutrition Knowledge Base Loaded.")
except Exception as e:
    print(f"⚠️ Nutrition RAG missing: {e}. Falling back to LLM knowledge.")
    nutri_index = None
    nutri_chunks = []

def retrieve_medical_facts(query, k=5):
    """
    Retrieves facts from the FAISS index with safety checks.
    """
    # BUG FIX 1: Use the correct global variable name 'nutri_index'
    if nutri_index is None:
        return []

    try:
        # 1. Encode Query
        vec = embedder.encode([query]).astype('float32')
        faiss.normalize_L2(vec)
        
        # 2. Search Index (BUG FIX 1: medical_index -> nutri_index)
        D, I = nutri_index.search(vec, k)
        
        retrieved = []
        print(f"\n🔍 DEBUG: Retrieved for '{query}':") # DEBUG PRINT
        
        for i, idx in enumerate(I[0]):
            if idx < 0: continue
            
            # BUG FIX 3: Robust lookup
            # In Script 1, you appended to a list 'final_chunks'. 
            # FAISS ID 0 = final_chunks[0], ID 1 = final_chunks[1].
            try:
                item = nutri_chunks[idx]
                
                # Handle if item is dict (from Script 1) or string (legacy)
                content = item.get('content', str(item)) if isinstance(item, dict) else str(item)
                source = item.get('source', 'Unknown') if isinstance(item, dict) else 'Unknown'
                
                print(f"   {i+1}. [{source}] {content[:100]}...") # Verify relevance here
                retrieved.append(content)
                
            except IndexError:
                print(f"   ⚠️ Index {idx} out of bounds in JSON chunks.")
                continue

        return retrieved

    except Exception as e:
        print(f"RAG Error: {e}")
        return []

def nutrition_analysis_agent(rag_context_snippets, user_query=None, user_profile=None):
    """
    RAG Scientist: 
    1. Search PDFs for medical facts based on User Query/Profile.
    2. Recommend ingredients/nutrients to look for.
    """
    
    # --- STEP 1: DEFINE THE MEDICAL SEARCH QUERY ---
    if user_query:
        medical_search_query = user_query 
    else:
        # Proactive health check
        conditions = []
        if user_profile.get('bmi', 22) > 25: conditions.append("obesity weight loss low calorie")
        if user_profile.get('diet') == 'Vegan': conditions.append("vegan nutrition deficiency")
        conditions.extend(user_profile.get('allergies', []))
        medical_search_query = " ".join(conditions) if conditions else "general healthy nutrition"

    # --- STEP 2: RETRIEVE FACTS (The RAG Step) ---
    retrieved_facts = retrieve_medical_facts(medical_search_query)
    facts_text = "\n".join(retrieved_facts) if retrieved_facts else "No specific medical documents found. Use general nutritional knowledge."

    # --- STEP 3: THE PROMPT (UPDATED) ---
    # We changed this to ask for "Ingredients to Search For" instead of "Analyzing Recipes"
    system_prompt = f"""
    You are the Nutrition RAG Agent. You provide medical nutritional advice based on the facts provided.
    
    USER SITUATION:
    "{user_query}"
    
    RELEVANT MEDICAL FACTS (From Textbooks):
    {facts_text}
    
    TASK:
    Based on the medical facts, identify specific ingredients or nutrients the user should eat to help their condition.
    
    OUTPUT JSON ONLY:
    {{
        "insight": "One sentence explaining WHY, citing the medical fact (e.g. 'Zinc reduces cold duration according to...').",
        "recommended_keywords": ["ingredient1", "ingredient2", "nutrient1"] 
    }}
    """
    
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": system_prompt}],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        return json.loads(completion.choices[0].message.content)
    except Exception as e:
        print(f"Agent Error: {e}")
        return {"insight": "Standard healthy choice.", "recommended_keywords": []}