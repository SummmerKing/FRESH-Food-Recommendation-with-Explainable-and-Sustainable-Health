import os
import json
import glob
import faiss
import numpy as np
from groq import Groq
from sentence_transformers import SentenceTransformer

# --- CONFIGURATION ---
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "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")

# Paths (Update folder path if needed)
DATA_DIR = "/data1/home/sathvik/Documents/FRESH/New Chunks" 
INDEX_PATH = os.path.join(DATA_DIR, "chapter_chunks_index.json")

# --- INITIALIZATION ---
print("⚙️  Initializing RAG Agent...")

embedder = SentenceTransformer('all-MiniLM-L6-v2')
client = Groq(api_key=GROQ_API_KEY)

# Load Knowledge Base
nutri_index = None
nutri_chunks = []

try:
    # 1. Load FAISS Index
    # Check if index exists and handle potential file extension issues
    if os.path.exists(INDEX_PATH) and not INDEX_PATH.endswith('.json'):
        nutri_index = faiss.read_index(INDEX_PATH)
    else:
        # Fallback: Look for a .index file if the path above was just a guess
        index_files = glob.glob(os.path.join(DATA_DIR, "*.index"))
        if index_files:
            nutri_index = faiss.read_index(index_files[0])
            print(f"   Loaded Index: {index_files[0]}")

    # 2. Load Text Chunks
    json_pattern = os.path.join(DATA_DIR, "*_chunks.json") 
    json_files = sorted(glob.glob(json_pattern))
    
    for jf in json_files:
        with open(jf, 'r') as f:
            data = json.load(f)
            if isinstance(data, list):
                nutri_chunks.extend(data)

    if nutri_index:
        print(f"✅ Knowledge Base Loaded: {nutri_index.ntotal} vectors, {len(nutri_chunks)} text chunks.")
    else:
        print("⚠️  FAISS Index not found. RAG will run in fallback mode (LLM only).")

except Exception as e:
    print(f"❌ Error loading RAG resources: {e}")

# --- CORE FUNCTIONS ---

def retrieve(query, k=5):
    """Searches the index and returns the corresponding text chunks."""
    if not nutri_index or not nutri_chunks:
        return []

    try:
        query_vec = embedder.encode([query]).astype('float32')
        faiss.normalize_L2(query_vec)
        distances, indices = nutri_index.search(query_vec, k)
        
        results = []
        for idx in indices[0]:
            if idx < 0 or idx >= len(nutri_chunks): continue
            
            chunk = nutri_chunks[idx]
            text = chunk.get('content', str(chunk)) if isinstance(chunk, dict) else str(chunk)
            results.append(text)
            
        return results
    except Exception as e:
        print(f"⚠️ Retrieval failed: {e}")
        return []

# --- 🛠️ FIX IS HERE: UPDATED FUNCTION SIGNATURE ---
def nutrition_analysis_agent(rag_context_snippets=None, user_query="", user_profile=None):
    """
    Analyzes the user's situation using RAG + LLM.
    Args:
        rag_context_snippets: Ignored (legacy arg), we do retrieval internally.
        user_query (str): The user's input string.
        user_profile (dict): User's BMI, Diet, etc.
    """
    # 1. Construct Search Query
    if not user_query and user_profile:
        conditions = []
        if user_profile.get('bmi', 22) > 25: conditions.append("obesity weight loss")
        if user_profile.get('diet') == 'Vegan': conditions.append("vegan nutrition")
        conditions.extend(user_profile.get('allergies', []))
        search_query = " ".join(conditions)
    else:
        search_query = user_query

    # 2. Internal Retrieval
    retrieved_texts = retrieve(search_query, k=3)
    
    if retrieved_texts:
        context_block = "\n\n".join([f"--- TEXTBOOK FACT ---\n{s}" for s in retrieved_texts])
    else:
        context_block = "No specific textbook context found. Use general nutritional knowledge."

    # 3. Prompt Engineering
    system_prompt = f"""
    You are an expert Nutritionist and AI Medical Assistant.
    
    USER PROFILE:
    - Query: "{user_query}"
    - BMI: {user_profile.get('bmi', 'Unknown') if user_profile else 'Unknown'}
    - Diet: {user_profile.get('diet', 'Any') if user_profile else 'Any'}
    
    TEXTBOOK KNOWLEDGE:
    {context_block}
    
    TASK:
    Analyze the situation and provide:
    1. A one-sentence medical insight.
    2. A list of 3-5 specific food ingredients or nutrients to search for.
    
    OUTPUT JSON FORMAT ONLY:
    {{
        "insight": "Your medical/nutritional reasoning here...",
        "recommended_keywords": ["ingredient1", "nutrient1", "food_item"]
    }}
    """

    # 4. Call LLM
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt}
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        return json.loads(completion.choices[0].message.content)
    
    except Exception as e:
        print(f"❌ Agent LLM Error: {e}")
        return {
            "insight": "Standard nutritional balance.",
            "recommended_keywords": [user_query] if user_query else ["healthy"]
        }

if __name__ == "__main__":
    # Test locally
    print(nutrition_analysis_agent(user_query="I have diabetes", user_profile={"bmi": 28}))