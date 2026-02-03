import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import json
import os
from model import FRESH_Network

# --- CONFIG ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 50

# --- 1. LOAD HELPERS ---
print("📂 Loading Metadata & Vectors...")
METADATA_PATH = "/data1/home/sathvik/Documents/FRESH/claude_recipes_metadata_FINAL.json"
VECTORS_PATH = "/data1/home/sathvik/Documents/FRESH/claude_recipe_vectors_FINAL.npy"
MACROS_PATH = "/data1/home/sathvik/Documents/FRESH/macro_vectors.pt"
CSV_PATH = "/data1/home/sathvik/Documents/FRESH/distilled_labels.csv"

# Load Metadata
with open(METADATA_PATH, "r") as f:
    meta_raw = json.load(f)
    if isinstance(meta_raw, list):
        # Normalize IDs to strings immediately
        metadata = {str(r.get("recipe_id") or r.get("id")): r for r in meta_raw}
        recipe_ids = [str(r.get("recipe_id") or r.get("id")) for r in meta_raw]
    else:
        metadata = meta_raw
        # Normalize IDs to strings
        recipe_ids = [str(k) for k in metadata.keys()]

# Load Vectors
print("   Loading vectors from .npy file...")
numpy_vectors = np.load(VECTORS_PATH)
vectors = torch.from_numpy(numpy_vectors).float().to(DEVICE)

# Map IDs to Index
id_to_idx = {rid: i for i, rid in enumerate(recipe_ids)}

# Load Macros
try:
    macro_vectors = torch.load(MACROS_PATH, map_location=DEVICE)
except FileNotFoundError:
    print("❌ ERROR: 'macro_vectors.pt' is missing. Run setup_data.py first!")
    exit()

def get_health_score(vec, bmi):
    """Calculates Health Score dynamically"""
    if isinstance(vec, np.ndarray): vec = torch.from_numpy(vec).to(DEVICE)
        
    prot = max(0.1, torch.dot(vec, macro_vectors["Protein"]).item())
    carb = max(0.1, torch.dot(vec, macro_vectors["Carbs"]).item())
    fat = max(0.1, torch.dot(vec, macro_vectors["Fats"]).item())
    total = prot + carb + fat
    
    p_ratio = prot / total
    c_ratio = carb / total
    f_ratio = fat / total
    
    score = 1.0
    if bmi >= 25.0: 
        if p_ratio > 0.4: score = 1.2
        elif (c_ratio + f_ratio) > 0.7: score = 0.6
    elif bmi <= 18.5:
        if (c_ratio + f_ratio) > 0.6: score = 1.2
    return score

# --- 2. PREPARE DATASET ---
print("📊 Processing Training Data...")
if not os.path.exists(CSV_PATH):
    print(f"❌ Error: Could not find {CSV_PATH}. Did you run the generator script?")
    exit()

df = pd.read_csv(CSV_PATH)

X_vec_list = []
X_feat_list = []
y_list = []

print(f"   Analyzing {len(df)} rows from CSV against {len(recipe_ids)} Metadata IDs...")

# DEBUG VARIABLES
mismatch_count = 0
first_failure = True

for _, row in df.iterrows():
    # [SMART FIX] clean the ID
    raw_id = str(row['recipe_id'])
    
    # Remove '.0' if pandas accidentally added it (e.g. "91198.0" -> "91198")
    if raw_id.endswith(".0"):
        rid = raw_id[:-2]
    else:
        rid = raw_id

    # Check match
    if rid not in id_to_idx: 
        mismatch_count += 1
        if first_failure:
            print(f"⚠️  DEBUG MISMATCH: CSV ID '{raw_id}' (cleaned: '{rid}') NOT FOUND in Metadata.")
            print(f"    -> Sample Metadata ID: '{recipe_ids[0]}'")
            first_failure = False
        continue 
    
    # Get Index
    idx = id_to_idx[rid]
    if idx >= len(vectors): continue 
    
    vec_tensor = vectors[idx]
    
    # Features
    pantry_score = min(row['user_pantry_count'] / 5.0, 1.0)
    health_score = get_health_score(vec_tensor, row['user_bmi'])
    
    time_mins = metadata[rid].get("time_minutes", 30)
    time_score = 1.0 if time_mins <= 45 else max(0, 1.0 - ((time_mins - 45)/60))
    
    X_vec_list.append(vec_tensor)
    X_feat_list.append([pantry_score, time_score, health_score])
    y_list.append(row['label_score'])

# CHECK RESULTS
if not X_vec_list:
    print("\n❌ CRITICAL FAILURE: 0 Matches found.")
    print(f"   Skipped {mismatch_count} rows due to ID mismatch.")
    print("   SOLUTION: Run 'python distill_training_data.py' again to regenerate the CSV with correct IDs.")
    exit()

X_vec = torch.stack(X_vec_list).to(DEVICE)
X_feat = torch.tensor(X_feat_list, dtype=torch.float32).to(DEVICE)
y = torch.tensor(y_list, dtype=torch.float32).unsqueeze(1).to(DEVICE)

print(f"✅ Success! Loaded {len(y)} training samples.")

# --- 3. TRAIN ---
model = FRESH_Network().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss() 

print("🚀 Starting Distillation Training...")
model.train()

for epoch in range(EPOCHS):
    optimizer.zero_grad()
    preds = model(X_vec, X_feat)
    loss = criterion(preds, y)
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

torch.save(model.state_dict(), "fresh_model.pth")
print("\n🎓 Graduation Day: 'fresh_model.pth' updated!")