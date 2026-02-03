import numpy as np
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
from sklearn.model_selection import train_test_split
from collections import Counter

# --- CONFIGURATION ---
DATA_DIR = r"/data1/home/sathvik/Documents/FRESH/FRESH/data"
VECS_PATH = os.path.join(DATA_DIR, "/data1/home/sathvik/Documents/FRESH/claude_recipe_vectors_FINAL.npy") 
METADATA_PATH = os.path.join(DATA_DIR, "/data1/home/sathvik/Documents/FRESH/claude_recipes_metadata_FINAL.json")
MODEL_SAVE_PATH = "meal_classifier.pth"

# --- MODEL DEFINITION ---
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
            nn.Sigmoid() # Multi-label classification (a recipe can be Breakfast AND Snack)
        )

    def forward(self, x):
        return self.network(x)

def train_classifier():
    print("🚀 Loading Data for Meal Training...")
    
    # 1. Load Vectors
    vectors = np.load(VECS_PATH).astype(np.float32)
    
    # 2. Load Metadata & Extract Labels
    with open(METADATA_PATH, 'r') as f:
        metadata = json.load(f)
    
    # Classes: 0: Breakfast, 1: Lunch, 2: Dinner, 3: Snack
    labels = []
    valid_indices = []
    
    print("🏷️  Extracting Ground Truth Labels...")
    for i, meta in enumerate(metadata):
        # Normalize tags
        tags = str(meta.get('meal_type', [])).lower() + " " + str(meta.get('title', '')).lower()
        
        # Ground Truth Logic (Heuristics to build training set)
        is_breakfast = 1 if any(x in tags for x in ['breakfast', 'morning', 'pancake', 'oat', 'cereal', 'paratha', 'dosa', 'idli', 'scramble']) else 0
        is_lunch = 1 if any(x in tags for x in ['lunch', 'sandwich', 'burger', 'wrap', 'salad', 'noon']) else 0
        is_dinner = 1 if any(x in tags for x in ['dinner', 'supper', 'roast', 'steak', 'main course']) else 0
        is_snack = 1 if any(x in tags for x in ['snack', 'appetizer', 'bite', 'dip', 'cookie']) else 0
        
        # Only train on recipes that actually have a label
        if is_breakfast or is_lunch or is_dinner or is_snack:
            labels.append([is_breakfast, is_lunch, is_dinner, is_snack])
            valid_indices.append(i)
            
    if not valid_indices:
        print("❌ Error: No labeled data found in metadata. Check your tags.")
        return

    # Filter vectors to only those with labels
    X = vectors[valid_indices]
    y = np.array(labels, dtype=np.float32)
    
    print(f"✅ Training on {len(X)} labeled recipes.")
    print(f"   Breakfast: {y[:,0].sum()} | Lunch: {y[:,1].sum()} | Dinner: {y[:,2].sum()} | Snack: {y[:,3].sum()}")

    # 3. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. Prepare Tensors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = TensorDataset(torch.tensor(X_train).to(device), torch.tensor(y_train).to(device))
    test_dataset = TensorDataset(torch.tensor(X_test).to(device), torch.tensor(y_test).to(device))
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # 5. Initialize Model
    model = MealClassifier().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss() # Binary Cross Entropy for Multi-label
    
    # 6. Training Loop
    print("\n🧠 Training Neural Probe...")
    epochs = 15
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"   Epoch {epoch+1}/{epochs} - Loss: {total_loss / len(train_loader):.4f}")

    # 7. Evaluation
    model.eval()
    with torch.no_grad():
        test_out = model(torch.tensor(X_test).to(device))
        # Simple accuracy check (Round to nearest 0 or 1)
        predicted = test_out.round()
        acc = (predicted == torch.tensor(y_test).to(device)).float().mean()
        print(f"\n🎯 Test Set Accuracy: {acc:.2%}")

    # 8. Save
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"✅ Classifier Saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_classifier()