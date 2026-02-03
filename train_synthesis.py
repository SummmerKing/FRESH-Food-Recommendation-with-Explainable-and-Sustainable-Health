import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from model import FRESH_Network # Ensure model.py has the 2-input changes!

def generate_synthetic_data(num_samples=5000):
    """
    Generates synthetic data compatible with the new 2-Input architecture.
    Inputs: [Pantry Score, Health Score]
    """
    print(f"Generating {num_samples} smart synthetic samples (2-Input Mode)...")
    
    # 1. Random Fake Recipe Vectors (384 dimensions)
    vectors = torch.randn(num_samples, 384)
    
    # 2. Random Features: [Pantry(0-1), Health(0-1)]
    # We removed Time because it's now a hard filter in main.py
    features = torch.rand(num_samples, 2)
    
    labels = []
    for i in range(num_samples):
        pantry_score = features[i][0]
        health_score = features[i][1] 
        
        # --- BASE LOGIC (Reflected from main.py weights) ---
        # 40% Taste (Implicit in vector training), 40% Pantry, 20% Health
        # For synthetic training, we assume "Taste" is neutral since vectors are random noise here.
        probability = (pantry_score * 0.6) + (health_score * 0.4)
        
        # --- COMPLEX "TEACHER" RULES ---
        
        # Rule 1: The "Useless Recipe" Penalty
        # If pantry match is near zero, it shouldn't be recommended even if healthy.
        if pantry_score < 0.1:
            probability -= 0.3
            
        # Rule 2: The "Perfect Match" Bonus
        # If you have the ingredients (>0.8) AND it fits your diet (>0.8), boost it.
        if pantry_score > 0.8 and health_score > 0.8:
            probability += 0.2

        # Add Noise & Clamp
        noise = np.random.normal(0, 0.05)
        probability += noise
        probability = max(0.0, min(1.0, probability))
        
        # Label (Regression target 0.0-1.0 is better for ranking than binary 0/1)
        # But if you use BCELoss, keep it as probability or binary.
        # Let's keep your binary logic for consistency with previous setup.
        label = 1.0 if probability > 0.6 else 0.0
        labels.append(label)
        
    return vectors, features, torch.tensor(labels).unsqueeze(1)

def train():
    # Initialize Model
    model = FRESH_Network() # This will use the new 2-input logic from model.py
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss() # Binary Cross Entropy
    
    # Get Data
    X_vec, X_feat, y = generate_synthetic_data()
    
    print("Starting Training...")
    model.train()
    for epoch in range(50): # 50 Epochs
        optimizer.zero_grad()
        
        # Forward pass
        predictions = model(X_vec, X_feat)
        loss = criterion(predictions, y)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
            
    print("Training Complete.")
    # Save as fresh_model.pth to match what main.py looks for
    torch.save(model.state_dict(), "fresh_model.pth") 
    print("✅ Model saved as 'fresh_model.pth'")

if __name__ == "__main__":
    train()