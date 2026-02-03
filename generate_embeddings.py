"""
Generate embeddings from recipes_metadata_FINAL.json
Creates recipe_vectors_FINAL.npy
"""

import json
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch

def generate_recipe_embeddings(
    metadata_path='/data1/home/sathvik/Documents/FRESH/Metadata/Claude_metadata_FINAL.json',
    output_path='claude_recipe_vectors_FINAL.npy',
    model_name='sentence-transformers/all-MiniLM-L6-v2',
    batch_size=32
):
    """
    Generate embeddings for all recipes
    """
    
    print(f"Loading metadata from {metadata_path}...")
    with open(metadata_path, 'r', encoding='utf-8') as f:
        recipes = json.load(f)
    
    print(f"Total recipes: {len(recipes):,}")
    
    # Load model
    print(f"\nLoading embedding model: {model_name}...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model = SentenceTransformer(model_name)
    model.to(device)
    
    # Create text representations for embedding
    print("\nCreating text representations...")
    texts = []
    for recipe in tqdm(recipes, desc="Processing recipes"):
        # Combine title and ingredients for rich representation
        title = recipe.get('title', '')
        ingredients = recipe.get('cleaned_ingredients', [])
        
        # Format: "Title: [title]. Ingredients: [ing1, ing2, ...]"
        text = f"{title}. " + " ".join(ingredients)
        texts.append(text)
    
    # Generate embeddings in batches
    print(f"\nGenerating embeddings (batch_size={batch_size})...")
    all_embeddings = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding batches"):
        batch = texts[i:i+batch_size]
        embeddings = model.encode(
            batch,
            convert_to_numpy=True,
            show_progress_bar=False,
            device=device
        )
        all_embeddings.append(embeddings)
    
    # Combine all embeddings
    embeddings_array = np.vstack(all_embeddings)
    
    print(f"\nEmbeddings shape: {embeddings_array.shape}")
    print(f"Embedding dimension: {embeddings_array.shape[1]}")
    
    # Save embeddings
    print(f"Saving embeddings to {output_path}...")
    np.save(output_path, embeddings_array)
    
    print(f"✓ Embeddings saved: {output_path}")
    print(f"✓ Total vectors: {len(embeddings_array):,}")
    print(f"✓ Vector dimension: {embeddings_array.shape[1]}")
    
    return embeddings_array

if __name__ == "__main__":
    generate_recipe_embeddings()