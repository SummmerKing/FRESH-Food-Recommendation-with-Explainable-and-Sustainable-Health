"""
Build FAISS index from recipe vectors
Creates: faiss index file + recipe_ids_FINAL.npy
"""

import numpy as np
import faiss
import json
from tqdm import tqdm

def build_faiss_index(
    vectors_path='claude_recipe_vectors_FINAL.npy',
    metadata_path='/data1/home/sathvik/Documents/FRESH/Metadata/Claude_metadata_FINAL.json',
    index_path='claude_recipe_index.faiss',
    ids_path='claude_recipe_ids_FINAL.npy',
    use_gpu=False
):
    """
    Build FAISS index for fast similarity search
    """
    
    print(f"Loading vectors from {vectors_path}...")
    vectors = np.load(vectors_path,allow_pickle=True)
    
    print(f"Loading metadata from {metadata_path}...")
    with open(metadata_path, 'r', encoding='utf-8') as f:
        recipes = json.load(f)
    
    print(f"\nVectors shape: {vectors.shape}")
    print(f"Total recipes: {len(recipes):,}")
    
    # Verify alignment
    if len(vectors) != len(recipes):
        raise ValueError(f"Mismatch: {len(vectors)} vectors but {len(recipes)} recipes!")
    
    # Extract recipe IDs in order
    recipe_ids = np.array([recipe['recipe_id'] for recipe in recipes])
    
    # Normalize vectors for cosine similarity
    print("\nNormalizing vectors...")
    faiss.normalize_L2(vectors)
    
    # Get vector dimension
    dimension = vectors.shape[1]
    
    # Build FAISS index
    print(f"\nBuilding FAISS index (dimension={dimension})...")
    
    # Use IndexFlatIP for exact cosine similarity search
    # For large datasets, consider IndexIVFFlat for faster (approximate) search
    if len(vectors) > 100000:
        print("Large dataset detected. Using IVF index for faster search...")
        # Number of clusters (rule of thumb: sqrt of dataset size)
        nlist = min(int(np.sqrt(len(vectors))), 4096)
        quantizer = faiss.IndexFlatIP(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
        
        print(f"Training index with {nlist} clusters...")
        index.train(vectors)
        print("Adding vectors to index...")
        index.add(vectors)
        
        # Set search parameters
        index.nprobe = 10  # Number of clusters to search
    else:
        print("Using flat index for exact search...")
        index = faiss.IndexFlatIP(dimension)
        print("Adding vectors to index...")
        index.add(vectors)
    
    # Move to GPU if available and requested
    if use_gpu:
        try:
            print("\nAttempting to use GPU...")
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
            print("✓ GPU acceleration enabled")
        except Exception as e:
            print(f"GPU not available: {e}")
            print("Continuing with CPU...")
    
    print(f"\nIndex contains {index.ntotal:,} vectors")
    
    # Save index
    print(f"Saving FAISS index to {index_path}...")
    if use_gpu:
        # Convert back to CPU for saving
        index = faiss.index_gpu_to_cpu(index)
    faiss.write_index(index, index_path)
    
    # Save recipe IDs
    print(f"Saving recipe IDs to {ids_path}...")
    np.save(ids_path, recipe_ids)
    
    print("\n✓ FAISS index built successfully!")
    print(f"✓ Index file: {index_path}")
    print(f"✓ IDs file: {ids_path}")
    print(f"✓ Total indexed recipes: {index.ntotal:,}")
    
    # Test search
    print("\n--- Testing Search ---")
    test_vector = vectors[0:1]  # Use first recipe as test
    k = 5
    distances, indices = index.search(test_vector, k)
    
    print(f"Test query: Recipe '{recipes[0]['title']}'")
    print(f"Top {k} similar recipes:")
    for i, (dist, idx) in enumerate(zip(distances[0], indices[0]), 1):
        print(f"  {i}. {recipes[idx]['title']} (similarity: {dist:.3f})")
    
    return index, recipe_ids

if __name__ == "__main__":
    build_faiss_index()