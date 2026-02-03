"""
Recipe Embedding Generator
===========================
Generate embeddings and FAISS index for preprocessed recipe data.

This script:
1. Loads cleaned recipe metadata
2. Creates text representations optimized for search
3. Generates embeddings using SentenceTransformer
4. Normalizes vectors for cosine similarity
5. Builds FAISS index
6. Saves all artifacts

Usage:
    python generate_embeddings.py

Output:
    - recipe_vectors_FINAL.npy
    - recipe_ids_FINAL.npy
    - recipe_index_FINAL.faiss
    - embedding_report.json
"""

import json
import numpy as np
from sentence_transformers import SentenceTransformer
import logging
from datetime import datetime
from tqdm import tqdm
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('embedding_generation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class RecipeEmbeddingGenerator:
    """Generate optimized embeddings for recipe recommendations."""
    
    def __init__(self, 
                 metadata_path: str,
                 output_dir: str,
                 model_name: str = "all-MiniLM-L6-v2",
                 batch_size: int = 256):
        self.metadata_path = metadata_path
        self.output_dir = output_dir
        self.model_name = model_name
        self.batch_size = batch_size
        
        self.model = None
        self.recipes = []
        self.stats = {}
        
    def load_model(self):
        """Load sentence transformer model."""
        logger.info(f"Loading model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded. Embedding dimension: {embedding_dim}")
        return embedding_dim
    
    def load_recipes(self):
        """Load recipe metadata."""
        logger.info(f"Loading recipes from: {self.metadata_path}")
        with open(self.metadata_path, 'r', encoding='utf-8') as f:
            self.recipes = json.load(f)
        logger.info(f"Loaded {len(self.recipes):,} recipes")
        return len(self.recipes)
    
    def create_text_representation(self, recipe: dict) -> str:
        """
        Create optimized text representation for embedding.
        
        Strategy:
        - Emphasize meal type (appears 2x)
        - Include title (descriptive)
        - Add diet tags (for filtering)
        - Include key ingredients (for pantry matching)
        - Add cuisine/cooking method keywords
        """
        parts = []
        
        # Meal type (high weight - appears twice)
        meal_type = recipe.get("meal_type", "")
        if meal_type:
            parts.extend([meal_type, meal_type])
        
        # Title (very descriptive)
        title = recipe.get("title", "")
        if title:
            parts.append(title)
        
        # Diet tags (for vegetarian/vegan matching)
        diet = recipe.get("diet", [])
        if isinstance(diet, list):
            parts.extend(diet)
        elif diet:
            parts.append(str(diet))
        
        # Key ingredients (top 10 for pantry matching)
        ingredients = recipe.get("cleaned_ingredients", [])
        if ingredients:
            # Prioritize common ingredients
            parts.extend(ingredients[:10])
        
        # Search keywords (cuisine, method)
        keywords = recipe.get("search_keywords", [])
        if keywords:
            # Add top keywords
            relevant_keywords = [kw for kw in keywords if len(kw) > 3][:5]
            parts.extend(relevant_keywords)
        
        # Create final text
        text = " ".join(str(p) for p in parts if p)
        return text.strip()
    
    def generate_embeddings_batch(self, texts: list) -> np.ndarray:
        """Generate embeddings in batches for efficiency."""
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False  # We'll normalize later
        )
        return embeddings.astype(np.float32)
    
    def normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """Normalize vectors for cosine similarity."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)  # Avoid division by zero
        normalized = vectors / norms
        return normalized.astype(np.float32)
    
    def build_faiss_index(self, vectors: np.ndarray):
        """Build FAISS index for fast similarity search."""
        try:
            import faiss
            
            logger.info("Building FAISS index...")
            dimension = vectors.shape[1]
            
            # Use IndexFlatIP (Inner Product) for cosine similarity with normalized vectors
            index = faiss.IndexFlatIP(dimension)
            index.add(vectors)
            
            logger.info(f"FAISS index built with {index.ntotal:,} vectors")
            return index
            
        except ImportError:
            logger.warning("FAISS not available. Skipping index creation.")
            logger.warning("Install with: pip install faiss-cpu  or  pip install faiss-gpu")
            return None
    
    def generate_all_embeddings(self):
        """Main embedding generation pipeline."""
        logger.info("=" * 70)
        logger.info("Starting Embedding Generation Pipeline")
        logger.info("=" * 70)
        
        start_time = datetime.now()
        
        # Step 1: Load model
        embedding_dim = self.load_model()
        self.stats["embedding_dimension"] = embedding_dim
        
        # Step 2: Load recipes
        num_recipes = self.load_recipes()
        self.stats["total_recipes"] = num_recipes
        
        # Step 3: Create text representations
        logger.info("Creating text representations...")
        texts = []
        recipe_ids = []
        
        for i, recipe in enumerate(tqdm(self.recipes, desc="Processing recipes")):
            text = self.create_text_representation(recipe)
            texts.append(text)
            recipe_ids.append(recipe.get("recipe_id"))
            
            # Log sample for verification
            if i < 3:
                logger.info(f"\nSample {i+1}:")
                logger.info(f"  Recipe: {recipe.get('title', 'Unknown')}")
                logger.info(f"  Text: {text[:200]}...")
        
        self.stats["text_representations_created"] = len(texts)
        
        # Step 4: Generate embeddings in batches
        logger.info(f"\nGenerating embeddings (batch_size={self.batch_size})...")
        all_embeddings = []
        
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Embedding batches"):
            batch_texts = texts[i:i + self.batch_size]
            batch_embeddings = self.generate_embeddings_batch(batch_texts)
            all_embeddings.append(batch_embeddings)
        
        embeddings = np.vstack(all_embeddings)
        logger.info(f"Generated embeddings shape: {embeddings.shape}")
        self.stats["embeddings_generated"] = embeddings.shape[0]
        
        # Step 5: Normalize vectors
        logger.info("Normalizing vectors...")
        normalized_embeddings = self.normalize_vectors(embeddings)
        
        # Verify normalization
        sample_norms = np.linalg.norm(normalized_embeddings[:5], axis=1)
        logger.info(f"Sample vector norms (should be ~1.0): {sample_norms}")
        
        # Step 6: Build FAISS index
        faiss_index = self.build_faiss_index(normalized_embeddings)
        
        # Step 7: Save all artifacts
        logger.info("\nSaving artifacts...")
        
        # Save recipe IDs
        ids_path = os.path.join(self.output_dir, "recipe_ids_FINAL.npy")
        np.save(ids_path, np.array(recipe_ids, dtype=object))
        logger.info(f"  ✓ Saved recipe IDs: {ids_path}")
        self.stats["ids_path"] = ids_path
        
        # Save embeddings (normalized)
        vectors_path = os.path.join(self.output_dir, "recipe_vectors_FINAL.npy")
        np.save(vectors_path, normalized_embeddings)
        logger.info(f"  ✓ Saved vectors: {vectors_path}")
        self.stats["vectors_path"] = vectors_path
        
        # Save FAISS index
        if faiss_index:
            import faiss
            index_path = os.path.join(self.output_dir, "recipe_index_FINAL.faiss")
            faiss.write_index(faiss_index, index_path)
            logger.info(f"  ✓ Saved FAISS index: {index_path}")
            self.stats["faiss_index_path"] = index_path
        
        # Step 8: Generate report
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        self.stats["generation_time_seconds"] = duration
        self.stats["recipes_per_second"] = num_recipes / duration
        self.stats["model_name"] = self.model_name
        self.stats["batch_size"] = self.batch_size
        self.stats["timestamp"] = end_time.isoformat()
        
        report_path = os.path.join(self.output_dir, "embedding_report.json")
        with open(report_path, 'w') as f:
            json.dump(self.stats, f, indent=2)
        logger.info(f"  ✓ Saved report: {report_path}")
        
        # Summary
        logger.info("\n" + "=" * 70)
        logger.info("Embedding Generation Complete!")
        logger.info("=" * 70)
        logger.info(f"Total recipes: {num_recipes:,}")
        logger.info(f"Embedding dimension: {embedding_dim}")
        logger.info(f"Generation time: {duration:.1f}s ({num_recipes/duration:.1f} recipes/sec)")
        logger.info(f"Files created:")
        logger.info(f"  - {ids_path}")
        logger.info(f"  - {vectors_path}")
        if faiss_index:
            logger.info(f"  - {index_path}")
        logger.info(f"  - {report_path}")
        
        return self.stats
    
    def verify_embeddings(self):
        """Verify generated embeddings with sample queries."""
        logger.info("\n" + "=" * 70)
        logger.info("Verifying Embeddings with Sample Queries")
        logger.info("=" * 70)
        
        # Load generated artifacts
        ids_path = os.path.join(self.output_dir, "recipe_ids_FINAL.npy")
        vectors_path = os.path.join(self.output_dir, "recipe_vectors_FINAL.npy")
        
        recipe_ids = np.load(ids_path, allow_pickle=True)
        recipe_vectors = np.load(vectors_path)
        
        logger.info(f"Loaded {len(recipe_ids):,} recipe IDs")
        logger.info(f"Loaded vectors with shape: {recipe_vectors.shape}")
        
        # Create recipe lookup
        recipe_map = {r["recipe_id"]: r for r in self.recipes}
        
        # Sample queries
        test_queries = [
            "indian breakfast dosa",
            "quick dinner pasta",
            "vegetarian paneer curry",
            "chocolate dessert cake",
            "healthy lunch salad"
        ]
        
        logger.info("\nTesting sample queries:")
        for query in test_queries:
            logger.info(f"\n📝 Query: '{query}'")
            
            # Generate query embedding
            query_vec = self.model.encode(query, convert_to_numpy=True)
            query_vec = query_vec / np.linalg.norm(query_vec)
            
            # Compute similarities
            similarities = recipe_vectors.dot(query_vec)
            top_5_idx = np.argsort(-similarities)[:5]
            
            logger.info("  Top 5 matches:")
            for rank, idx in enumerate(top_5_idx, 1):
                recipe_id = recipe_ids[idx]
                score = similarities[idx]
                recipe = recipe_map.get(recipe_id, {})
                title = recipe.get("title", "Unknown")
                meal_type = recipe.get("meal_type", "Unknown")
                logger.info(f"    {rank}. {title[:60]} ({meal_type}) - Score: {score:.3f}")
        
        logger.info("\n✓ Verification complete!")


def main():
    """Main entry point."""
    print("""
╔═══════════════════════════════════════════════════════════════════╗
║              Recipe Embedding Generator v1.0                      ║
║                                                                   ║
║  This script generates optimized embeddings for your recipe       ║
║  recommendation system.                                           ║
║                                                                   ║
║  Requirements:                                                    ║
║    - sentence-transformers                                        ║
║    - faiss-cpu or faiss-gpu (optional but recommended)           ║
║    - numpy                                                        ║
║    - tqdm                                                         ║
║                                                                   ║
║  Run this AFTER preprocessing your recipe data!                  ║
╚═══════════════════════════════════════════════════════════════════╝
    """)
    
    # Configuration
    METADATA_PATH = "/data1/home/sathvik/Documents/FRESH/recipes_metadata_FINAL.json"
    OUTPUT_DIR = "/data1/home/sathvik/Documents/FRESH"
    MODEL_NAME = "all-MiniLM-L6-v2"  # Fast and good quality
    BATCH_SIZE = 256  # Adjust based on your GPU/CPU
    
    print(f"Configuration:")
    print(f"  Metadata: {METADATA_PATH}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Batch size: {BATCH_SIZE}")
    print()
    
    # Check if input file exists
    if not os.path.exists(METADATA_PATH):
        print(f"❌ Error: Metadata file not found at {METADATA_PATH}")
        print(f"   Please run preprocessing first!")
        return
    
    # Create output directory if needed
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Generate embeddings
    generator = RecipeEmbeddingGenerator(
        metadata_path=METADATA_PATH,
        output_dir=OUTPUT_DIR,
        model_name=MODEL_NAME,
        batch_size=BATCH_SIZE
    )
    
    stats = generator.generate_all_embeddings()
    
    # Verify with sample queries
    print("\n" + "="*70)
    response = input("Would you like to verify embeddings with sample queries? (y/n): ")
    if response.lower() == 'y':
        generator.verify_embeddings()
    
    print("\n✅ All done!")
    print("\nNext steps:")
    print("1. Update your recommender.py with new file paths:")
    print(f"   METADATA_PATH = '{METADATA_PATH}'")
    print(f"   VECS_PATH = '{OUTPUT_DIR}/recipe_vectors_FINAL.npy'")
    print(f"   IDS_PATH = '{OUTPUT_DIR}/recipe_ids_FINAL.npy'")
    print(f"   INDEX_PATH = '{OUTPUT_DIR}/recipe_index_FINAL.faiss'")
    print("2. Restart your recommendation server")
    print("3. Test recommendations with your favorite queries!")


if __name__ == "__main__":
    main()