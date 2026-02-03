"""
Fresh_metrics.py
================
Evaluation metrics and heuristic judge for the FRESH recommendation system.

Contains:
- Heuristic judge with enhanced regional/dietary/pantry matching
- Metric calculation utilities (NDCG, Precision, MRR, MAP)
- Live evaluation wrapper for Streamlit dashboard
- Batch evaluation for research experiments
"""

import numpy as np
from typing import List, Dict, Tuple
import requests
from sklearn.metrics import ndcg_score

# =====================================================================
# METRIC CALCULATION UTILITIES
# =====================================================================

def calculate_ndcg(relevance_scores: List[float], k: int = None) -> float:
    """
    Calculates Normalized Discounted Cumulative Gain.
    
    Args:
        relevance_scores: List of binary relevance (0.0 or 1.0)
        k: Cutoff rank (None = use all items)
    
    Returns:
        NDCG score between 0.0 and 1.0
    """
    if not relevance_scores: 
        return 0.0
    
    # Ground Truth: Binary relevance
    true_relevance = np.asarray([relevance_scores])
    
    # Predicted Score: Position in list (rank-based scoring)
    n_items = len(relevance_scores)
    predicted_scores = np.asarray([list(range(n_items, 0, -1))])
    
    current_k = k if k is not None and k < n_items else n_items
    if current_k < 1: 
        return 0.0
    
    return float(ndcg_score(true_relevance, predicted_scores, k=current_k))


def calculate_precision(relevant_items: List[bool], k: int = None) -> float:
    """
    Calculates Precision@K.
    
    Args:
        relevant_items: List of boolean relevance judgments
        k: Cutoff rank (None = use all items)
    
    Returns:
        Precision score between 0.0 and 1.0
    """
    if not relevant_items:
        return 0.0
    if k is not None:
        relevant_items = relevant_items[:k]
    return sum(relevant_items) / len(relevant_items)


def calculate_mrr(relevant_items: List[bool]) -> float:
    """
    Calculates Mean Reciprocal Rank.
    
    Args:
        relevant_items: List of boolean relevance judgments
    
    Returns:
        MRR score (reciprocal of first relevant item's rank)
    """
    for i, is_relevant in enumerate(relevant_items):
        if is_relevant:
            return 1.0 / (i + 1)
    return 0.0


def calculate_ap(relevant_items: List[bool]) -> float:
    """
    Calculates Average Precision for a single ranking.
    
    Args:
        relevant_items: List of boolean relevance judgments
    
    Returns:
        AP score between 0.0 and 1.0
    """
    if not relevant_items or sum(relevant_items) == 0:
        return 0.0
    
    hits = 0
    sum_precisions = 0.0
    for i, is_rel in enumerate(relevant_items):
        if is_rel:
            hits += 1
            sum_precisions += hits / (i + 1.0)
            
    return sum_precisions / hits


# =====================================================================
# ⚖️ HEURISTIC JUDGE (Enhanced Version)
# =====================================================================

def heuristic_judge(recipe: Dict, user_profile: Dict) -> Tuple[bool, float, str]:
    """
    Enhanced heuristic judge with improved Auto Mode support and regional matching.
    
    Args:
        recipe: Recipe dictionary with 'title' and 'ingredients'
        user_profile: User context with 'diet', 'query_intent', 'pantry', etc.
    
    Returns:
        Tuple of (is_relevant: bool, score: float, reason: str)
    """
    score = 0.0
    reason = []

    title = str(recipe.get('title', '')).lower()
    ings = " ".join(recipe.get('ingredients', [])).lower()
    full_text = title + " " + ings

    # ===================================================================
    # 1. DIET SAFETY (Hard Constraints)
    # ===================================================================
    diet = user_profile.get('diet', '')
    
    if diet in ['Vegetarian', 'Veg']:
        forbidden = {'chicken', 'beef', 'pork', 'fish', 'mutton', 'prawn', 
                    'egg', 'bacon', 'ham', 'meat', 'sausage'}
        if any(x in full_text for x in forbidden):
            return False, 0.0, "✗ Diet Violation (Vegetarian)"
    
    if diet == 'Vegan':
        forbidden_vegan = {'chicken', 'beef', 'pork', 'fish', 'mutton', 'prawn',
                          'egg', 'bacon', 'ham', 'meat', 'milk', 'cheese', 'butter',
                          'cream', 'yogurt', 'curd', 'ghee', 'paneer', 'honey'}
        if any(x in full_text for x in forbidden_vegan):
            return False, 0.0, "✗ Diet Violation (Vegan)"
    
    # ===================================================================
    # 2. PANTRY MATCHING (Enhanced Scoring)
    # ===================================================================
    pantry = user_profile.get('pantry', [])
    staples = {"salt", "sugar", "oil", "water", "ice", "pepper", "turmeric", 
              "chili", "cumin", "mustard", "ginger", "garlic"}
    
    hero_matches = 0
    for p in pantry:
        p_clean = p.lower().strip()
        
        # Handle plurals: "potatoes" -> "potato"
        if p_clean.endswith('es'): 
            p_root = p_clean[:-2]
        elif p_clean.endswith('s'): 
            p_root = p_clean[:-1]
        else: 
            p_root = p_clean
        
        # Check if root word exists in recipe (exclude staples)
        if p_root in full_text and p_clean not in staples:
            hero_matches += 1

    # Scoring based on pantry matches
    if hero_matches >= 4:
        score += 10.0
        reason.append(f"✓ Excellent Pantry Match ({hero_matches} items)")
    elif hero_matches >= 2:
        score += 7.0
        reason.append(f"✓ Good Pantry Match ({hero_matches} items)")
    elif hero_matches >= 1:
        score += 4.0
        reason.append("✓ Uses Available Ingredients")

    # ===================================================================
    # 3. QUERY INTENT (Regional, Health, Dietary)
    # ===================================================================
    query = user_profile.get('query_intent', '').lower()

    # --- REGIONAL CUISINE ---
    if 'south' in query and 'indian' in query:
        south_keywords = ['dosa', 'idli', 'sambar', 'vada', 'coconut', 'curry leaf', 
                         'kerala', 'chettinad', 'upma', 'parotta', 'pongal', 'rasam', 
                         'appam', 'uttapam', 'podi']
        
        if any(t in full_text for t in south_keywords):
            score += 8.0
            reason.append("✓ South Indian Match")
        elif any(t in title for t in ['malaysian', 'thai', 'east african', 'chinese', 'mexican']):
            score -= 5.0
            reason.append("✗ Wrong Cuisine (Not South Indian)")
    
    elif 'north' in query and 'indian' in query:
        north_keywords = ['roti', 'naan', 'paneer', 'dal', 'butter', 'masala', 
                         'paratha', 'chana', 'saag', 'korma', 'makhani', 'tandoori', 
                         'kulcha', 'rajma', 'chole']
        
        if any(t in full_text for t in north_keywords):
            score += 8.0
            reason.append("✓ North Indian Match")
        elif any(t in title for t in ['malaysian', 'thai', 'east african', 'chinese', 'mexican']):
            score -= 5.0
            reason.append("✗ Wrong Cuisine (Not North Indian)")

    # --- HEALTH CONTEXT ---
    if any(keyword in query for keyword in ['cold', 'flu', 'sick', 'comfort']):
        comfort_foods = ['soup', 'stew', 'broth', 'khichdi', 'tea', 'porridge', 
                        'ginger', 'turmeric', 'rasam', 'warm', 'hot']
        if any(t in full_text for t in comfort_foods):
            score += 8.0
            reason.append("✓ Comfort Food Match")
    
    if any(keyword in query for keyword in ['weight loss', 'low calorie', 'diet', 'light']):
        if any(t in full_text for t in ['salad', 'grilled', 'steamed', 'boiled', 'soup']):
            score += 6.0
            reason.append("✓ Low Calorie Match")
    
    if any(keyword in query for keyword in ['protein', 'muscle', 'gym', 'workout']):
        protein_foods = ['chicken', 'egg', 'dal', 'lentil', 'chickpea', 'quinoa', 
                        'tofu', 'paneer', 'sprout']
        if any(t in full_text for t in protein_foods):
            score += 7.0
            reason.append("✓ High Protein Match")
    if any(keyword in query for keyword in ['cold', 'flu', 'sick', 'comfort']):
        # ... (keep existing code) ...
        pass

    # 🆕 UPDATED DIABETIC BLOCK
    if 'diabetic' in query or 'diabetes' in query or 'sugar' in query:
        # Diabetic-friendly isn't just "Diabetic". It's low-carb, high-fiber, and protein.
        diabetic_safe_words = [
            'low carb', 'keto', 'sugar free', 'no sugar', 'high fiber', 
            'whole wheat', 'ragi', 'millet', 'oats', 'salad', 'grilled', 
            'roasted', 'steamed', 'soup', 'lentil', 'dal', 'sprouts',
            'bitter gourd', 'karela', 'methi', 'fenugreek', 'barley'
        ]
        
        # Check if title/ingredients match valid diabetic concepts
        if any(t in full_text for t in diabetic_safe_words):
            score += 8.0
            reason.append("✓ Diabetic Friendly Match")
        
        # Heavy Penalty for sugary/refined carb words
        sugary_words = ['sweet', 'sugar', 'honey', 'syrup', 'cake', 'dessert', 'chocolate', 'jam', 'jelly']
        if any(bad in full_text for bad in sugary_words):
            return False, 0.0, "✗ Violated Diabetic Constraint (Sugary)"
        
    # --- DIETARY RESTRICTIONS ---
    if any(keyword in query for keyword in ['no onion', 'no garlic', 'jain']):
        if 'onion' not in full_text and 'garlic' not in full_text:
            score += 10.0
            reason.append("✓ Jain Compliant")
        else:
            return False, 0.0, "✗ Violated Dietary Restriction (Onion/Garlic)"
    
    if 'gluten free' in query or 'celiac' in query:
        gluten_items = ['wheat', 'atta', 'maida', 'bread', 'pasta', 'noodle']
        if not any(g in full_text for g in gluten_items):
            score += 7.0
            reason.append("✓ Gluten Free")
    
    # --- LIFESTYLE ---
    if 'quick' in query or '15 min' in query or 'fast' in query:
        if any(t in full_text for t in ['instant', 'quick', 'easy', 'simple']):
            score += 5.0
            reason.append("✓ Quick Recipe")

    # ===================================================================
    # 4. FINAL VERDICT
    # ===================================================================
    
    # AUTO MODE (no query provided)
    if not query:
        if hero_matches >= 2:
            reason_text = ", ".join(reason) if reason else "Good Pantry Utilization"
            return True, max(score, 6.0), reason_text
        elif hero_matches >= 1:
            reason_text = ", ".join(reason) if reason else "Uses Available Ingredients"
            return True, max(score, 5.0), reason_text
        else:
            # Fallback: Accept basic Indian recipes even without pantry match
            if any(t in full_text for t in ['dal', 'rice', 'roti', 'curry', 'masala', 'paneer']):
                return True, 4.0, "Basic Indian Recipe"
            return False, 0.0, "No Pantry Match"
    
    # SMART MODE (query-driven evaluation)
    final_score = min(score, 10.0)
    is_relevant = final_score >= 4.0  # Lowered threshold for better recall
    
    # Reject if score went negative (wrong cuisine penalty)
    if final_score < 0:
        return False, 0.0, ", ".join(reason) if reason else "Negative Score"
    
    return is_relevant, final_score, ", ".join(reason) if reason else "Marginal Match"


def llm_judge_relevance(recipe: Dict, user_profile: Dict) -> Tuple[bool, float, str]:
    """
    Wrapper function for compatibility with existing code.
    Calls the enhanced heuristic_judge.
    """
    return heuristic_judge(recipe, user_profile)


# =====================================================================
# LIVE EVALUATION WRAPPER (For Streamlit Dashboard)
# =====================================================================

def calculate_live_metrics(recommendations: Dict, user_profile: Dict, k: int = 3) -> Dict:
    """
    Computes metrics for a single user session.
    Used by the Streamlit dashboard for real-time evaluation.
    
    Args:
        recommendations: Dict with 'breakfast', 'lunch', 'dinner' meal lists
        user_profile: User context dictionary
        k: Number of top items to evaluate per meal
    
    Returns:
        Dict with 'overall' and 'per_meal' metrics
    """
    all_rels = []
    per_meal_metrics = {}
    
    total_recipes = 0
    total_relevant = 0
    safety_violations = 0

    for meal in ['breakfast', 'lunch', 'dinner']:
        meal_recs = recommendations.get(meal, [])
        if not meal_recs:
            per_meal_metrics[meal] = {
                'ndcg': 0.0, 
                'precision': 0.0, 
                'mrr': 0.0, 
                'num_relevant': 0, 
                'num_recipes': 0
            }
            continue

        meal_rels = []
        for r in meal_recs[:k]:  # Only evaluate top K
            is_rel, score, reason = llm_judge_relevance(r, user_profile)
            meal_rels.append(1.0 if is_rel else 0.0)
            
            if "Violation" in reason:
                safety_violations += 1
        
        # Calculate per-meal stats
        num_rel = sum(meal_rels)
        total_relevant += num_rel
        total_recipes += len(meal_rels)
        
        per_meal_metrics[meal] = {
            'ndcg': calculate_ndcg(meal_rels, k=k),
            'precision': calculate_precision(meal_rels, k=k),
            'mrr': calculate_mrr(meal_rels),
            'num_relevant': int(num_rel),
            'num_recipes': len(meal_rels)
        }
        
        all_rels.extend(meal_rels)

    # Aggregate overall metrics
    overall = {
        'ndcg': calculate_ndcg(all_rels, k=k*3) if all_rels else 0.0,
        'precision': calculate_precision(all_rels, k=k*3) if all_rels else 0.0,
        'map': calculate_ap(all_rels),
        'mrr': calculate_mrr(all_rels),
        'safety_rate': 1.0 - (safety_violations / total_recipes) if total_recipes > 0 else 1.0,
        'total_recipes': total_recipes,
        'total_relevant': int(total_relevant)
    }

    return {
        'overall': overall,
        'per_meal': per_meal_metrics
    }


# Alias for compatibility
evaluate_recommendations = calculate_live_metrics


# =====================================================================
# BATCH EVALUATION (For Research Experiments)
# =====================================================================

def batch_evaluate_model(test_cases: List[Dict], api_url: str) -> Dict:
    """
    Runs batch evaluation on a list of test cases.
    
    Args:
        test_cases: List of test case dictionaries
        api_url: Base URL of the FRESH API
    
    Returns:
        Dict with evaluation results for each query
    """
    results = {'queries': []}

    for case in test_cases:
        print(f"\n" + "="*60)
        print(f"🔹 Testing: {case['id']} ({case.get('mode', 'Unknown')})")
        print(f"   Query: {case.get('query') or '(Auto Mode)'}")
        print(f"   Pantry: {len(case.get('pantry', []))} items")
        print("="*60)

        # Prepare API payload
        payload = {
            'user_id': case.get('user_id', 'test_user'),
            'diet': case['profile']['diet'],
            'bmi': case['profile']['bmi'],
            'pantry': case.get('pantry', []),
            'likes': case.get('query', '').split() if case.get('query') else [],
            'query_keywords': case.get('query', '').split() if case.get('query') else [],
            'indian_only': case['profile'].get('indian_mode', True),
            'num_recs': 10,
            'time_budget': 60
        }

        # Call API
        try:
            resp = requests.post(f"{api_url}/generate_meal_plan", json=payload, timeout=30)
            if resp.status_code != 200:
                print(f"❌ API Error: {resp.status_code}")
                continue
            recs = resp.json().get('meal_plan', {})
        except Exception as e:
            print(f"❌ Connection Error: {e}")
            continue

        # Prepare judge context
        prof_ctx = case['profile'].copy()
        prof_ctx['query_intent'] = case.get('query', '')
        prof_ctx['pantry'] = case.get('pantry', [])

        all_rels = []

        # Evaluate each meal
        for meal in ['breakfast', 'lunch', 'dinner']:
            print(f"\n🍽️  {meal.upper()}:")
            meal_recs = recs.get(meal, [])
            
            if not meal_recs:
                print("   (No recommendations)")
                continue
            
            for i, r in enumerate(meal_recs):
                is_rel, score, reason = llm_judge_relevance(r, prof_ctx)
                all_rels.append(1.0 if is_rel else 0.0)
                
                # Log top 3 per meal
                if i < 3:
                    icon = "✅" if is_rel else "❌"
                    pantry_pct = r.get('match_details', {}).get('pantry_match_level', '0%')
                    print(f"   {i+1}. {icon} {r['title']} [Pantry: {pantry_pct}]")
                    if not is_rel:
                        print(f"      └─ ⚠️  {reason}")

        # Calculate metrics
        metrics = {
            'MRR': calculate_mrr(all_rels),
            'MAP': calculate_ap(all_rels)
        }

        for K in [3, 5, 10]:
            metrics[f'P@{K}'] = calculate_precision(all_rels, K)
            
            # Calculate NDCG per meal and average
            ndcgs = []
            for meal in ['breakfast', 'lunch', 'dinner']:
                meal_items = recs.get(meal, [])
                if not meal_items: 
                    continue
                
                m_rels = []
                for r in meal_items:
                    is_rel, _, _ = llm_judge_relevance(r, prof_ctx)
                    m_rels.append(1.0 if is_rel else 0.0)
                
                if len(m_rels) >= 2:
                    ndcgs.append(calculate_ndcg(m_rels, k=K))
            
            metrics[f'NDCG@{K}'] = float(np.mean(ndcgs)) if ndcgs else 0.0

        print(f"\n📊 NDCG@5: {metrics['NDCG@5']:.3f} | MRR: {metrics['MRR']:.3f}")
        
        results['queries'].append({
            'id': case['id'],
            'metrics': metrics
        })

    return results