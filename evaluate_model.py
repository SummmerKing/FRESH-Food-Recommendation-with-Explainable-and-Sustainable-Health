"""
FRESH Model - Comprehensive Research Evaluation Suite
=====================================================
Authors: [Your Name]
Date: January 2026

This script evaluates the FRESH recommendation system across:
1. Static Benchmark (25 test cases)
2. Online Learning Experiment (Cold vs Warm start)
3. Statistical Analysis & Visualization
"""

import json
import os
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from Fresh_metrics import (
    calculate_ndcg, 
    calculate_precision, 
    calculate_mrr, 
    llm_judge_relevance
)

# ==========================================
# 🔧 CONFIGURATION
# ==========================================
API_URL = "http://localhost:8001"
TEST_USER_ID = "user_eval_main"
LEARNING_USER_ID = "user_learning_test"

# Set random seed for reproducibility
np.random.seed(42)

# ==========================================
# 🥗 PANTRY DEFINITIONS
# ==========================================

Indian_veg_pantry = [
    "basmati rice", "toor dal", "moong dal", "urad dal", "chana dal", "atta", 
    "besan", "sooji", "poha", "potatoes", "onions", "tomatoes", "ginger", 
    "garlic", "green chilies", "curry leaves", "coriander leaves", "spinach", 
    "paneer", "curd", "milk", "ghee", "mustard seeds", "cumin seeds", 
    "turmeric", "red chili powder", "garam masala", "sambar powder", "dal", 
    "chana", "rajma"
]

Indian_nonveg_pantry = Indian_veg_pantry + [
    "chicken", "chicken breast", "eggs", "mutton", "fish", "prawns", 
    "boneless chicken", "minced meat"
]

minimal_pantry = [
    "rice", "potatoes", "onions", "tomatoes", "oil", "salt"
]

south_indian_pantry = [
    "idli rice", "urad dal", "rice flour", "coconut", "curry leaves", 
    "tamarind", "sambar powder", "rasam powder", "mustard seeds", "hing",
    "drumsticks", "banana", "jaggery", "sesame oil"
]

north_indian_pantry = [
    "atta", "maida", "paneer", "butter", "cream", "kasuri methi", "rajma",
    "chana", "chickpeas", "kidney beans", "garam masala", "cumin", "coriander"
]

high_protein_pantry = [
    "chicken breast", "eggs", "greek yogurt", "quinoa", "chickpeas", 
    "lentils", "tofu", "cottage cheese", "almonds", "protein powder",
    "broccoli", "spinach", "sweet potato"
]

vegan_pantry = [
    "brown rice", "quinoa", "chickpeas", "black beans", "tofu", "tempeh",
    "nutritional yeast", "almond milk", "cashews", "chia seeds", "flax seeds",
    "kale", "sweet potato", "avocado", "coconut oil"
]

# ==========================================
# 📋 COMPREHENSIVE TEST CASES
# ==========================================

test_cases = [
    # ============================================
    # 🤖 AUTOMATIC MODE (8 cases)
    # ============================================
    
    {
        'id': 'AUTO_FULL_INDIAN_VEG',
        'category': 'Auto-Basic',
        'mode': 'Auto',
        'user_id': 'user_auto_1',
        'query': '', 
        'profile': {'diet': 'Vegetarian', 'bmi': 22.0, 'indian_mode': True, 'allergies': []},
        'pantry': Indian_veg_pantry,
        'expected_behavior': 'High pantry utilization with dal/rice/paneer recipes'
    },
    
    {
        'id': 'AUTO_NONVEG_INDIAN',
        'category': 'Auto-Basic',
        'mode': 'Auto',
        'user_id': 'user_auto_2',
        'query': '',
        'profile': {'diet': 'Non-Vegetarian', 'bmi': 24.5, 'indian_mode': True, 'allergies': []},
        'pantry': Indian_nonveg_pantry,
        'expected_behavior': 'Mix of veg and non-veg with chicken/egg recipes'
    },
    
    {
        'id': 'AUTO_MINIMAL_PANTRY',
        'category': 'Auto-Challenge',
        'mode': 'Auto',
        'user_id': 'user_auto_3',
        'query': '',
        'profile': {'diet': 'Vegetarian', 'bmi': 20.0, 'indian_mode': True, 'allergies': []},
        'pantry': minimal_pantry,
        'expected_behavior': 'Simple 2-3 ingredient recipes (resilience test)'
    },
    
    {
        'id': 'AUTO_EMPTY_PANTRY',
        'category': 'Auto-Challenge',
        'mode': 'Auto',
        'user_id': 'user_auto_4',
        'query': '',
        'profile': {'diet': 'Vegetarian', 'bmi': 21.0, 'indian_mode': True, 'allergies': []},
        'pantry': [],
        'expected_behavior': 'Basic Indian staples (fallback test)'
    },
    
    {
        'id': 'AUTO_VEGAN_PANTRY',
        'category': 'Auto-Safety',
        'mode': 'Auto',
        'user_id': 'user_auto_5',
        'query': '',
        'profile': {'diet': 'Vegan', 'bmi': 22.5, 'indian_mode': False, 'allergies': []},
        'pantry': vegan_pantry,
        'expected_behavior': 'No animal products (diet safety test)'
    },
    
    {
        'id': 'AUTO_UNDERWEIGHT',
        'category': 'Auto-Health',
        'mode': 'Auto',
        'user_id': 'user_auto_6',
        'query': '',
        'profile': {'diet': 'Vegetarian', 'bmi': 17.0, 'indian_mode': True, 'allergies': []},
        'pantry': Indian_veg_pantry,
        'expected_behavior': 'Calorie-dense foods (ghee, paneer, nuts)'
    },
    
    {
        'id': 'AUTO_OVERWEIGHT',
        'category': 'Auto-Health',
        'mode': 'Auto',
        'user_id': 'user_auto_7',
        'query': '',
        'profile': {'diet': 'Vegetarian', 'bmi': 28.0, 'indian_mode': True, 'allergies': []},
        'pantry': Indian_veg_pantry,
        'expected_behavior': 'Light meals with vegetables and dal'
    },
    
    {
        'id': 'AUTO_WITH_ALLERGIES',
        'category': 'Auto-Safety',
        'mode': 'Auto',
        'user_id': 'user_auto_8',
        'query': '',
        'profile': {'diet': 'Vegetarian', 'bmi': 22.0, 'indian_mode': True, 'allergies': ['peanuts', 'dairy']},
        'pantry': [item for item in Indian_veg_pantry if item not in ['paneer', 'milk', 'curd', 'ghee']],
        'expected_behavior': 'Strict dairy avoidance (allergy safety)'
    },
    
    # ============================================
    # 🧠 SMART MODE (17 cases)
    # ============================================
    
    # Regional Cuisine (3)
    {
        'id': 'SMART_SOUTH_FULL',
        'category': 'Smart-Regional',
        'mode': 'Smart',
        'user_id': 'user_smart_1',
        'query': 'I want south indian breakfast lunch and dinner for today',
        'profile': {'diet': 'Vegetarian', 'bmi': 22.0, 'indian_mode': True, 'allergies': []},
        'pantry': south_indian_pantry,
        'expected_behavior': 'Idli/dosa breakfast, sambar rice lunch, rasam dinner'
    },
    
    {
        'id': 'SMART_NORTH_FULL',
        'category': 'Smart-Regional',
        'mode': 'Smart',
        'user_id': 'user_smart_2',
        'query': 'suggest me north indian recipes for the whole day',
        'profile': {'diet': 'Vegetarian', 'bmi': 23.0, 'indian_mode': True, 'allergies': []},
        'pantry': north_indian_pantry,
        'expected_behavior': 'Paratha breakfast, rajma/chana lunch, paneer dinner'
    },
    
    {
        'id': 'SMART_SOUTH_NORTH_MIX',
        'category': 'Smart-Regional',
        'mode': 'Smart',
        'user_id': 'user_smart_3',
        'query': 'I want a south indian breakfast with north indian dinner for today',
        'profile': {'diet': 'Vegetarian', 'bmi': 22.0, 'indian_mode': True, 'allergies': []},
        'pantry': Indian_veg_pantry,
        'expected_behavior': 'Dosa/idli breakfast, roti/paneer dinner (regional separation)'
    },
    
    # Health Context (4)
    {
        'id': 'SMART_COLD_FLU',
        'category': 'Smart-Health',
        'mode': 'Smart',
        'user_id': 'user_smart_4',
        'query': 'i am having cold and flu, suggest me some healthy and comfort food for today',
        'profile': {'diet': 'Vegetarian', 'bmi': 21.0, 'indian_mode': True, 'allergies': []},
        'pantry': Indian_veg_pantry,
        'expected_behavior': 'Khichdi, rasam, ginger tea, soups'
    },
    
    {
        'id': 'SMART_WEIGHT_LOSS',
        'category': 'Smart-Health',
        'mode': 'Smart',
        'user_id': 'user_smart_5',
        'query': 'i want to lose weight, suggest me low calorie high protein meals',
        'profile': {'diet': 'Non-Vegetarian', 'bmi': 29.0, 'indian_mode': False, 'allergies': []},
        'pantry': high_protein_pantry,
        'expected_behavior': 'Grilled chicken, salads, egg whites, lentils'
    },
    
    {
        'id': 'SMART_MUSCLE_GAIN',
        'category': 'Smart-Health',
        'mode': 'Smart',
        'user_id': 'user_smart_6',
        'query': 'i am trying to build muscle, need high protein meals for gym',
        'profile': {'diet': 'Non-Vegetarian', 'bmi': 22.0, 'indian_mode': False, 'allergies': []},
        'pantry': high_protein_pantry,
        'expected_behavior': 'Chicken breast, eggs, quinoa bowls'
    },
    
    {
        'id': 'SMART_PREGNANCY',
        'category': 'Smart-Health',
        'mode': 'Smart',
        'user_id': 'user_smart_7',
        'query': 'i am pregnant, need nutritious meals with iron and calcium',
        'profile': {'diet': 'Vegetarian', 'bmi': 24.0, 'indian_mode': True, 'allergies': []},
        'pantry': Indian_veg_pantry,
        'expected_behavior': 'Spinach dal, ragi porridge, milk-based items'
    },
    
    # Dietary Restrictions (3)
    {
        'id': 'SMART_JAIN_DIET',
        'category': 'Smart-Dietary',
        'mode': 'Smart',
        'user_id': 'user_smart_8',
        'query': 'suggest me no onion and no garlic recipes for today',
        'profile': {'diet': 'Vegan', 'bmi': 20.5, 'indian_mode': True, 'allergies': []},
        'pantry': [],
        'expected_behavior': 'STRICT onion/garlic avoidance (negative constraint test)'
    },
    
    {
        'id': 'SMART_GLUTEN_FREE',
        'category': 'Smart-Dietary',
        'mode': 'Smart',
        'user_id': 'user_smart_9',
        'query': 'i have celiac disease, need gluten free meals',
        'profile': {'diet': 'Vegetarian', 'bmi': 21.0, 'indian_mode': False, 'allergies': ['gluten', 'wheat']},
        'pantry': vegan_pantry,
        'expected_behavior': 'Rice/quinoa based, avoid wheat/atta'
    },
    
    {
        'id': 'SMART_DIABETIC',
        'category': 'Smart-Dietary',
        'mode': 'Smart',
        'user_id': 'user_smart_10',
        'query': 'i am diabetic, suggest me low sugar low carb meals',
        'profile': {'diet': 'Vegetarian', 'bmi': 26.0, 'indian_mode': True, 'allergies': []},
        'pantry': Indian_veg_pantry,
        'expected_behavior': 'Dal/vegetables, avoid rice/potato heavy'
    },
    
    # Lifestyle (2)
    {
        'id': 'SMART_QUICK_BREAKFAST',
        'category': 'Smart-Lifestyle',
        'mode': 'Smart',
        'user_id': 'user_smart_11',
        'query': 'i need quick and easy breakfast under 15 minutes for office',
        'profile': {'diet': 'Vegetarian', 'bmi': 22.0, 'indian_mode': True, 'allergies': []},
        'pantry': Indian_veg_pantry,
        'expected_behavior': 'Poha, upma, bread sandwich, oats'
    },
    
    {
        'id': 'SMART_MEAL_PREP',
        'category': 'Smart-Lifestyle',
        'mode': 'Smart',
        'user_id': 'user_smart_12',
        'query': 'suggest me meals that i can meal prep for the entire week',
        'profile': {'diet': 'Non-Vegetarian', 'bmi': 23.0, 'indian_mode': False, 'allergies': []},
        'pantry': high_protein_pantry,
        'expected_behavior': 'Batch-cookable: dal, grilled chicken, quinoa'
    },
    
    # Edge Cases (5)
    {
        'id': 'EDGE_CONTRADICTORY',
        'category': 'Edge-Case',
        'mode': 'Smart',
        'user_id': 'user_edge_1',
        'query': 'i want high protein vegan meals with chicken',
        'profile': {'diet': 'Vegan', 'bmi': 22.0, 'indian_mode': False, 'allergies': []},
        'pantry': vegan_pantry,
        'expected_behavior': 'Should prioritize vegan diet, ignore chicken'
    },
    
    {
        'id': 'EDGE_MULTI_CONSTRAINT',
        'category': 'Edge-Case',
        'mode': 'Smart',
        'user_id': 'user_edge_2',
        'query': 'south indian vegan gluten free high protein no oil breakfast',
        'profile': {'diet': 'Vegan', 'bmi': 20.0, 'indian_mode': True, 'allergies': ['gluten']},
        'pantry': [],
        'expected_behavior': 'Extremely challenging (idli with urad dal)'
    },
    
    {
        'id': 'EDGE_FUSION',
        'category': 'Edge-Case',
        'mode': 'Smart',
        'user_id': 'user_edge_3',
        'query': 'suggest me indo-chinese fusion recipes',
        'profile': {'diet': 'Vegetarian', 'bmi': 22.0, 'indian_mode': False, 'allergies': []},
        'pantry': Indian_veg_pantry,
        'expected_behavior': 'Manchurian, fried rice, chilli paneer'
    },
    
    {
        'id': 'EDGE_SEASONAL',
        'category': 'Edge-Case',
        'mode': 'Smart',
        'user_id': 'user_edge_4',
        'query': 'suggest me summer cooling recipes',
        'profile': {'diet': 'Vegetarian', 'bmi': 21.0, 'indian_mode': True, 'allergies': []},
        'pantry': Indian_veg_pantry,
        'expected_behavior': 'Buttermilk, salads, cucumber raita'
    },
    
    {
        'id': 'EDGE_BUDGET',
        'category': 'Edge-Case',
        'mode': 'Smart',
        'user_id': 'user_edge_5',
        'query': 'i am a student, need cheap and filling meals',
        'profile': {'diet': 'Vegetarian', 'bmi': 20.0, 'indian_mode': True, 'allergies': []},
        'pantry': minimal_pantry,
        'expected_behavior': 'Dal-rice, potato curry, basic staples'
    }
]

# ==========================================
# 📐 METRIC HELPERS
# ==========================================

def calculate_ap(relevant_items):
    """Calculates Average Precision (AP) for a single ranking list."""
    if not relevant_items or sum(relevant_items) == 0:
        return 0.0
    
    hits = 0
    sum_precisions = 0.0
    for i, is_rel in enumerate(relevant_items):
        if is_rel:
            hits += 1
            sum_precisions += hits / (i + 1.0)
            
    return sum_precisions / hits

def process_metrics(recs_dict, profile, k_list=[3, 5, 10]):
    """
    Aggregates metrics across Breakfast/Lunch/Dinner.
    Returns dictionary of evaluation metrics.
    """
    all_rels = []
    total_recs = 0
    total_rel = 0
    safety_violations = 0
    
    # Collect relevance judgments for all meals
    for meal in ['breakfast', 'lunch', 'dinner']:
        meal_recs = recs_dict.get(meal, [])
        for r in meal_recs:
            is_rel, _, reason = llm_judge_relevance(r, profile)
            all_rels.append(1.0 if is_rel else 0.0)
            total_recs += 1
            if is_rel: 
                total_rel += 1
            if "Violation" in reason:
                safety_violations += 1

    # Calculate metrics
    metrics = {
        'Total_Recipes': total_recs,
        'Relevant_Recipes': total_rel,
        'Safety_Violations': safety_violations,
        'Safety_Rate': 1.0 - (safety_violations / total_recs) if total_recs > 0 else 1.0,
        'MRR': calculate_mrr(all_rels),
        'MAP': calculate_ap(all_rels)
    }
    
    for k in k_list:
        metrics[f'P@{k}'] = calculate_precision(all_rels, k)
        metrics[f'NDCG@{k}'] = calculate_ndcg(all_rels, k)
        
    return metrics

# ==========================================
# 🧪 EXPERIMENT 1: STATIC BENCHMARK
# ==========================================

def run_static_benchmark():
    """
    Runs all test cases and collects metrics.
    Prints Top 3 recipes per meal for visual verification.
    """
    rows = []
    print("\n" + "="*80)
    print(" 🧪 EXPERIMENT 1: STATIC BENCHMARK (25 Test Cases)")
    print("="*80)

    for i, case in enumerate(test_cases, 1):
        print(f"\n[{i}/{len(test_cases)}] 🔹 {case['id']}")
        print(f"   Category: {case['category']}")
        print(f"   Query: {case['query'] if case['query'] else '(Auto Mode)'}")
        print("-" * 60)
        
        payload = {
            'user_id': case['user_id'],
            'diet': case['profile']['diet'],
            'bmi': case['profile']['bmi'],
            'pantry': case['pantry'],
            'likes': case['query'].split() if case['query'] else [],
            'query_keywords': case['query'].split() if case['query'] else [],
            'indian_only': case['profile']['indian_mode'],
            'num_recs': 10,
            'time_budget': 60
        }

        try:
            resp = requests.post(f"{API_URL}/generate_meal_plan", json=payload, timeout=30)
            if resp.status_code != 200:
                print(f"   ❌ API Error: {resp.status_code}")
                continue
            recs = resp.json().get('meal_plan', {})
        except Exception as e:
            print(f"   ❌ Connection Error: {e}")
            continue

        # Prepare context for judge
        prof_ctx = case['profile'].copy()
        prof_ctx['query_intent'] = case['query']
        prof_ctx['pantry'] = case['pantry']

        # --- 🟢 DISPLAY RECOMMENDATIONS LOGIC START ---
        for meal in ['breakfast', 'lunch', 'dinner']:
            meal_recs = recs.get(meal, [])
            if not meal_recs:
                continue
            
            print(f"   🍽️  {meal.upper()}:")
            for idx, r in enumerate(meal_recs):
                if idx < 3: # Display only top 3
                    is_rel, _, reason = llm_judge_relevance(r, prof_ctx)
                    icon = "✅" if is_rel else "❌"
                    p_match = r.get('match_details', {}).get('pantry_match_level', '0%')
                    
                    print(f"      {idx+1}. {icon} {r['title']} [Pantry: {p_match}]")
                    if not is_rel:
                        print(f"         └─ ⚠️  {reason}")
        # --- 🟢 DISPLAY RECOMMENDATIONS LOGIC END ---

        # Calculate metrics
        m = process_metrics(recs, prof_ctx)
        
        # Build result row
        row = {
            'Test_Case': case['id'],
            'Category': case['category'],
            'Mode': case['mode']
        }
        row.update(m)
        rows.append(row)
        
        print(f"   📊 Metrics: NDCG@5={m['NDCG@5']:.3f} | MRR={m['MRR']:.3f}")
        
    return rows

# ==========================================
# 🧠 EXPERIMENT 2: ONLINE LEARNING
# ==========================================

def specialized_judge(recipe, hidden_pref):
    """
    Specialized judge for the learning experiment.
    Checks if recipe matches the hidden South Indian preference.
    """
    title = str(recipe.get('title', '')).lower()
    ings = " ".join(recipe.get('ingredients', [])).lower()
    full_text = title + " " + ings
    
    # South Indian keywords
    keywords = ["dosa", "idli", "sambar", "vada", "curry leaf", "coconut", 
                "chettinad", "kerala", "upma", "rice", "rasam", "appam"]
    
    hits = [k for k in keywords if k in full_text]
    return (True, 1.0, "South Indian Match") if hits else (False, 0.0, "Not South Indian")

def run_learning_experiment():
    """
    Tests the system's ability to learn from user interactions.
    Compares cold start vs warm start performance.
    """
    print("\n" + "="*80)
    print(" 🧠 EXPERIMENT 2: ONLINE LEARNING TEST")
    print("="*80)
    
    rows = []
    hidden_pref = "south indian"
    
    base_payload = {
        "user_id": LEARNING_USER_ID,
        "likes": [], 
        "pantry": ["rice", "dal", "spices"], 
        "query_keywords": [], 
        "diet": "Vegetarian", 
        "bmi": 22.0,
        "num_recs": 10,
        "time_budget": 60
    }

    # --- ROUND 1: COLD START ---
    print("\n❄️  ROUND 1: Cold Start (No User History)")
    print("-" * 60)
    
    try:
        resp1 = requests.post(f"{API_URL}/generate_meal_plan", json=base_payload, timeout=30)
        recs1 = resp1.json()['meal_plan']
    except Exception as e:
        print(f"❌ Cold start failed: {e}")
        return rows
    
    # Judge & Print Round 1
    rels1 = []
    target_id, target_title = None, ""
    
    all_recs_1 = recs1.get('dinner', []) # Focus on Dinner for simplicity
    
    print("   🍽️  DINNER RECOMMENDATIONS:")
    for idx, r in enumerate(all_recs_1):
        is_rel, _, reason = specialized_judge(r, hidden_pref)
        rels1.append(1.0 if is_rel else 0.0)
        
        if idx < 5:
            icon = "✅" if is_rel else "❌"
            print(f"      {idx+1}. {icon} {r['title']}")

        if is_rel and not target_id:
            target_id = r['recipe_id']
            target_title = r['title']

    # Calculate metrics for Round 1
    m1 = {
        'Total_Recipes': len(all_recs_1),
        'Relevant_Recipes': int(sum(rels1)),
        'Safety_Violations': 0,
        'Safety_Rate': 1.0,
        'MRR': calculate_mrr(rels1),
        'MAP': calculate_ap(rels1)
    }
    for k in [3, 5, 10]:
        m1[f'P@{k}'] = calculate_precision(rels1, k)
        m1[f'NDCG@{k}'] = calculate_ndcg(rels1, k)
    
    row1 = {'Test_Case': 'LEARNING_COLD_START', 'Category': 'Learning', 'Mode': 'Cold'}
    row1.update(m1)
    rows.append(row1)
    
    print(f"   📊 Cold Start NDCG@5: {m1['NDCG@5']:.3f}")

    if not target_id:
        print("\n⚠️  No relevant South Indian recipe found to 'Like'. Cannot proceed.")
        return rows

    # --- USER INTERACTION ---
    print(f"\n👍 SIMULATING INTERACTION: Liking '{target_title}'")
    print("-" * 60)
    
    # Send 3 strong positive signals
    for _ in range(3):
        try:
            requests.post(f"{API_URL}/log_cooking", json={
                "user_id": LEARNING_USER_ID, 
                "recipe_id": target_id, 
                "interaction_type": "like", 
                "recipe_title": target_title
            }, timeout=10)
        except: pass
    
    time.sleep(2)  # Allow DB update

    # --- ROUND 2: WARM START ---
    print("\n🔥 ROUND 2: Warm Start (After Learning)")
    print("-" * 60)
    
    try:
        resp2 = requests.post(f"{API_URL}/generate_meal_plan", json=base_payload, timeout=30)
        recs2 = resp2.json()['meal_plan']
    except Exception as e:
        print(f"❌ Warm start failed: {e}")
        return rows
    
    # Judge & Print Round 2
    rels2 = []
    all_recs_2 = recs2.get('dinner', [])
    
    print("   🍽️  DINNER RECOMMENDATIONS (ADAPTED):")
    for idx, r in enumerate(all_recs_2):
        is_rel, _, reason = specialized_judge(r, hidden_pref)
        rels2.append(1.0 if is_rel else 0.0)
        
        if idx < 5:
            icon = "✅" if is_rel else "❌"
            print(f"      {idx+1}. {icon} {r['title']}")

    # Calculate metrics for Round 2
    m2 = {
        'Total_Recipes': len(all_recs_2),
        'Relevant_Recipes': int(sum(rels2)),
        'Safety_Violations': 0,
        'Safety_Rate': 1.0,
        'MRR': calculate_mrr(rels2),
        'MAP': calculate_ap(rels2)
    }
    for k in [3, 5, 10]:
        m2[f'P@{k}'] = calculate_precision(rels2, k)
        m2[f'NDCG@{k}'] = calculate_ndcg(rels2, k)

    row2 = {'Test_Case': 'LEARNING_WARM_START', 'Category': 'Learning', 'Mode': 'Warm'}
    row2.update(m2)
    rows.append(row2)
    
    print(f"   📊 Warm Start NDCG@5: {m2['NDCG@5']:.3f}")
    
    # Calculate improvement
    improvement = ((m2['NDCG@5'] - m1['NDCG@5']) / max(m1['NDCG@5'], 0.001)) * 100
    print(f"\n   📈 Learning Improvement: {improvement:+.1f}%")
    
    return rows

# ==========================================
# 📊 VISUALIZATION & ANALYSIS
# ==========================================

def generate_comprehensive_graphs(df):
    """
    Generates publication-quality visualizations for the research paper.
    """
    print("\n📊 Generating Publication-Quality Graphs...")
    
    # Set professional style
    plt.style.use('seaborn-v0_8-paper')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(18, 12))
    
    # ===== GRAPH 1: Category-wise NDCG@5 =====
    ax1 = plt.subplot(2, 3, 1)
    category_stats = df.groupby('Category')['NDCG@5'].agg(['mean', 'std']).reset_index()
    bars = ax1.bar(range(len(category_stats)), category_stats['mean'], 
                   yerr=category_stats['std'], capsize=5,
                   color=['#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#59A14F', '#EDC948'])
    ax1.set_xticks(range(len(category_stats)))
    ax1.set_xticklabels(category_stats['Category'], rotation=45, ha='right')
    ax1.set_ylabel('NDCG@5 Score')
    ax1.set_title('Performance by Category (NDCG@5)', fontsize=12, fontweight='bold')
    ax1.set_ylim(0, 1.1)
    
    # ===== GRAPH 2: Auto vs Smart Mode Comparison =====
    ax2 = plt.subplot(2, 3, 2)
    mode_stats = df[df['Mode'].isin(['Auto', 'Smart'])].groupby('Mode')[['NDCG@5', 'MAP', 'MRR']].mean()
    x = np.arange(len(mode_stats.columns))
    width = 0.35
    ax2.bar(x - width/2, mode_stats.loc['Auto'], width, label='Auto Mode', color='#4E79A7')
    ax2.bar(x + width/2, mode_stats.loc['Smart'], width, label='Smart Mode', color='#F28E2B')
    ax2.set_ylabel('Score')
    ax2.set_title('Auto vs Smart Mode Comparison', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(['NDCG@5', 'MAP', 'MRR'])
    ax2.legend()
    ax2.set_ylim(0, 1.1)
    
    # ===== GRAPH 3: Learning Experiment =====
    ax3 = plt.subplot(2, 3, 3)
    learning_df = df[df['Category'] == 'Learning']
    if len(learning_df) == 2:
        modes = ['Cold Start', 'Warm Start']
        ndcg_scores = learning_df['NDCG@5'].values
        bars = ax3.bar(modes, ndcg_scores, color=['#E15759', '#59A14F'], width=0.5)
        ax3.set_ylabel('NDCG@5 Score')
        ax3.set_title('Online Learning Performance', fontsize=12, fontweight='bold')
        ax3.set_ylim(0, 1.1)
        for bar, val in zip(bars, ndcg_scores):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height, f'{val:.3f}', ha='center', va='bottom')
    
    # ===== GRAPH 4: Safety Analysis =====
    ax4 = plt.subplot(2, 3, 4)
    safety_rates = df.groupby('Category')['Safety_Rate'].mean().sort_values(ascending=False)
    bars = ax4.barh(range(len(safety_rates)), safety_rates.values, color='#76B7B2')
    ax4.set_yticks(range(len(safety_rates)))
    ax4.set_yticklabels(safety_rates.index)
    ax4.set_xlabel('Safety Rate')
    ax4.set_title('Dietary Safety Compliance', fontsize=12, fontweight='bold')
    ax4.set_xlim(0, 1.1)
    
    # ===== GRAPH 5: Precision vs NDCG Scatter =====
    ax5 = plt.subplot(2, 3, 5)
    for category in df['Category'].unique():
        cat_data = df[df['Category'] == category]
        ax5.scatter(cat_data['P@5'], cat_data['NDCG@5'], label=category, s=100, alpha=0.7)
    ax5.set_xlabel('Precision@5')
    ax5.set_ylabel('NDCG@5')
    ax5.set_title('Precision vs NDCG Correlation', fontsize=12, fontweight='bold')
    ax5.legend(loc='best', fontsize=8)
    ax5.grid(alpha=0.3)
    
    # ===== GRAPH 6: Top & Bottom Performers =====
    ax6 = plt.subplot(2, 3, 6)
    sorted_df = df.sort_values('NDCG@5', ascending=False)
    top_5 = sorted_df.head(5)
    bottom_5 = sorted_df.tail(5)
    combined = pd.concat([top_5, bottom_5])
    colors = ['#59A14F'] * 5 + ['#E15759'] * 5
    bars = ax6.barh(range(len(combined)), combined['NDCG@5'], color=colors)
    ax6.set_yticks(range(len(combined)))
    ax6.set_yticklabels([tc[:20] + '...' if len(tc) > 20 else tc for tc in combined['Test_Case']], fontsize=9)
    ax6.set_xlabel('NDCG@5 Score')
    ax6.set_title('Top 5 & Bottom 5 Test Cases', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('fresh_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: 'fresh_comprehensive_analysis.png'")
    
    # ===== PERFORMANCE HEATMAP =====
    fig2, ax = plt.subplots(figsize=(12, 8))
    heatmap_data = df.pivot_table(index='Test_Case', values=['NDCG@5', 'MAP', 'MRR', 'P@5', 'Safety_Rate'], aggfunc='mean')
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn', center=0.5, vmin=0, vmax=1, ax=ax)
    ax.set_title('FRESH Model Performance Heatmap', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('fresh_performance_heatmap.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: 'fresh_performance_heatmap.png'")

def generate_statistical_analysis(df):
    """Generates statistical summary for the research paper."""
    print("\n" + "="*80)
    print(" 📈 STATISTICAL ANALYSIS")
    print("="*80)
    
    print("\n1. OVERALL PERFORMANCE:")
    metrics = ['NDCG@5', 'MAP', 'MRR', 'P@5', 'Safety_Rate']
    print(df[metrics].agg(['mean', 'std', 'min', 'max']).to_string(float_format='%.3f'))
    
    print("\n2. CATEGORY-WISE PERFORMANCE:")
    category_stats = df.groupby('Category')['NDCG@5'].agg(['count', 'mean', 'std'])
    print(category_stats.to_string(float_format='%.3f'))

# ==========================================
# 🚀 MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print(" 🚀 FRESH MODEL RESEARCH EVALUATION SUITE")
    print(" Version 1.0 | January 2026")
    print("="*80)
    
    try:
        requests.get(API_URL, timeout=5)
    except:
        print(f"❌ CRITICAL: API is offline at {API_URL}")
        exit(1)

    # 1. Run Static Benchmark
    static_results = run_static_benchmark()
    
    # 2. Run Learning Experiment
    learning_results = run_learning_experiment()
    
    # 3. Combine & Analyze
    all_results = static_results + learning_results
    df = pd.DataFrame(all_results)
    
    generate_statistical_analysis(df)
    
    print("\n" + "="*80)
    print(" 📄 COMPREHENSIVE RESULTS TABLE")
    print("="*80)
    display_cols = ['Test_Case', 'Category', 'Mode', 'NDCG@5', 'MAP', 'MRR', 'P@5', 'Safety_Rate']
    print(df[display_cols].to_string(index=False, float_format='%.3f'))
    
    df.to_csv("fresh_research_results.csv", index=False)
    generate_comprehensive_graphs(df)
    
    print("\n✅ EVALUATION COMPLETE! Files generated.")