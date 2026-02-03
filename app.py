import streamlit as st
import requests
import json
import time
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import random
import re
from supabase import create_client, Client
import os

# --- PAGE CONFIGURATION (Must be first) ---
st.set_page_config(page_title="FRESH: Intelligent Food AI", layout="wide", page_icon="🥑")

# --- ROBUST METRICS IMPORT ---
METRICS_AVAILABLE = False
try:
    from Fresh_metrics import (
        calculate_live_metrics,
        evaluate_recommendations,
        llm_judge_relevance,
        calculate_ndcg, 
        calculate_precision
    )
    METRICS_AVAILABLE = True
    print("✅ Metrics loaded successfully.")
except Exception as e:
    print(f"❌ CRITICAL ERROR IMPORTING METRICS: {e}") 
    METRICS_AVAILABLE = False
    st.error(f"Error loading metrics: {e}")

# --- IMPORTS FOR LOCAL AGENTS ---
try:
    from pantry_agent import expand_pantry_item
    from nutrition_agent import nutrition_analysis_agent
except ImportError:
    def expand_pantry_item(item): return [item]
    def nutrition_analysis_agent(ctx, query, profile): return {"insight": "Standard Optimization", "recommended_keywords": []}

# --- CONFIGURATION ---
API_URL = "http://localhost:8001"
SUPABASE_URL = "https://wrwbqawmwqcknqlntpfb.supabase.co"
SUPABASE_KEY = "sb_publishable_tEak2s2lQyaMgBu4fQmG6Q_IjkcWLZU"

@st.cache_resource
def init_supabase():
    return create_client(SUPABASE_URL, SUPABASE_KEY)

supabase = init_supabase()

# --- SESSION STATE INITIALIZATION ---
if "user" not in st.session_state: st.session_state.user = None
if "profile" not in st.session_state: st.session_state.profile = {}
if "pantry" not in st.session_state: st.session_state.pantry = []
if "recommendations" not in st.session_state: st.session_state.recommendations = {}
if "settings" not in st.session_state:
    st.session_state.settings = {"indian_only": False, "num_recs": 3, "time_budget": 45}

# =====================================================================
# 🧠 NEURO-SYMBOLIC LOGIC & EXPLAINABILITY
# =====================================================================

def calculate_shapley_values(neural_score, pantry_score, penalty_val, weights):
    alpha, beta, lam = weights
    contributions = {
        'Taste_Agent': alpha * neural_score,
        'Pantry_Agent': beta * pantry_score,
        'Safety_Agent': -1 * (lam * penalty_val)
    }
    total_score = (alpha * neural_score) + (beta * pantry_score) - (lam * penalty_val)
    return contributions, total_score

def get_nutrition_rag_insight(recipe_title, ingredients, nutrition_data):
    macros = {k.lower(): v for k, v in nutrition_data.items()}
    p = macros.get('protein', 0)
    c = macros.get('carbs', 0)
    f = macros.get('fats', 0)
    total_mass = p + c + f or 1 
    
    insight = ""
    if p / total_mass > 0.30:
        insight = f"💪 **High Protein ({int(p)}g):** Excellent for muscle repair. "
    elif f / total_mass > 0.45:
        insight = f"🥑 **Keto-Friendly ({int(f)}g Fat):** High healthy fats. "
    elif c / total_mass > 0.60:
        insight = f"⚡ **Energy Boost ({int(c)}g Carbs):** Good pre-workout. "
    else:
        insight = "⚖️ **Balanced Profile:** Good mix for maintenance. "

    title_lower = recipe_title.lower()
    if "chicken" in title_lower: insight += "Complete lean protein."
    elif "paneer" in title_lower: insight += "Casein protein source."
    elif "dal" in title_lower: insight += "Fiber rich."
    elif "spinach" in title_lower: insight += "High Iron."
    
    return insight

def generate_rag_enhanced_explanation(recipe_meta, neural_score, pantry_score, is_violation):
    weights = (1.0, 1.0, 0.3)
    penalty_val = 1.0 if is_violation else 0.0
    shaps, score = calculate_shapley_values(neural_score, pantry_score, penalty_val, weights)
    
    nutrition_data = recipe_meta.get('nutrition', {})
    rag_insight = get_nutrition_rag_insight(recipe_meta['title'], recipe_meta.get('ingredients', []), nutrition_data)
    
    md = f"### 🥗 **Analysis: {recipe_meta['title']}**\n"
    md += f"> {rag_insight}\n\n"
    md += "**Evaluation Factors:**\n"
    
    taste_emoji = "🤤" if shaps['Taste_Agent'] > 0.8 else "😋"
    md += f"- {taste_emoji} **Taste Match:** `{shaps['Taste_Agent']:.2f}`\n"
    
    if shaps['Pantry_Agent'] > 0.5:
        md += f"- 🎒 **Pantry:** `{shaps['Pantry_Agent']:.2f}` (Uses ingredients)\n"
    else:
        md += f"- 🛒 **Pantry:** `{shaps['Pantry_Agent']:.2f}` (Shop needed)\n"
    
    if is_violation:
        md += f"- 🛡️ **Dietary Check:** `{shaps['Safety_Agent']:.2f}` (Penalty Applied)\n"
    else:
        md += f"- 🛡️ **Dietary Check:** `+0.00` (Safe)\n"

    return md

# =====================================================================
# DATABASE & API HELPERS
# =====================================================================
def fetch_user_data(user_id):
    try:
        p_res = supabase.table("users").select("*").eq("id", user_id).execute()
        if p_res.data: st.session_state.profile = p_res.data[0]
        i_res = supabase.table("pantry_items").select("ingredient_name").eq("user_id", user_id).execute()
        st.session_state.pantry = [row['ingredient_name'] for row in i_res.data]
    except Exception as e: st.error(f"Error loading data: {e}")

def db_add_pantry_item(item):
    if not st.session_state.user: return
    try:
        supabase.table("pantry_items").insert({
            "user_id": st.session_state.user.id, "ingredient_name": item
        }).execute()
        if item not in st.session_state.pantry: st.session_state.pantry.append(item)
    except Exception as e: print(f"DB Error: {e}")

def db_update_profile(data):
    if not st.session_state.user: return
    try:
        supabase.table("users").update(data).eq("id", st.session_state.user.id).execute()
        st.session_state.profile.update(data)
        st.success("✅ Profile Synced!")
    except Exception as e: st.error(f"Save Failed: {e}")

def send_interaction(user_id, recipe_id, action_type, title=""):
    payload = {"user_id": user_id, "recipe_id": recipe_id, "interaction_type": action_type, "recipe_title": title}
    try:
        requests.post(f"{API_URL}/log_cooking", json=payload)
        st.cache_data.clear()
        return True
    except Exception as e:
        print(f"API Error: {e}")
        return False

def get_user_history(user_id):
    try:
        res = supabase.table("interactions").select("*").eq("user_id", user_id).order("created_at", desc=True).limit(20).execute()
        return res.data
    except Exception as e: return []

# =====================================================================
# SIDEBAR
# =====================================================================
st.sidebar.title("FRESH Navigation")

if st.session_state.user:
    menu = st.sidebar.radio("Go to:", ["Dashboard", "History", "Profile", "📊 Metrics", "Logout"])
    st.sidebar.divider()
    
    st.sidebar.header("🥕 Your Pantry")
    def add_pantry_item_callback():
        raw_item = st.session_state.new_item_input.strip()
        if raw_item:
            with st.spinner(f"🥕 AI expanding '{raw_item}'..."):
                try:
                    expanded = expand_pantry_item(raw_item)
                    for tag in expanded:
                        if tag.lower().strip() not in st.session_state.pantry:
                            db_add_pantry_item(tag.lower().strip())
                    st.toast("✅ Pantry Updated")
                except: db_add_pantry_item(raw_item)
        st.session_state.new_item_input = "" 

    st.sidebar.text_input("Add Ingredients:", key="new_item_input", on_change=add_pantry_item_callback)
    if st.session_state.pantry:
        st.sidebar.write(f"**Items ({len(st.session_state.pantry)}):**")
        for item in st.session_state.pantry: st.sidebar.caption(f"• {item}")
        if st.sidebar.button("🗑️ Clear Pantry"):
            supabase.table("pantry_items").delete().eq("user_id", st.session_state.user.id).execute()
            st.session_state.pantry = []
            st.rerun()
else: menu = "Login"

# =====================================================================
# PAGE: LOGIN
# =====================================================================
if menu == "Login":
    st.title("Welcome to FRESH 🥑")
    tab1, tab2 = st.tabs(["Sign In", "Create Account"])
    with tab1:
        email = st.text_input("Email", key="l_e"); pwd = st.text_input("Password", type="password", key="l_p")
        if st.button("Sign In"):
            try:
                res = supabase.auth.sign_in_with_password({"email": email, "password": pwd})
                st.session_state.user = res.user; fetch_user_data(res.user.id); st.rerun()
            except Exception as e: st.error(f"Error: {e}")
    with tab2:
        n_email = st.text_input("Email", key="s_e"); n_pwd = st.text_input("Password", type="password", key="s_p")
        if st.button("Sign Up"):
            try:
                supabase.auth.sign_up({"email": n_email, "password": n_pwd})
                st.success("Account created! Sign in now.")
            except Exception as e: st.error(f"Error: {e}")

# =====================================================================
# PAGE: PROFILE
# =====================================================================
elif menu == "Profile":
    st.title("👤 User Profile")
    prof = st.session_state.profile
    c1, c2 = st.columns(2)
    with c1:
        d_opts = ["Non-Veg", "Vegetarian", "Vegan"]
        curr_d = (prof.get("dietary_constraints") or ["Non-Veg"])[0]
        s_diet = st.selectbox("Diet", d_opts, index=d_opts.index(curr_d) if curr_d in d_opts else 0)
        s_alg = st.multiselect("Allergies", ["Nuts", "Dairy", "Gluten"], default=prof.get("allergies", []))
        st.markdown("**Generator Settings**")
        s_ind = st.checkbox("🇮🇳 Indian Only", value=st.session_state.settings["indian_only"])
        s_num = st.slider("Recipes/Meal", 1, 5, value=st.session_state.settings["num_recs"])
        s_time = st.slider("Time (mins)", 15, 120, value=st.session_state.settings["time_budget"])
    with c2:
        bd = prof.get("bmi_data", {})
        h = st.number_input("Height (cm)", 100, 250, value=bd.get("height", 175))
        w = st.number_input("Weight (kg)", 30, 200, value=bd.get("weight", 70))
        st.metric("BMI", f"{w/((h/100)**2):.1f}")
    
    if st.button("💾 Save"):
        db_update_profile({"dietary_constraints":[s_diet], "allergies":s_alg, "bmi_data":{"height":h,"weight":w,"bmi":round(w/((h/100)**2),2)}})
        st.session_state.settings.update({"indian_only":s_ind, "num_recs":s_num, "time_budget":s_time})
        st.toast("Saved!")

# =====================================================================
# PAGE: METRICS
# =====================================================================
elif menu == "📊 Metrics":
    st.title("📊 Recommendation Quality Metrics")
    
    if not st.session_state.recommendations:
        st.info("Generate a meal plan first to see metrics!")
    elif not METRICS_AVAILABLE:
        st.error("⚠️ `fresh_metrics.py` is missing! Cannot run evaluation.")
    else:
        with st.spinner("🔍 AI Judge Analyzing Recommendation Quality..."):
            user_profile = {
                'diet': (st.session_state.profile.get("dietary_constraints") or ["Non-Veg"])[0],
                'allergies': st.session_state.profile.get("allergies", []),
                'bmi': st.session_state.profile.get("bmi_data", {}).get("bmi", 22.0),
                'indian_mode': st.session_state.settings.get("indian_only", False)
            }
            
            metrics = evaluate_recommendations(
                st.session_state.recommendations,
                user_profile,
                k=st.session_state.settings.get("num_recs", 3)
            )
        
        # 1. Overall Metrics
        st.markdown("### 🎯 Overall Performance")
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("NDCG@3", f"{metrics['overall']['ndcg']:.3f}", help="Ranking Quality")
        with col2: st.metric("Precision@3", f"{metrics['overall']['precision']:.1%}", help="Relevance")
        with col3: st.metric("MAP@3", f"{metrics['overall']['map']:.3f}", help="Mean Average Precision")
        with col4: st.metric("MRR", f"{metrics['overall']['mrr']:.3f}", help="Reciprocal Rank")
        
        # 2. Safety Dashboard
        st.markdown("### 🛡️ Safety Analysis")
        col1, col2 = st.columns(2)
        with col1: st.metric("Dietary Safety Rate", f"{metrics['overall']['safety_rate']:.1%}", help="Adherence to constraints")
        with col2: st.metric("Total Recommendations", f"{metrics['overall']['total_recipes']}", delta=f"{metrics['overall']['total_relevant']} relevant")
        
        # 3. Per-Meal Data
        st.markdown("### 📋 Per-Meal Breakdown")
        meal_data = []
        for meal, m in metrics['per_meal'].items():
            meal_data.append({
                'Meal': meal.title(),
                'NDCG': f"{m['ndcg']:.3f}",
                'Precision': f"{m['precision']:.1%}",
                'MRR': f"{m['mrr']:.3f}",
                'Relevant': f"{m['num_relevant']}/{m['num_recipes']}"
            })
        st.dataframe(pd.DataFrame(meal_data), use_container_width=True, hide_index=True)
        
        # 4. Charts
        st.markdown("### 📊 Visual Analysis")
        col1, col2 = st.columns(2)
        
        meal_names = [m.title() for m in metrics['per_meal'].keys()]
        ndcg_vals = [m['ndcg'] for m in metrics['per_meal'].values()]
        prec_vals = [m['precision'] for m in metrics['per_meal'].values()]
        
        with col1:
            fig = go.Figure([go.Bar(x=meal_names, y=ndcg_vals, marker_color='#FF6B6B', text=[f"{v:.2f}" for v in ndcg_vals], textposition='auto')])
            fig.update_layout(title="NDCG by Meal", yaxis_range=[0, 1], height=300)
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            fig = go.Figure([go.Bar(x=meal_names, y=prec_vals, marker_color='#4ECDC4', text=[f"{v:.1%}" for v in prec_vals], textposition='auto')])
            fig.update_layout(title="Precision by Meal", yaxis_range=[0, 1], height=300)
            st.plotly_chart(fig, use_container_width=True)

# =====================================================================
# PAGE: DASHBOARD (MAIN UI)
# =====================================================================
elif menu == "Dashboard":
    st.title("🍽️ Your Daily Meal Plan")
    user = st.session_state.user
    prof = st.session_state.profile
    
    constraints = prof.get("dietary_constraints") or []
    diet_label = constraints[0] if len(constraints) > 0 else "Non-Veg"
    indian_mode = st.session_state.settings.get("indian_only", False)
    mode_text = "🇮🇳 Indian Mode" if indian_mode else "Global Mode"
    
    st.markdown(f"""
    <div style="background-color: #1E1E1E; padding: 10px; border-radius: 5px;">
        <b>⚙️ Status:</b> <span style="color:#0f0">{diet_label}</span> | 
        <span style="color:#fa0">{mode_text}</span> |
        <span style="color:#0af">{len(st.session_state.pantry)} Pantry Items</span>
    </div><br>""", unsafe_allow_html=True)

    q = st.text_input("Ask the Chef...", placeholder="e.g. High protein breakfast")
    c1, c2 = st.columns([1, 1])
    with c1: b_smart = st.button("✨ Smart Generate", use_container_width=True)
    with c2: b_auto = st.button("🔄 Auto Generate", use_container_width=True)
    
    # --- GENERATION LOGIC ---
    if b_smart or b_auto:
        run_agent = b_smart or (b_auto and q)
        kws = []
        reason = "Standard Optimization"
        bmi_val = prof.get("bmi_data", {}).get("bmi", 22.0)
        
        if run_agent and q:
            with st.spinner("🧬 Dr. FRESH is analyzing..."):
                nd = nutrition_analysis_agent([], q, {"diet": diet_label, "bmi": bmi_val})
                kws = nd.get("recommended_keywords", [])
                reason = nd.get("insight", "Optimized")
                st.toast(f"Insight: {reason[:40]}...")

        with st.spinner("🍳 Cooking recommendations..."):
            payload = {
                "user_id": user.id, "pantry": st.session_state.pantry, "likes": kws,
                "diet": diet_label, "time_budget": st.session_state.settings["time_budget"],
                "num_recs": st.session_state.settings["num_recs"],
                "bmi": float(bmi_val), "regenerate": True, 
                "indian_only": st.session_state.settings["indian_only"], "query_keywords": kws
            }
            try:
                res = requests.post(f"{API_URL}/generate_meal_plan", json=payload)
                if res.status_code == 200:
                    st.session_state.recommendations = res.json()["meal_plan"]
                    st.success(f"✅ Generated: {reason}")
                else: st.error(res.text)
            except: st.error("Backend offline. Run main.py!")

    # --- DISPLAY LOGIC (FIXED) ---
    if st.session_state.recommendations:
        meals = ["breakfast", "lunch", "dinner"]
        tabs = st.tabs(["Breakfast", "Lunch", "Dinner"])
        
        for i, m_name in enumerate(meals):
            with tabs[i]:
                recs = st.session_state.recommendations.get(m_name, [])
                if not recs: st.info("No recipes found.")
                
                for rec in recs:
                    with st.expander(f"**{rec['title']}** ({rec['score']:.0%})"):
                        try:
                            n_score = rec['score']
                            p_txt = rec['match_details']['pantry_match_level'].replace('%', '')
                            p_score = float(p_txt) / 100
                        except: n_score, p_score = 0.8, 0.5
                        
                        is_violation = False
                        non_veg_keywords = ["chicken", "beef", "pork", "fish", "egg", "ham"]
                        if "Veg" in diet_label: 
                             if any(k in rec['title'].lower() for k in non_veg_keywords): is_violation = True
                        missing = rec.get('missing_ingredients', [])
                        if missing:
                            st.warning(f"🛒 **Missing Items:** {', '.join(missing)}")
                        else:
                            st.success("✅ You have all main ingredients!")
                        explanation_md = generate_rag_enhanced_explanation(rec, n_score, p_score, is_violation)
                        st.markdown(explanation_md)

                        c1, c2 = st.columns([1.5, 1])
                        with c1:
                            matched_tags = rec.get('matched_nutrients', [])
                            if matched_tags:
                                st.markdown("---")
                                badges = "".join([f"<span style='background:#dbfbbd;color:#1a5e20;padding:2px 10px;border-radius:12px;margin-right:5px;border:1px solid #1a5e20'>✅ {tag}</span>" for tag in matched_tags])
                                st.markdown(badges, unsafe_allow_html=True)
                            if rec.get('link'): st.markdown(f"[View Recipe]({rec['link']})")
                            
                        with c2:
                            nd = rec.get('nutrition', {'p':0})
                            st.plotly_chart(
                                px.pie(values=list(nd.values()), names=list(nd.keys()), hole=0.5), 
                                key=f"p_{m_name}_{rec['recipe_id']}", 
                                use_container_width=True
                            )
                            
                            if st.button("🍳 Cook This", key=f"c_{m_name}_{rec['recipe_id']}", use_container_width=True):
                                send_interaction(user.id, rec['recipe_id'], "cook", rec['title'])
                                st.toast("Logged & Learned!"); time.sleep(0.5); st.rerun()

# =====================================================================
# PAGE: HISTORY
# =====================================================================
elif menu == "History":
    st.title("📜 Cooking History")
    h = get_user_history(st.session_state.user.id)
    if not h: st.info("No history yet.")
    for x in h:
        with st.container():
            c1, c2, c3 = st.columns([3, 1, 1])
            c1.markdown(f"**{x.get('recipe_title', 'Recipe')}**")
            c1.caption(x['created_at'][:10])
            if c2.button("👍 Like", key=f"l_{x['id']}"):
                send_interaction(st.session_state.user.id, x['recipe_id'], "like")
                st.success("Liked!")
            if c3.button("👎 Dislike", key=f"d_{x['id']}"):
                send_interaction(st.session_state.user.id, x['recipe_id'], "dislike")
                st.warning("Disliked.")
            st.divider()

# =====================================================================
# PAGE: LOGOUT
# =====================================================================
elif menu == "Logout":
    st.session_state.user = None
    st.rerun()