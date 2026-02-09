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

# --- IMPORTS FOR LOCAL AGENTS ---
try:
    from pantry_agent import expand_pantry_item
    from nutrition_agent import nutrition_analysis_agent
except ImportError:
    # Fallback if agents are not found locally
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
# 🧠 NEURO-SYMBOLIC DISPLAY ENGINE (HUMAN STYLE)
# =====================================================================

def generate_human_explanation(rec):
    narrative = rec.get('explanation_text', 'Optimized based on your preferences.')
    trace = rec.get('decision_trace', [])
    
    md = f"### 💡 **Why this?**\n"
    md += f"> \"{narrative}\"\n\n"
    
    if trace:
        md += "**🕵️ Model Trace:**\n"
        for step in trace:
            md += f"- {step}\n"
            
    return md

def plot_attribution_chart(attribution_data):
    """
    Creates a visual bar chart for Taste vs Pantry vs Health
    """
    if not attribution_data: return None
    
    # Clean data keys for display
    clean_data = {
        "Taste": attribution_data.get('taste_contribution', 33),
        "Pantry": attribution_data.get('pantry_contribution', 33),
        "Health": attribution_data.get('health_contribution', 33)
    }
    
    df = pd.DataFrame(list(clean_data.items()), columns=['Agent', 'Contribution'])
    
    fig = px.bar(
        df, x='Contribution', y='Agent', orientation='h', 
        text='Contribution', color='Agent',
        color_discrete_map={"Taste": "#FF6B6B", "Pantry": "#4ECDC4", "Health": "#45B7D1"}
    )
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='inside')
    fig.update_layout(
        xaxis_range=[0, 100], 
        height=120, 
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis_title=None, yaxis_title=None,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig

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
        
        # Metrics Display...
        st.markdown("### 🎯 Overall Performance")
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("NDCG@3", f"{metrics['overall']['ndcg']:.3f}")
        with col2: st.metric("Precision@3", f"{metrics['overall']['precision']:.1%}")
        with col3: st.metric("MAP@3", f"{metrics['overall']['map']:.3f}")
        with col4: st.metric("MRR", f"{metrics['overall']['mrr']:.3f}")
        
        # Charts...
        st.markdown("### 📊 Visual Analysis")
        col1, col2 = st.columns(2)
        meal_names = [m.title() for m in metrics['per_meal'].keys()]
        ndcg_vals = [m['ndcg'] for m in metrics['per_meal'].values()]
        prec_vals = [m['precision'] for m in metrics['per_meal'].values()]
        
        with col1:
            fig = go.Figure([go.Bar(x=meal_names, y=ndcg_vals, marker_color='#FF6B6B')])
            fig.update_layout(title="NDCG by Meal", height=300)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = go.Figure([go.Bar(x=meal_names, y=prec_vals, marker_color='#4ECDC4')])
            fig.update_layout(title="Precision by Meal", height=300)
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

    # --- DISPLAY LOGIC (HUMAN STYLE) ---
    # --- DISPLAY LOGIC (HUMAN STYLE) ---
    if st.session_state.recommendations:
        meals = ["breakfast", "lunch", "dinner"]
        tabs = st.tabs(["Breakfast", "Lunch", "Dinner"])
        
        for i, m_name in enumerate(meals):
            with tabs[i]:
                recs = st.session_state.recommendations.get(m_name, [])
                if not recs: st.info("No recipes found.")
                
                for rec in recs:
                    score_color = "green" if rec['score'] > 0.8 else "orange"
                    
                    with st.expander(f"**{rec['title']}** (:{score_color}[{rec['score']:.0%}])"):
                        
                        # 1. Main Columns
                        c1, c2 = st.columns([1.5, 1])
                        
                        with c1:
                            # EXPLANATION
                            explanation_md = generate_human_explanation(rec)
                            st.markdown(explanation_md)
                            
                            # MISSING ITEMS
                            missing = rec.get('missing_ingredients', [])
                            if missing:
                                st.warning(f"🛒 **Shop for:** {', '.join(missing)}")
                            else:
                                st.success("✅ Fully Pantry Matched!")
                                
                            # BADGES
                            matched_tags = rec.get('matched_nutrients', [])
                            if matched_tags:
                                st.markdown("---")
                                badges = "".join([f"<span style='background:#dbfbbd;color:#1a5e20;padding:2px 10px;border-radius:12px;margin-right:5px;border:1px solid #1a5e20'>{tag}</span>" for tag in matched_tags])
                                st.markdown(badges, unsafe_allow_html=True)
                            
                        with c2:
                            # ATTRIBUTION CHART
                            st.caption("🤖 **Decision Logic**")
                            attr_chart = plot_attribution_chart(rec.get('attribution', {}))
                            if attr_chart:
                                # ✅ FIXED: Added unique key based on meal and recipe ID
                                st.plotly_chart(
                                    attr_chart, 
                                    use_container_width=True, 
                                    config={'displayModeBar': False},
                                    key=f"attr_{m_name}_{rec['recipe_id']}" 
                                )
                            
                            # PIE CHART
                            nd = rec.get('nutrition', {'p':0})
                            st.plotly_chart(
                                px.pie(values=list(nd.values()), names=list(nd.keys()), hole=0.5), 
                                key=f"p_{m_name}_{rec['recipe_id']}", 
                                use_container_width=True
                            )
                            
                            # ACTION BUTTONS
                            if st.button("🍳 Cook This", key=f"c_{m_name}_{rec['recipe_id']}", use_container_width=True):
                                send_interaction(user.id, rec['recipe_id'], "cook", rec['title'])
                                st.toast("Logged & Learned!"); time.sleep(0.5); st.rerun()
                                
                        if rec.get('link'): st.caption(f"🔗 [View Full Recipe]({rec['link']})")

# =====================================================================
# PAGE: HISTORY
# =====================================================================
elif menu == "History":
    st.title("📜 Cooking History")
    
    # 1. Fetch Data
    h = get_user_history(st.session_state.user.id)
    
    if h:
        # --- HEADER WITH CLEAR BUTTON ---
        c1, c2 = st.columns([5, 1])
        with c1:
            st.caption(f"Found {len(h)} past activities.")
        with c2:
            if st.button("🗑️ Clear All", type="primary", use_container_width=True):
                try:
                    # 1. Execute Delete
                    response = supabase.table("interactions").delete().eq("user_id", st.session_state.user.id).execute()
                    
                    # 2. Verify Deletion (Check if data was actually returned/deleted)
                    if response.data and len(response.data) > 0:
                        st.toast(f"✅ Deleted {len(response.data)} items!")
                        time.sleep(1)
                        st.rerun()
                    else:
                        # This happens if RLS blocks the delete
                        st.error("❌ Delete failed. Check Supabase RLS policies.")
                        
                except Exception as e:
                    st.error(f"Error: {e}")
        
        st.divider()

        # --- LIST ITEMS ---
        for x in h:
            with st.container():
                c1, c2, c3 = st.columns([3, 1, 1])
                c1.markdown(f"**{x.get('recipe_title', 'Recipe')}**")
                c1.caption(x.get('created_at', '')[:10])
                
                if c2.button("👍 Like", key=f"l_{x['id']}"):
                    send_interaction(st.session_state.user.id, x['recipe_id'], "like")
                    st.toast("Liked!")
                
                if c3.button("👎 Dislike", key=f"d_{x['id']}"):
                    send_interaction(st.session_state.user.id, x['recipe_id'], "dislike")
                    st.toast("Disliked.")
                
                st.divider()
    else:
        st.info("No history found.")

# =====================================================================
# PAGE: LOGOUT
# =====================================================================
elif menu == "Logout":
    st.session_state.user = None
    st.rerun()