import streamlit as st
import json
import requests
import pandas as pd

st.set_page_config(page_title="AI Recipe Recommender", page_icon="🥗", layout="wide")

# Backend API URL
API_URL = "http://127.0.0.1:8000"

# --- HELPER FUNCTION ---
def get_recommendations(user_data):
    """Call the FastAPI backend to get recommendations"""
    try:
        # Build request payload matching your RecommendationRequest model
        payload = {
            "user_id": user_data.get("user_id", st.session_state['user']['name']),
            "diet": user_data.get("diet"),
            "allergies": user_data.get("allergies", []),
            "dislikes": user_data.get("dislikes", []),
            "time_budget": user_data.get("time_budget", {}),
            "pantry": user_data.get("pantry", []),
            "likes": user_data.get("likes", []),
            "recommendations_per_meal": user_data.get("recommendations_per_meal", 3)
        }
        
        # Call the meal_plan endpoint
        response = requests.post(f"{API_URL}/meal_plan", json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to backend: {e}")
        return None

# --- USER AUTH ---
def login_page():
    st.title("🥗 Recipe Recommender Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username and password:
            st.session_state['user'] = {'name': username}
            st.success(f"Welcome back, {username}!")
            st.rerun()
        else:
            st.error("Please enter both username and password.")

# --- MAIN PAGE ---
def main_page():
    st.sidebar.title("👤 User Info")
    st.sidebar.write(f"Logged in as: {st.session_state['user']['name']}")
    if st.sidebar.button("Logout"):
        st.session_state.pop('user')
        st.rerun()

    st.title("🍽️ Smart Recipe Recommendations")

    # User Preferences Form
    with st.form("preferences_form"):
        st.subheader("Your Preferences")
        
        col1, col2 = st.columns(2)
        
        with col1:
            diet = st.selectbox("Diet Type", ["None", "vegetarian", "vegan", "gluten-free"])
            allergies = st.multiselect("Allergies", ["dairy", "nuts", "shellfish", "eggs", "soy"])
            likes = st.text_input("Foods you like (comma-separated)", "paneer, biryani")
        
        with col2:
            dislikes = st.text_input("Foods to avoid (comma-separated)", "")
            pantry = st.text_input("Pantry items (comma-separated)", "rice, tomatoes, onions")
            recommendations_per_meal = st.slider("Recipes per meal", 1, 5, 3)
        
        submitted = st.form_submit_button("Get Recommendations")
    
    if submitted:
        # Build user data
        user_data = {
            "user_id": st.session_state['user']['name'],
            "diet": None if diet == "None" else diet,
            "allergies": allergies,
            "dislikes": [d.strip() for d in dislikes.split(",") if d.strip()],
            "likes": [l.strip() for l in likes.split(",") if l.strip()],
            "pantry": [p.strip() for p in pantry.split(",") if p.strip()],
            "recommendations_per_meal": recommendations_per_meal,
            "time_budget": {
                "breakfast": 30,
                "lunch": 45,
                "dinner": 60
            }
        }
        
        with st.spinner("Fetching recommendations..."):
            recommendations = get_recommendations(user_data)
            if recommendations:
                show_recommendations(recommendations)

# --- DISPLAY RESULTS ---
def show_recommendations(recs):
    st.subheader("🍲 Your Personalized Meal Plan")
    
    # Display insights if available
    if recs.get("insights"):
        st.info(" | ".join(recs["insights"]))
    
    meal_plan = recs.get("meal_plan", {})
    
    # Create tabs for each meal
    tab1, tab2, tab3 = st.tabs(["🌅 Breakfast", "☀️ Lunch", "🌙 Dinner"])
    
    with tab1:
        display_meal_recipes(meal_plan.get("breakfast", []))
    
    with tab2:
        display_meal_recipes(meal_plan.get("lunch", []))
    
    with tab3:
        display_meal_recipes(meal_plan.get("dinner", []))
    
    # Display scoring info
    with st.expander("📊 Scoring Details"):
        st.json(recs.get("scoring_info", {}))

def display_meal_recipes(recipes):
    if not recipes:
        st.warning("No recipes found for this meal type")
        return
    
    for rec in recipes:
        with st.container(border=True):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"### {rec.get('title', 'Unknown')}")
                st.markdown(f"**Meal Type:** {rec.get('meal_type', 'N/A')}")
                st.markdown(f"**Prep Time:** {rec.get('time_display', 'N/A')}")
                st.markdown(f"**Diet:** {', '.join(rec.get('diet', []))}")
                
                if rec.get('ingredients'):
                    with st.expander("🥗 Ingredients"):
                        for ing in rec.get('ingredients', []):
                            st.write(f"- {ing}")
            
            with col2:
                st.metric("Score", f"{rec.get('score', 0):.2f}")
                st.metric("Pantry Matches", rec.get('pantry_matches', 0))
                
                if rec.get('link'):
                    st.link_button("View Recipe", rec.get('link'))

# --- ROUTING ---
if 'user' not in st.session_state:
    login_page()
else:
    main_page()