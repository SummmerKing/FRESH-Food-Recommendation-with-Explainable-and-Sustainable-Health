# streamlit_app.py
import streamlit as st
import requests
import json
from pathlib import Path
import time

# -------- CONFIG --------
API_URL = "http://127.0.0.1:8000"
RECOMMEND_ENDPOINT = f"{API_URL}/recommend"
HEALTH_ENDPOINT = f"{API_URL}/health"


st.set_page_config(page_title="FRESH — Meal Planner", layout="wide")

# -------- Helpers --------
def check_api_health():
    """Check if the API is running"""
    try:
        r = requests.get(HEALTH_ENDPOINT, timeout=5)
        return r.status_code == 200
    except:
        return False

def post_recommend(payload: dict):
    try:
        r = requests.post(RECOMMEND_ENDPOINT, json=payload, timeout=30)
        r.raise_for_status()
        return r.json(), None
    except Exception as e:
        return None, str(e)

def pretty_shopping_list(items):
    return "\n".join(f"- {i}" for i in items)

def format_meal_card(meal_name, meal_data):
    """Format a meal card with all the details"""
    if not meal_data:
        return f"**{meal_name.title()}**: No recommendation available"
    
    title = meal_data.get('title', 'Unknown')
    time_min = meal_data.get('time', 0)
    over_time = meal_data.get('over_time', False)
    used_ingredients = meal_data.get('used_ingredients', [])
    missing_ingredients = meal_data.get('missing_ingredients', [])
    
    time_indicator = "⏰" if not over_time else "⚠️"
    time_text = f"{time_min} min" + (" (over budget)" if over_time else "")
    
    card = f"""
    **{meal_name.title()}**: {title}  
    {time_indicator} {time_text}
    
    🏠 **From pantry**: {', '.join(used_ingredients) if used_ingredients else 'None'}
    🛒 **Need to buy**: {', '.join(missing_ingredients[:5]) if missing_ingredients else 'None'}
    """
    
    if len(missing_ingredients) > 5:
        card += f"\n    ... and {len(missing_ingredients) - 5} more items"
    
    return card

# -------- UI --------
st.title("🍽️ FRESH · Daily Meal Planner")
st.caption("AI-powered meal recommendations based on your pantry and preferences")

# Check API health
api_status = check_api_health()
if api_status:
    st.success("✅ API is running")
else:
    st.error("❌ API is not running. Please start the backend with: `uvicorn src.api.main:app --reload`")

# Left column: user inputs
with st.sidebar:
    st.header("👤 Your Profile")
    user_id = st.text_input("User ID", value="user_001", help="Unique identifier for your profile")
    
    st.subheader("🍽️ Dietary Preferences")
    diet = st.selectbox("Diet Type", options=["", "vegetarian", "vegan", "gluten-free", "keto", "paleo"], help="Select your dietary preference")
    
    st.subheader("⚠️ Restrictions")
    allergies = st.text_area("Allergies", value="", placeholder="nuts, dairy, shellfish", help="Comma-separated list of allergies")
    dislikes = st.text_area("Dislikes", value="", placeholder="mushrooms, olives", help="Comma-separated list of foods you don't like")
    
    st.subheader("⏰ Time Budget")
    col1, col2 = st.columns(2)
    with col1:
        b = st.number_input("Breakfast", value=30, min_value=5, max_value=120, help="Max cooking time for breakfast")
    with col2:
        l = st.number_input("Lunch", value=45, min_value=5, max_value=120, help="Max cooking time for lunch")
    d = st.number_input("Dinner", value=60, min_value=5, max_value=180, help="Max cooking time for dinner")
    
    st.subheader("🏠 Your Pantry")
    pantry = st.text_area("Available Ingredients", value="rice,eggs,milk,bread,tomatoes,onions,cheese", 
                         placeholder="rice,eggs,milk,bread,tomatoes,onions", 
                         help="Comma-separated list of ingredients you have")
    
    st.subheader("❤️ Favorites")
    likes = st.text_area("Favorite Foods", value="", placeholder="pasta,curry,stir-fry", 
                        help="Comma-separated list of foods you love")
    
    st.markdown("---")
    run_btn = st.button("🚀 Get My Meal Plan", type="primary", use_container_width=True)

# Main content area
if run_btn and api_status:
    # Build payload from UI inputs
    payload = {
        "user_id": user_id,
        "diet": diet or None,
        "allergies": [x.strip() for x in allergies.split(",") if x.strip()],
        "dislikes": [x.strip() for x in dislikes.split(",") if x.strip()],
        "time_budget": {"breakfast": int(b), "lunch": int(l), "dinner": int(d)},
        "pantry": [x.strip() for x in pantry.split(",") if x.strip()],
        "likes": [x.strip() for x in likes.split(",") if x.strip()]
    }
    
    with st.spinner("🤖 Generating your personalized meal plan..."):
        resp, err = post_recommend(payload)
    
    if err:
        st.error(f"❌ Error calling API: {err}")
    else:
        st.success("✅ Meal plan generated successfully!")
        
        # Display the meal plan
        st.header("📅 Your Daily Meal Plan")
        
        # Create columns for the three meals
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 🌅 Breakfast")
            breakfast = resp.get("plan", {}).get("breakfast")
            if breakfast:
                st.markdown(format_meal_card("breakfast", breakfast))
            else:
                st.warning("No breakfast recommendation available")
        
        with col2:
            st.markdown("### ☀️ Lunch")
            lunch = resp.get("plan", {}).get("lunch")
            if lunch:
                st.markdown(format_meal_card("lunch", lunch))
            else:
                st.warning("No lunch recommendation available")
        
        with col3:
            st.markdown("### 🌙 Dinner")
            dinner = resp.get("plan", {}).get("dinner")
            if dinner:
                st.markdown(format_meal_card("dinner", dinner))
            else:
                st.warning("No dinner recommendation available")
        
        # Shopping list section
        st.header("🛒 Shopping List")
        shopping_list = resp.get("shopping_list", [])
        if shopping_list:
            col_shopping, col_download = st.columns([3, 1])
            
            with col_shopping:
                st.write(f"**{len(shopping_list)} items needed:**")
                # Display shopping list in a nice format
                for i, item in enumerate(shopping_list, 1):
                    st.write(f"{i}. {item}")
            
            with col_download:
                st.download_button(
                    "📥 Download List",
                    "\n".join(shopping_list),
                    file_name=f"shopping_list_{user_id}_{time.strftime('%Y%m%d')}.txt",
                    mime="text/plain"
                )
        else:
            st.info("🎉 Great! You have all the ingredients you need!")
        
        # Show explanations
        st.header("💡 Why These Recommendations?")
        explanations = resp.get("explanation", {})
        for meal, explanation in explanations.items():
            st.write(f"**{meal.title()}**: {explanation}")
        
        # Debug section (collapsible)
        with st.expander("🔧 Debug Information"):
            st.json(resp)

elif run_btn and not api_status:
    st.error("❌ Cannot generate meal plan. Please start the API server first.")
    st.code("uvicorn src.api.main:app --reload", language="bash")

else:
    # Welcome screen
    st.header("👋 Welcome to FRESH!")
    st.markdown("""
    **Get personalized meal recommendations based on:**
    - 🏠 What's in your pantry
    - ⏰ Your available cooking time
    - 🍽️ Your dietary preferences
    - ❤️ Your favorite foods
    - ⚠️ Your allergies and dislikes
    
    **To get started:**
    1. Fill out your profile in the sidebar
    2. Click "Get My Meal Plan"
    3. Get your personalized recommendations!
    """)
    
    # Show sample data
    st.subheader("📊 Sample Data Preview")
    if st.checkbox("Show sample recipes"):
        try:
            with open("data/sample_recipes.json", "r") as f:
                sample_data = json.load(f)
            st.write(f"**{len(sample_data)} recipes available**")
            st.write("Sample recipes:")
            for i, recipe in enumerate(sample_data[:3]):
                st.write(f"{i+1}. **{recipe.get('title', 'Unknown')}** ({recipe.get('time_minutes', 'Unknown')} min)")
        except Exception as e:
            st.error(f"Could not load sample data: {e}")

# Footer
st.markdown("---")
st.caption("🍽️ FRESH - AI-Powered Meal Planning | Built with FastAPI + Streamlit")
