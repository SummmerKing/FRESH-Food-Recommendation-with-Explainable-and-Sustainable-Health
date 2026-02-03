"""
Pantry Management System for Recipe Recommendation
Integrated with existing recommender system

Features:
- Add/Update/Remove pantry items with quantities
- Expiry date tracking
- Auto-suggestions for common ingredients
- Usage tracking when recipes are cooked
- Low stock alerts
- Pantry-based recipe recommendations
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from enum import Enum
import json
import os

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class UnitType(str, Enum):
    """Standard measurement units"""
    # Weight
    GRAMS = "g"
    KILOGRAMS = "kg"
    POUNDS = "lb"
    OUNCES = "oz"
    
    # Volume
    MILLILITERS = "ml"
    LITERS = "l"
    CUPS = "cup"
    TABLESPOONS = "tbsp"
    TEASPOONS = "tsp"
    
    # Count
    PIECES = "pieces"
    WHOLE = "whole"
    
    # Generic
    SOME = "some"

class PantryItem(BaseModel):
    """Individual pantry item"""
    item_id: Optional[str] = None
    name: str = Field(..., description="Name of the ingredient")
    canonical_name: str = Field(..., description="Standardized name for matching")
    quantity: float = Field(default=1.0, ge=0)
    unit: UnitType = Field(default=UnitType.SOME)
    category: Optional[str] = None
    expiry_date: Optional[datetime] = None
    added_date: datetime = Field(default_factory=datetime.now)
    last_updated: datetime = Field(default_factory=datetime.now)
    notes: Optional[str] = None
    low_stock_threshold: Optional[float] = None

class AddPantryItemRequest(BaseModel):
    """Request to add item to pantry"""
    user_id: str
    name: str
    quantity: float = 1.0
    unit: UnitType = UnitType.SOME
    category: Optional[str] = None
    expiry_date: Optional[str] = None
    notes: Optional[str] = None

class UpdatePantryItemRequest(BaseModel):
    """Request to update pantry item"""
    user_id: str
    item_id: str
    quantity: Optional[float] = None
    unit: Optional[UnitType] = None
    expiry_date: Optional[str] = None
    notes: Optional[str] = None

class RemovePantryItemRequest(BaseModel):
    """Request to remove item from pantry"""
    user_id: str
    item_id: str

class UsageRecord(BaseModel):
    """Track ingredient usage when recipe is cooked"""
    recipe_id: str
    recipe_name: str
    ingredients_used: List[str]
    cooked_date: datetime = Field(default_factory=datetime.now)

class RecordUsageRequest(BaseModel):
    """Request to record recipe cooking"""
    user_id: str
    recipe_id: str
    recipe_name: str
    ingredients_used: List[str]

# ============================================================================
# PANTRY DATABASE
# ============================================================================

PANTRY_DB: Dict[str, Dict[str, PantryItem]] = {}
USAGE_HISTORY: Dict[str, List[UsageRecord]] = {}
PANTRY_DB_PATH = "pantry_database.json"
USAGE_HISTORY_PATH = "usage_history.json"

# ============================================================================
# INGREDIENT STANDARDIZATION
# ============================================================================

INGREDIENT_ALIASES = {
    "tomato": ["tomatoes", "tomato", "cherry tomato", "roma tomato"],
    "onion": ["onions", "onion", "red onion", "white onion", "yellow onion"],
    "potato": ["potatoes", "potato", "russet potato", "sweet potato"],
    "carrot": ["carrots", "carrot", "baby carrot"],
    "garlic": ["garlic", "garlic clove", "garlic cloves", "minced garlic"],
    "ginger": ["ginger", "fresh ginger", "ginger root", "ground ginger"],
    "chicken": ["chicken", "chicken breast", "chicken thigh", "whole chicken"],
    "beef": ["beef", "ground beef", "beef steak", "beef roast"],
    "eggs": ["egg", "eggs", "whole egg"],
    "paneer": ["paneer", "cottage cheese"],
    "rice": ["rice", "white rice", "brown rice", "basmati rice", "jasmine rice"],
    "flour": ["flour", "all-purpose flour", "wheat flour", "maida"],
    "pasta": ["pasta", "spaghetti", "penne", "macaroni"],
    "milk": ["milk", "whole milk", "skim milk", "2% milk"],
    "butter": ["butter", "unsalted butter", "salted butter"],
    "cheese": ["cheese", "cheddar", "mozzarella", "parmesan"],
    "yogurt": ["yogurt", "yoghurt", "curd", "dahi"],
    "turmeric": ["turmeric", "turmeric powder", "haldi"],
    "cumin": ["cumin", "cumin seeds", "jeera"],
    "coriander": ["coriander", "coriander powder", "dhania"],
    "garam masala": ["garam masala", "garam masala powder"],
    "chili powder": ["chili powder", "red chili powder", "cayenne"],
    "oil": ["oil", "vegetable oil", "cooking oil", "olive oil"],
    "salt": ["salt", "table salt", "sea salt", "kosher salt"],
    "pepper": ["pepper", "black pepper", "ground pepper"],
}

INGREDIENT_CATEGORIES = {
    "vegetables": ["tomato", "onion", "potato", "carrot", "garlic", "ginger", 
                   "bell pepper", "spinach", "broccoli", "cauliflower", "peas"],
    "proteins": ["chicken", "beef", "pork", "lamb", "fish", "eggs", "paneer", "tofu"],
    "grains": ["rice", "flour", "pasta", "bread", "oats", "quinoa"],
    "dairy": ["milk", "butter", "cheese", "yogurt", "cream", "ghee"],
    "spices": ["turmeric", "cumin", "coriander", "garam masala", "chili powder",
               "cardamom", "cinnamon", "bay leaf", "mustard seeds"],
    "condiments": ["oil", "salt", "pepper", "soy sauce", "vinegar", "ketchup"],
    "pantry_staples": ["sugar", "flour", "rice", "pasta", "oil", "salt"],
}

def canonicalize_ingredient(name: str) -> str:
    """Standardize ingredient name"""
    name_lower = name.lower().strip()
    for canonical, aliases in INGREDIENT_ALIASES.items():
        if name_lower in aliases:
            return canonical
    return name_lower

def get_ingredient_category(canonical_name: str) -> Optional[str]:
    """Get category for ingredient"""
    for category, items in INGREDIENT_CATEGORIES.items():
        if canonical_name in items:
            return category
    return "other"

# ============================================================================
# PERSISTENCE FUNCTIONS
# ============================================================================

def load_pantry_data():
    """Load pantry data from disk"""
    global PANTRY_DB, USAGE_HISTORY
    
    if os.path.exists(PANTRY_DB_PATH):
        try:
            with open(PANTRY_DB_PATH, 'r') as f:
                data = json.load(f)
                PANTRY_DB = {
                    user_id: {
                        item_id: PantryItem(**item_data)
                        for item_id, item_data in items.items()
                    }
                    for user_id, items in data.items()
                }
            print(f"Loaded pantry data for {len(PANTRY_DB)} users")
        except Exception as e:
            print(f"Error loading pantry data: {e}")
    
    if os.path.exists(USAGE_HISTORY_PATH):
        try:
            with open(USAGE_HISTORY_PATH, 'r') as f:
                data = json.load(f)
                USAGE_HISTORY = {
                    user_id: [UsageRecord(**record) for record in records]
                    for user_id, records in data.items()
                }
            print(f"Loaded usage history for {len(USAGE_HISTORY)} users")
        except Exception as e:
            print(f"Error loading usage history: {e}")

def save_pantry_data():
    """Save pantry data to disk"""
    try:
        data = {
            user_id: {
                item_id: item.dict()
                for item_id, item in items.items()
            }
            for user_id, items in PANTRY_DB.items()
        }
        with open(PANTRY_DB_PATH, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        data = {
            user_id: [record.dict() for record in records]
            for user_id, records in USAGE_HISTORY.items()
        }
        with open(USAGE_HISTORY_PATH, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        return True
    except Exception as e:
        print(f"Error saving pantry data: {e}")
        return False

load_pantry_data()

# ============================================================================
# PANTRY MANAGEMENT FUNCTIONS
# ============================================================================

def generate_item_id(user_id: str, name: str) -> str:
    """Generate unique item ID"""
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    return f"{user_id}_{name[:10]}_{timestamp}"

def add_pantry_item(user_id: str, request: AddPantryItemRequest) -> PantryItem:
    """Add new item to user's pantry"""
    if user_id not in PANTRY_DB:
        PANTRY_DB[user_id] = {}
    
    canonical_name = canonicalize_ingredient(request.name)
    
    for item_id, item in PANTRY_DB[user_id].items():
        if item.canonical_name == canonical_name:
            item.quantity += request.quantity
            item.last_updated = datetime.now()
            if request.notes:
                item.notes = request.notes
            save_pantry_data()
            return item
    
    item_id = generate_item_id(user_id, canonical_name)
    category = request.category or get_ingredient_category(canonical_name)
    
    expiry_date = None
    if request.expiry_date:
        try:
            expiry_date = datetime.fromisoformat(request.expiry_date)
        except:
            pass
    
    item = PantryItem(
        item_id=item_id,
        name=request.name,
        canonical_name=canonical_name,
        quantity=request.quantity,
        unit=request.unit,
        category=category,
        expiry_date=expiry_date,
        notes=request.notes
    )
    
    PANTRY_DB[user_id][item_id] = item
    save_pantry_data()
    return item

def update_pantry_item(user_id: str, request: UpdatePantryItemRequest) -> PantryItem:
    """Update existing pantry item"""
    if user_id not in PANTRY_DB:
        raise HTTPException(status_code=404, detail="User pantry not found")
    
    if request.item_id not in PANTRY_DB[user_id]:
        raise HTTPException(status_code=404, detail="Item not found")
    
    item = PANTRY_DB[user_id][request.item_id]
    
    if request.quantity is not None:
        item.quantity = request.quantity
    if request.unit is not None:
        item.unit = request.unit
    if request.expiry_date is not None:
        try:
            item.expiry_date = datetime.fromisoformat(request.expiry_date)
        except:
            pass
    if request.notes is not None:
        item.notes = request.notes
    
    item.last_updated = datetime.now()
    save_pantry_data()
    return item

def remove_pantry_item(user_id: str, item_id: str) -> bool:
    """Remove item from pantry"""
    if user_id not in PANTRY_DB:
        raise HTTPException(status_code=404, detail="User pantry not found")
    
    if item_id not in PANTRY_DB[user_id]:
        raise HTTPException(status_code=404, detail="Item not found")
    
    del PANTRY_DB[user_id][item_id]
    save_pantry_data()
    return True

def get_user_pantry(user_id: str) -> List[PantryItem]:
    """Get all items in user's pantry"""
    if user_id not in PANTRY_DB:
        return []
    return list(PANTRY_DB[user_id].values())

def record_recipe_usage(user_id: str, request: RecordUsageRequest):
    """Record ingredients used when cooking a recipe"""
    if user_id not in USAGE_HISTORY:
        USAGE_HISTORY[user_id] = []
    
    record = UsageRecord(
        recipe_id=request.recipe_id,
        recipe_name=request.recipe_name,
        ingredients_used=request.ingredients_used
    )
    
    USAGE_HISTORY[user_id].append(record)
    save_pantry_data()
    return record

def get_expiring_soon(user_id: str, days: int = 7) -> List[PantryItem]:
    """Get items expiring within specified days"""
    if user_id not in PANTRY_DB:
        return []
    
    threshold_date = datetime.now() + timedelta(days=days)
    expiring = []
    
    for item in PANTRY_DB[user_id].values():
        if item.expiry_date and item.expiry_date <= threshold_date:
            expiring.append(item)
    
    return sorted(expiring, key=lambda x: x.expiry_date or datetime.max)

def get_low_stock_items(user_id: str) -> List[PantryItem]:
    """Get items that are low in stock"""
    if user_id not in PANTRY_DB:
        return []
    
    low_stock = []
    for item in PANTRY_DB[user_id].values():
        if item.low_stock_threshold and item.quantity <= item.low_stock_threshold:
            low_stock.append(item)
    
    return low_stock

# ============================================================================
# FASTAPI ENDPOINTS
# ============================================================================

app = FastAPI(title="Pantry Management API", version="1.0")

@app.post("/pantry/add")
def add_item(request: AddPantryItemRequest):
    """Add item to user's pantry"""
    try:
        item = add_pantry_item(request.user_id, request)
        return {
            "status": "success",
            "message": "Item added to pantry",
            "item": item.dict()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/pantry/update")
def update_item(request: UpdatePantryItemRequest):
    """Update pantry item"""
    try:
        item = update_pantry_item(request.user_id, request)
        return {
            "status": "success",
            "message": "Item updated",
            "item": item.dict()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/pantry/remove")
def remove_item(request: RemovePantryItemRequest):
    """Remove item from pantry"""
    try:
        success = remove_pantry_item(request.user_id, request.item_id)
        return {
            "status": "success",
            "message": "Item removed from pantry"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/pantry/{user_id}")
def get_pantry(user_id: str):
    """Get user's complete pantry"""
    items = get_user_pantry(user_id)
    return {
        "user_id": user_id,
        "total_items": len(items),
        "items": [item.dict() for item in items],
        "categories": _group_by_category(items)
    }

def _group_by_category(items: List[PantryItem]) -> Dict[str, int]:
    """Group items by category"""
    categories = {}
    for item in items:
        cat = item.category or "other"
        categories[cat] = categories.get(cat, 0) + 1
    return categories

@app.get("/pantry/{user_id}/expiring")
def get_expiring_items(user_id: str, days: int = 7):
    """Get items expiring soon"""
    items = get_expiring_soon(user_id, days)
    return {
        "user_id": user_id,
        "days_threshold": days,
        "expiring_count": len(items),
        "items": [item.dict() for item in items]
    }

@app.get("/pantry/{user_id}/low-stock")
def get_low_stock(user_id: str):
    """Get low stock items"""
    items = get_low_stock_items(user_id)
    return {
        "user_id": user_id,
        "low_stock_count": len(items),
        "items": [item.dict() for item in items]
    }

@app.post("/pantry/record-usage")
def record_usage(request: RecordUsageRequest):
    """Record recipe cooking and ingredient usage"""
    try:
        record = record_recipe_usage(request.user_id, request)
        return {
            "status": "success",
            "message": "Usage recorded",
            "record": record.dict()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/pantry/{user_id}/suggestions")
def get_pantry_suggestions(user_id: str):
    """Get ingredient suggestions based on user's pantry"""
    pantry_items = get_user_pantry(user_id)
    current_items = set(item.canonical_name for item in pantry_items)
    
    suggestions = []
    
    if any(item in current_items for item in ["cumin", "turmeric", "coriander"]):
        indian_staples = ["garam masala", "chili powder", "mustard seeds", "curry leaves"]
        suggestions.extend([s for s in indian_staples if s not in current_items])
    
    if any(item in current_items for item in ["chicken", "beef", "paneer"]):
        vegetables = ["onion", "tomato", "garlic", "ginger"]
        suggestions.extend([v for v in vegetables if v not in current_items])
    
    staples = ["salt", "pepper", "oil", "rice", "flour"]
    suggestions.extend([s for s in staples if s not in current_items])
    
    return {
        "user_id": user_id,
        "suggestions": list(set(suggestions[:10]))
    }

@app.get("/pantry/stats")
def get_pantry_stats():
    """Get global pantry statistics"""
    return {
        "total_users": len(PANTRY_DB),
        "total_items": sum(len(items) for items in PANTRY_DB.values()),
        "total_usage_records": sum(len(records) for records in USAGE_HISTORY.values()),
        "most_common_items": _get_most_common_items()
    }

def _get_most_common_items() -> Dict[str, int]:
    """Get most commonly stored items"""
    item_counts = {}
    for user_items in PANTRY_DB.values():
        for item in user_items.values():
            canonical = item.canonical_name
            item_counts[canonical] = item_counts.get(canonical, 0) + 1
    
    sorted_items = sorted(item_counts.items(), key=lambda x: x[1], reverse=True)
    return dict(sorted_items[:10])

if __name__ == "__main__":
    import uvicorn
    print("=" * 70)
    print("  PANTRY MANAGEMENT API")
    print("=" * 70)
    print(f"\n📦 Features:")
    print(f"   • Add/Update/Remove pantry items")
    print(f"   • Expiry date tracking")
    print(f"   • Low stock alerts")
    print(f"   • Usage history")
    print(f"   • Smart suggestions")
    print(f"\n🌐 Server: http://127.0.0.1:8002")
    print(f"📖 Docs: http://127.0.0.1:8002/docs")
    print("=" * 70)
    print("\n🚀 Starting server...\n")
    
    uvicorn.run(app, host="127.0.0.1", port=8002, log_level="info")