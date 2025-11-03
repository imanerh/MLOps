from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
import uvicorn
import pickle
import numpy as np

# Initialize FastAPI app
app = FastAPI(title="Furniture Price Prediction API")

# Setup Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

# Define the input schema using Pydantic
class FurnitureFeatures(BaseModel):
    """
    Input features for furniture price prediction based on IKEA furniture dataset
    """
    category: int = Field(..., ge=0, le=16, description="Furniture category (0-16)")
    sellable_online: int = Field(..., ge=0, le=1, description="Can be sold online (0=No, 1=Yes)")
    other_colors: int = Field(..., ge=0, le=1, description="Available in other colors (0=No, 1=Yes)")
    depth: float = Field(..., gt=0, description="Depth in cm")
    height: float = Field(..., gt=0, description="Height in cm")
    width: float = Field(..., gt=0, description="Width in cm")


# Category mapping
CATEGORIES = {
    0: "Bar furniture",
    1: "Beds",
    2: "Bookcases & shelving units",
    3: "Cabinets & cupboards",
    4: "Cafe furniture",
    5: "Chairs",
    6: "Chests of drawers & drawer units",
    7: "Children's furniture",
    8: "Nursery furniture",
    9: "Outdoor furniture",
    10: "Room dividers",
    11: "Sideboards, buffets & console tables",
    12: "Sofas & armchairs",
    13: "Tables & desks",
    14: "Trolleys",
    15: "TV & media furniture",
    16: "Wardrobes"
}

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """
    Home page with prediction form
    """
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "categories": CATEGORIES}
    )

@app.post("/predict_form", response_class=HTMLResponse)
async def predict_with_form(
    request: Request,
    category: int = Form(...),
    sellable_online: int = Form(...),
    other_colors: int = Form(...),
    depth: float = Form(...),
    height: float = Form(...),
    width: float = Form(...)
):
    """
    Make prediction and return HTML response
    """
    try:
        # Convert features to array format
        input_data = np.array([[
            category,
            sellable_online,
            other_colors,
            depth,
            height,
            width
        ]])
        
        prediction = model.predict(input_data)[0]
        
        return templates.TemplateResponse(
            "result.html",
            {
                "request": request,
                "prediction": round(prediction, 2),
                "category_name": CATEGORIES.get(category, "Unknown"),
                "sellable_online": "Yes" if sellable_online == 1 else "No",
                "other_colors": "Yes" if other_colors == 1 else "No",
                "depth": depth,
                "height": height,
                "width": width
            }
        )
    except Exception as e:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "categories": CATEGORIES,
                "error": str(e)
            }
        )

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=True)