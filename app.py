
import uvicorn
from fastapi import FastAPI, File, UploadFile
import google.generativeai as genai
import pandas as pd
import tempfile
import json
from fastapi.middleware.cors import CORSMiddleware
import math
from typing import Any, Dict, List, Union

# Configure the API key
api_key = 'AIzaSyBXC_o3DTYBLbVBOLoQHCOQtXVA_DCqp-o'
genai.configure(api_key=api_key)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def sanitize_float_values(obj: Any) -> Any:
    """Sanitize float values to ensure JSON compatibility."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return str(obj)
        return obj
    elif isinstance(obj, dict):
        return {k: sanitize_float_values(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_float_values(item) for item in obj]
    return obj

def upload_image(image_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        temp_file.write(image_bytes)
        temp_file_path = temp_file.name
    return genai.upload_file(temp_file_path)

def get_nutrition_and_ingredients(sample_file):
    model = genai.GenerativeModel(model_name="gemini-1.5-flash-latest")
    response = model.generate_content([
        sample_file,
        """Extract all the details regarding nutrients and ingredients shown in the given image and give me response in the format of valid JSON string Object where Nutrient:(given nutrient) and Ingredients:(given ingredients) and only these two things and only JSON response ,no other thing. like as follows:
        {
            {
            "Nutrients": {
                "Energy": "457.7 kcal",
                "Protein": "22.20 g",
                "Total Carbohydrate": "53.3 g",
                "Total Sugars": "2.3 g",
                "Added Sugars": "0.0 g",
                "Dietary Fibre": "12.3 g",
                "Total Fat": "17.3 g",
                "Saturated Fat": "6.1 g",
                "Trans Fat": "0.1 g",
                "Sodium": "571.3 mg"
            },
            "Ingredients": [
                "Split Green Gram (Moong Dal)",
                "Vegetable Oil (Cotton Seed, Groundnut and Rice Bran)",
                "Iodized Salt"
              ]
            }
        }"""
    ])

    response_text = response.text.strip()
    try:
        if response_text.startswith("```json") and response_text.endswith("```"):
            response_text = response_text[7:-3].strip()
        parsed_response = json.loads(response_text)
        # Sanitize the parsed response
        return sanitize_float_values(parsed_response)
    except json.JSONDecodeError:
        print("Failed to parse response text as JSON.")
        return {}

def parse_response_to_dataframe(response_data: Dict) -> pd.DataFrame:
    try:
        nutrients = response_data.get("Nutrients", {})
        ingredients = response_data.get("Ingredients", [])

        # Convert any potential problematic float values in nutrients
        sanitized_nutrients = {k: str(v) if isinstance(v, (float, int)) else v 
                             for k, v in nutrients.items()}

        # Create DataFrames
        nutrients_df = pd.DataFrame([sanitized_nutrients]).T.reset_index()
        nutrients_df.columns = ["Nutrient", "Value"]
        
        ingredients_df = pd.DataFrame({"Ingredient": ingredients})
        
        # Combine nutrients and ingredients
        combined_df = pd.concat(
            [nutrients_df, ingredients_df],
            keys=["Nutrients", "Ingredients"],
            axis=0,
            ignore_index=True
        )
        
        return combined_df
    except Exception as e:
        print(f"Error in parse_response_to_dataframe: {e}")
        return pd.DataFrame()

@app.get('/')
def index():
    return {'message': 'Image API - Send an image to /predict'}

@app.post('/predict')
async def predict_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        sample_file = upload_image(contents)

        if not sample_file:
            return {"error": "Failed to upload the image"}
        
        response_data = get_nutrition_and_ingredients(sample_file)
        if not response_data:
            return {"error": "Unable to extract product details."}
            
        df = parse_response_to_dataframe(response_data)
        if df.empty:
            return {
                "filename": file.filename,
                "message": "No relevant product details found."
            }
            
        # Convert DataFrame to dict and sanitize any problematic values
        product_details = sanitize_float_values(df.to_dict(orient='records'))
        
        return {
            "filename": file.filename,
            "message": "Product details extracted successfully.",
            "product_details": product_details
        }

    except Exception as e:
        return {"error": str(e)}

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)

