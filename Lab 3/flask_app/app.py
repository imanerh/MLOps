from flask import Flask, render_template, request, jsonify
import requests
import json
import numpy as np
import os 
from dotenv import load_dotenv

load_dotenv() 

app = Flask(__name__)

# Azure ML endpoint configuration
AZURE_ENDPOINT_URL = os.getenv('AZURE_ENDPOINT_URL')
AZURE_API_KEY = os.getenv('AZURE_API_KEY')

@app.route('/')
def index():
    """Render the home page with input form"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Store features for display
        features = {
            'fixed_acidity': request.form['fixed_acidity'],
            'volatile_acidity': request.form['volatile_acidity'],
            'citric_acid': request.form['citric_acid'],
            'residual_sugar': request.form['residual_sugar'],
            'chlorides': request.form['chlorides'],
            'free_sulfur_dioxide': request.form['free_sulfur_dioxide'],
            'total_sulfur_dioxide': request.form['total_sulfur_dioxide'],
            'density': request.form['density'],
            'pH': request.form['pH'],
            'sulphates': request.form['sulphates'],
            'alcohol': request.form['alcohol']
        }
        
        input_data = {
            "data": [[
                float(features['fixed_acidity']),
                float(features['volatile_acidity']),
                float(features['citric_acid']),
                float(features['residual_sugar']),
                float(features['chlorides']),
                float(features['free_sulfur_dioxide']),
                float(features['total_sulfur_dioxide']),
                float(features['density']),
                float(features['pH']),
                float(features['sulphates']),
                float(features['alcohol'])
            ]]
        }
        
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {AZURE_API_KEY}'
        }
        
        response = requests.post(
            AZURE_ENDPOINT_URL,
            data=json.dumps(input_data),
            headers=headers,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, str):
                try:
                    result = json.loads(result)
                except Exception:
                    pass
            predicted_quality = None
            
            # Try to extract prediction from various response formats
            try:
                # Method 1: Direct list
                if isinstance(result, list) and len(result) > 0:
                    predicted_quality = float(result[0])
                # Method 2: Dictionary with 'predictions' key
                elif isinstance(result, dict) and 'predictions' in result:
                    preds = result['predictions']
                    if isinstance(preds, list) and len(preds) > 0:
                        predicted_quality = float(preds[0])
                    else:
                        predicted_quality = float(preds)
                # Method 3: Dictionary with 'prediction' key
                elif isinstance(result, dict) and 'prediction' in result:
                    predicted_quality = float(result['prediction'])
                # Method 4: Dictionary with 'result' key
                elif isinstance(result, dict) and 'result' in result:
                    res = result['result']
                    if isinstance(res, list) and len(res) > 0:
                        predicted_quality = float(res[0])
                    else:
                        predicted_quality = float(res)
                # Method 5: Try to get first value from dictionary
                elif isinstance(result, dict) and len(result) > 0:
                    first_value = list(result.values())[0]
                    if isinstance(first_value, list) and len(first_value) > 0:
                        predicted_quality = float(first_value[0])
                    else:
                        predicted_quality = float(first_value)
                # Method 6: Direct numeric value
                else:
                    predicted_quality = float(result)
            except (ValueError, TypeError, IndexError, KeyError) as parse_error:
                return render_template('result.html', 
                    error=f"Could not parse prediction from response. Error: {str(parse_error)}. Response: {result}")
            
            # Ensure we got a valid prediction
            if predicted_quality is None:
                return render_template('result.html', 
                    error=f"Could not extract prediction value from response: {result}")
            
            # Determine quality description
            if predicted_quality <= 4:
                description = "Poor Quality"
            elif predicted_quality <= 6:
                description = "Average Quality"
            else:
                description = "Good Quality"
            
            return render_template(
                'result.html',
                features=features,
                prediction=round(predicted_quality, 2),
                description=description
            )
        else:
            return render_template('result.html', 
                error=f"API Error (Status {response.status_code}): {response.text}")
            
    except requests.exceptions.RequestException as e:
        return render_template('result.html', error=f"Network Error: {str(e)}")
    except json.JSONDecodeError as e:
        return render_template('result.html', error=f"JSON Parsing Error: {str(e)}")
    except KeyError as e:
        return render_template('result.html', error=f"Missing form field: {str(e)}")
    except ValueError as e:
        return render_template('result.html', error=f"Value Error: {str(e)}")
    except Exception as e:
        return render_template('result.html', error=f"Unexpected Error: {str(e)}")

def get_quality_description(quality_score):
    """Convert numeric quality to description"""
    quality_map = {
        3: "Poor Quality",
        4: "Below Average",
        5: "Average",
        6: "Good",
        7: "Very Good",
        8: "Excellent",
        9: "Outstanding"
    }
    return quality_map.get(int(round(quality_score)), "Unknown")

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'message': 'Flask app is running'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)