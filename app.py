from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load the trained model and label encoder
model_path = os.path.join('model', 'house_price_model.pkl')
encoder_path = os.path.join('model', 'label_encoder.pkl')

try:
    model = joblib.load(model_path)
    label_encoder = joblib.load(encoder_path)
    print("Model and encoder loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    label_encoder = None

# Get unique neighborhoods from the encoder
neighborhoods = list(label_encoder.classes_) if label_encoder else []

@app.route('/')
def home():
    """Render the home page"""
    return render_template('index.html', neighborhoods=neighborhoods)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Get input data from form
        overall_qual = int(request.form['overall_qual'])
        gr_liv_area = float(request.form['gr_liv_area'])
        total_bsmt_sf = float(request.form['total_bsmt_sf'])
        garage_cars = int(request.form['garage_cars'])
        year_built = int(request.form['year_built'])
        neighborhood = request.form['neighborhood']
        
        # Encode neighborhood
        neighborhood_encoded = label_encoder.transform([neighborhood])[0]
        
        # Prepare features array
        features = np.array([[
            overall_qual,
            gr_liv_area,
            total_bsmt_sf,
            garage_cars,
            year_built,
            neighborhood_encoded
        ]])
        
        # Make prediction
        prediction = model.predict(features)[0]
        
        # Format prediction
        predicted_price = f"${prediction:,.2f}"
        
        return jsonify({
            'success': True,
            'predicted_price': predicted_price,
            'inputs': {
                'Overall Quality': overall_qual,
                'Living Area (sq ft)': gr_liv_area,
                'Basement Area (sq ft)': total_bsmt_sf,
                'Garage Cars': garage_cars,
                'Year Built': year_built,
                'Neighborhood': neighborhood
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

if __name__ == '__main__':
    # Use environment variable for port (required for deployment)
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)