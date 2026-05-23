from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# I-load ang trained AI model na ginawa natin kanina
try:
    model = joblib.load('agritrack_model.pkl')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")

# Ito ang URL endpoint na tatawagin ng PHP mo
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Kunin ang JSON data na ipinadala ng PHP
        data = request.json
        
        # I-convert sa DataFrame (format na nauunawaan ng AI)
        df = pd.DataFrame([data])
        
        # Gumawa ng prediction gamit ang model
        prediction = model.predict(df)
        
        # I-return ang sagot bilang JSON pabalik sa PHP
        return jsonify({'predicted_sacks': round(prediction[0], 2)})
    
    except Exception as e:
        # Kung may error, ibalik ang error message
        return jsonify({'error': str(e)})

# I-run ang API server
if __name__ == '__main__':
    # Ang port na gagamitin natin ay 5000
    app.run(host='0.0.0.0', port=5000)