from flask import Flask, request, jsonify
import joblib, numpy as np
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)
CORS(app)   # 🔥 VERY IMPORTANT (Fixes fetch error)

# Load model
obj = joblib.load("model/demand_pipeline.pkl")
pipeline = obj['pipeline']
le_prod = obj['le_prod']
le_reg = obj['le_reg']
FEATURES = obj['features']

@app.route('/')
def home():
    return {'message': 'Supply Chain Forecast API running'}

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    try:
        req = data.copy()

        # Date features
        req['Date'] = pd.to_datetime(req['Date'])
        req['Day'] = req['Date'].day
        req['Month'] = req['Date'].month
        req['Year'] = req['Date'].year
        req['DayOfWeek'] = req['Date'].weekday()

        # Encoding
        prod_enc = int(le_prod.transform([req['Product']])[0])
        reg_enc = int(le_reg.transform([req['Region']])[0])

        # Prepare input
        feats = [
            prod_enc, reg_enc,
            float(req['Price']), float(req['Promo']), int(req['Inventory']),
            req['Day'], req['Month'], req['Year'], req['DayOfWeek'],
            float(req['lag_1']), float(req['lag_7']), float(req['MA_7'])
        ]

        pred = pipeline.predict([feats])[0]
        return jsonify({'predicted_demand': int(pred)})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
