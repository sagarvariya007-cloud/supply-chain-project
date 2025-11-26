
# SupplyChainOptimization (Advanced Project)

This repository contains a full supply chain analysis project with synthetic dataset (3 years), training pipeline, API, and simple frontend.

## How to use

1. Install requirements:
```
pip install -r requirements.txt
```

2. Train model:
```
python src/train_model.py
```

This will create `model/demand_pipeline.pkl`.

3. Run API:
```
python src/forecast_api.py
```

4. Open `web/index.html` in a browser and call the API (running on localhost:5000).

## Contents
- data/sales_data.csv : synthetic dataset (~10950 rows)
- src/train_model.py : training script
- src/forecast_api.py : simple Flask API
- web/index.html : simple frontend
- model/ : models saved after training

