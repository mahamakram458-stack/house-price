# California House Price Predictor

Problem: California housing data se ghar ki median price predict karna.

| Model | MAE | R² |
|-------|-----|----|
| Linear Regression | 0.533 | 0.576 |
| Random Forest | 0.328 | 0.805 |

Best model: Random Forest. Prices ki unit $100,000 hai.

## Run
pip install -r requirements.txt
streamlit run app.py