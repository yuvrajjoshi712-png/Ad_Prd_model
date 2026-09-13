import joblib
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the saved model
loaded_model = joblib.load('linear_regression_model.sav')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    # Assuming the input JSON will have keys 'TV', 'Radio', 'Newspaper'
    prediction_data = pd.DataFrame([data])
    predicted_sales = loaded_model.predict(prediction_data)
    return jsonify({'predicted_sales': predicted_sales[0]})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
