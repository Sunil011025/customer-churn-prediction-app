from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
import os

app = Flask(__name__)

print("🚀 App is starting...")


model_data = pickle.load(open('customer_churn_model.pkl', 'rb'))

model = model_data["model"]   # ✅ actual ML model
feature_order = model_data["features_names"]  # ✅ column order

encoders = pickle.load(open('encoders.pkl', 'rb'))


def prepare_input_dataframe(raw_data):
    data = raw_data.copy()
    numeric_fields = ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen']

    for field in numeric_fields:
        value = data.get(field)
        if value is None or str(value).strip() == "":
            data[field] = 0
        else:
            data[field] = float(value)

    input_df = pd.DataFrame([data])

    for col in feature_order:
        if col not in input_df:
            input_df[col] = 0

    input_df = input_df[feature_order]

    for column, encoder in encoders.items():
        if column in input_df:
            input_df[column] = encoder.transform(input_df[column])

    return input_df


def run_prediction(input_data):
    input_df = prepare_input_dataframe(input_data)
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)

    result = "Customer will CHURN ❌" if prediction[0] == 1 else "Customer will STAY ✅"
    confidence = round(max(probability[0]) * 100, 2)
    #print(confidence+" "+result)
    return result, f"Confidence: {confidence}%"


# =========================
# Home Route
# =========================
@app.route('/')
def home():
    return render_template('index.html')


# =========================
# API Prediction Route
# =========================
@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        input_data = request.get_json(force=True)
        prediction_text, prob_text = run_prediction(input_data)
        return jsonify(prediction_text=prediction_text, prob_text=prob_text)
    except Exception as e:
        return jsonify(error=str(e)), 400


# =========================
# Prediction Route
# =========================
@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.form.to_dict()
        prediction_text, prob_text = run_prediction(input_data)
        return render_template(
            'index.html',
            prediction_text=prediction_text,
            prob_text=prob_text
        )
    except Exception as e:
        return render_template(
            'index.html',
            prediction_text=f"Error: {str(e)}"
        )


# =========================
# Run App
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # for Render
    app.run(host="0.0.0.0", port=port, debug=True)