from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form

    input_data = np.array([
        int(data['gender']),
        float(data['study_hours']),
        int(data['activities']),
        float(data['sleep_hours']),
        float(data['internet_usage']),
        int(data['attendance']),
        float(data['previous_score']),
        int(data['books_read']),
        int(data['tutoring']),
        int(data['writing_score']),
        int(data['reading_score']),
        int(data['mental_health']),
        int(data['physical_health'])
    ]).reshape(1, -1)

    input_data = scaler.transform(input_data)

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1] * 100  
    result = "Pass" if probability >= 50 else "Fail"

    return render_template('index.html', results=result, percentage=f"{probability:.2f}")

if __name__ == '__main__':
    app.run(debug=True)
