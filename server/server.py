from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# loading the saved model and scaler

model = pickle.load(open('model.pkl','rb'))
scaler = pickle.load(open('scaler.pkl','rb'))

@app.route('/')
def home():
    return render_template('app.html')

@app.route('/predict',methods=['POST'])
def predict():
    data = request.json
    features = np.array([
        data['Pregnancies'],
        data['Glucose'],
        data['BloodPressure'],
        data['SkinThickness'],
        data['Insulin'],
        data['BMI'],
        data['DiabetesPedigreeFunction'],
        data['Age']
    ]).reshape(1, -1)

    # scaling input features

    scaled_features = scaler.transform(features)

    prediction = model.predict(scaled_features)

    return jsonify({'prediction': int(prediction[0])})

if __name__ == "__main__":
    app.run(debug=True)


