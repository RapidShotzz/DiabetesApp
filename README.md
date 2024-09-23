This repository contains a web-based Diabetes Prediction App that uses a Support Vector Machine (SVM) machine learning model to predict whether a person is diabetic based on key medical variables such as BMI, age, glucose levels, and more. The model was fine-tuned using GridSearchCV for hyperparameter optimisation and achieved an accuracy of 90%.

Features:

Machine Learning Model: Support Vector Machine (SVM) for prediction.
Hyperparameter Tuning: GridSearchCV used to optimize the model parameters.
Web Interface: Simple and intuitive web interface built using Flask, HTML, CSS, and JavaScript.
Medical Input Variables: Uses features such as BMI, age, glucose levels, and others to make predictions.
High Accuracy: The model achieves a prediction accuracy of 90%.

Table of Contents:

- Installation
- Usage
- Model Overview
- Technologies Used

Installation:
To get started with this project, follow these steps:

Clone the repository:
git clone https://github.com/RapidShotzz/DiabetesPredictionApp.git
cd DiabetesPredictionApp

Set up a virtual environment (optional but recommended):
python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

Install the dependencies:
pip install -r requirements.txt

Run the Flask app:

flask run
The app will be available at http://127.0.0.1:5000.

Usage:

Once the app is running, open your browser and navigate to http://127.0.0.1:5000. You can input values for various medical parameters like BMI, age, glucose levels, etc., and click on the Predict button to get a prediction on whether the person is diabetic or not.

Model Overview:

Algorithm: Support Vector Machine (SVM) was used due to its effectiveness in binary classification problems like diabetes prediction.
Hyperparameter Tuning: The model's hyperparameters were tuned using GridSearchCV, leading to a significant improvement in model accuracy.
Accuracy: The SVM model achieved an accuracy of 90%, making it a reliable tool for diabetes prediction.

Technologies Used:

Python: For building the SVM model and handling the backend.
Flask: As a lightweight framework for the web application.
HTML/CSS: To design the front end.
JavaScript: For interactivity and form validation.
scikit-learn: For implementing the SVM model and using GridSearchCV for hyperparameter tuning.
pandas: For data manipulation and analysis.

