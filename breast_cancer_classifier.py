from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

app = Flask(__name__)

# Load the breast cancer dataset
breast_cancer_data = load_breast_cancer()

# Create a DataFrame
df = pd.DataFrame(breast_cancer_data.data, columns=breast_cancer_data.feature_names)
df['label'] = breast_cancer_data.target

# Split the data
X = df.drop(columns='label', axis=1)
Y = df['label']
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Standardize features
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Define models
models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(random_state=2),
    'Gradient Boosting': GradientBoostingClassifier(random_state=2),
    'SVM': SVC(random_state=2),
    'XGBoost': XGBClassifier(random_state=2),
}

@app.route('/')
def index():
    feature_names = breast_cancer_data.feature_names
    return render_template('index.html', feature_names=feature_names)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        model_name = request.form['model']
        features = [float(request.form[feature]) for feature in breast_cancer_data.feature_names]
        scaled_features = scaler.transform([features])
        model = models[model_name]
        
        # Fit the model on the training data (if not already fitted)
        if not hasattr(model, 'classes_'):
            model.fit(x_train_scaled, y_train)
        
        prediction = model.predict(scaled_features)
        result = 'Malignant' if prediction[0] == 0 else 'Benign'
        return render_template('result.html', model=model_name, result=result)

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            # Save the uploaded file
            uploaded_file.save(uploaded_file.filename)
            
            # Load the CSV data
            uploaded_data = pd.read_csv(uploaded_file.filename)
            # Extract the features (assuming feature columns have the same names as in the training data)
            features = uploaded_data[breast_cancer_data.feature_names]

            # Scale the features
            scaled_features = scaler.transform(features)

            # Perform prediction using the selected model
            model_name = request.form['model']
            model = models[model_name]

            # Check if the model has been fitted, and fit it if not
            if not hasattr(model, 'classes_'):
                model.fit(x_train_scaled, y_train)

            predictions = model.predict(scaled_features)
            result = 'Malignant' if predictions[0] == 0 else 'Benign'
            return render_template('upload_result.html', predictions=result, model=model_name)
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)
