from flask import Flask, request, url_for, redirect, render_template
import pickle
import numpy as np
import pandas as pd
import warnings


warnings.filterwarnings("ignore", category=FutureWarning)

app = Flask(__name__, template_folder='./templates', static_folder='./static')

# Load the model
with open("knn_tuned.pkl", 'rb') as file:
    model = pickle.load(file)

# Load the scaler
with open("scaler.pkl", 'rb') as file:
    scaler = pickle.load(file)

# Define the feature names
feature_names = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']

@app.route('/')
def hello_world():
    return render_template('home.html')

@app.route('/predict', methods=['POST','GET'])
def predict():
    values = list(request.form.values())
    # age, sex, children, smoker, region should be integers
    # bmi should be a float

    features = [
        int(values[0]),  # age
        int(values[1]),  # sex
        float(values[2]),  # bmi
        int(values[3]),  # children
        int(values[4]),  # smoker
        int(values[5])   # region
    ]
    print(features)
    
    # Create a DataFrame with the input features and matching feature names
    final_df = pd.DataFrame([features], columns=feature_names)
    print(final_df)
    
    # Scale the input features
    final_scaled = scaler.transform(final_df)
    
    pred = model.predict(final_scaled)[0]
    print(pred)
    if pred < 0:
        return render_template('op.html', pred='Error calculating Amount!')
    else:
        return render_template('op.html', pred='Expected amount is {0:.3f}'.format(pred))

if __name__ == '__main__':
    app.run(debug=False)