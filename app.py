from flask import Flask, request, render_template
import numpy as np
from src.main import Prediction

app = Flask(__name__)

#  24 input features
feature_names = ['Age', 'Gender', 'self_employed', 'family_history', 'work_interfere',
    'no_employees', 'remote_work', 'tech_company', 'benefits', 'care_options',
    'wellness_program', 'seek_help', 'anonymity', 'leave',
    'mental_health_consequence', 'phys_health_consequence', 'coworkers', 'supervisor',
    'mental_health_interview', 'phys_health_interview', 'mental_vs_physical', 'obs_consequence',
    'age_normalized', 'age_standardized'] 

@app.route('/')
def hello_world():
    return render_template('index.html', feature_names=feature_names)

@app.route('/predict', methods=["POST"])
def predict():
    if request.method == 'POST':
        try:
            # Get values from the form
            # input_values = [float(request.form.get(feature, 0)) for feature in feature_names]
            input_values = [float(request.form.get(feature, 0)) if request.form.get(feature) else 0 for feature in feature_names]
            final_input = [np.array(input_values)]
            
            # Make prediction
            prediction = prediction_obj.get_prediction(final_input)

            # Set the output variable to "Yes" or "No" based on prediction
            output = 'Yes' if prediction == 1 else 'No'

            return render_template('index.html', output=f'Predicted Treatment: {output}')
            # return render_template('index.html', feature_names=feature_names, input_values=request.form, output=f'Predicted Treatment: {prediction}')
        except Exception as e:
            return render_template('index.html', output=f'Error: {str(e)}')
    else:
        return render_template('index.html', output='Method Not Allowed'), 405

if __name__ == "__main__":
    prediction_obj = Prediction()
    app.run(debug=True)
