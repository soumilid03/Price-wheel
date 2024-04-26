from flask import Flask, render_template, request
from flask_cors import CORS, cross_origin
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)
cors = CORS(app)

# Load the pre-trained machine learning model
with open('LinearRegressionModel.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load the pre-processed dataset
df1 = pd.read_csv('Cleaned_data.csv')

# Define the route for the home page
@app.route('/', methods=['GET', 'POST'])
def index():
    # Get unique values for dropdown menus
    companies = sorted(df1['company'].unique())
    car_models = sorted(df1['name'].unique())
    years = sorted(df1['year'].unique(), reverse=True)
    fuel_types = df1['fuel_type'].unique()

    # Add default option to dropdown menus
    companies.insert(0, 'Select Company')

    # Render the HTML template with dropdown menu options
    return render_template('index.html', companies=companies, car_models=car_models, years=years, fuel_types=fuel_types)

# Define the route for the prediction endpoint
@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    # Get user input data from the form
    company = request.form.get('company')
    car_model = request.form.get('car_model')
    year = request.form.get('year')
    fuel_type = request.form.get('fuel_type')
    driven = request.form.get('kilo_driven')

    # Make a prediction using the pre-trained model
    prediction = model.predict(pd.DataFrame(columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'],
                                            data=np.array([car_model, company, year, driven, fuel_type]).reshape(1, 5)))
    # Print the prediction (for debugging purposes)
    print(prediction)

    # Return the prediction as a string
    return str(np.round(prediction[0], 2))

# Run the Flask web application
if __name__ == '__main__':
    app.run()
