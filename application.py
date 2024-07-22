from flask import Flask, render_template, request
from flask_cors import CORS, cross_origin
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)
cors = CORS(app)


with open('LinearRegressionModel.pkl', 'rb') as model_file:
    model = pickle.load(model_file)


df1 = pd.read_csv('Cleaned_data.csv') 


@app.route('/', methods=['GET', 'POST'])
def index():
    
    companies = sorted(df1['company'].unique())
    car_models = sorted(df1['name'].unique())
    years = sorted(df1['year'].unique(), reverse=True)
    fuel_types = df1['fuel_type'].unique()

    
    companies.insert(0, 'Select Company')


    return render_template('index.html', companies=companies, car_models=car_models, years=years, fuel_types=fuel_types)


@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
   
    company = request.form.get('company')
    car_model = request.form.get('car_model')
    year = request.form.get('year')
    fuel_type = request.form.get('fuel_type')
    driven = request.form.get('kilo_driven')

   
    prediction = model.predict(pd.DataFrame(columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'],
                                            data=np.array([car_model, company, year, driven, fuel_type]).reshape(1, 5)))
   
    print(prediction)

  
    return str(np.round(prediction[0], 2))


if __name__ == '__main__':
    app.run()
