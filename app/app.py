from flask import Flask, request, render_template
import os
import pandas as pd
import numpy as np
import pickle
app=Flask(__name__)

model=pickle.load(open(r'C:\Users\tarak\Downloads\ML Projects\Car Price Prediction\models\linear_regression_model.pkl','rb'))
df=pd.read_csv(r'C:\Users\tarak\Downloads\ML Projects\Car Price Prediction\data\preprocessed\cleaned_data.csv')


@app.route('/')
def index():
    companies=sorted(df['company'].unique())
    car_models=sorted(df['name'].unique())
    year=sorted(df['year'].unique(),reverse=True)
    fuel_type=sorted(df['fuel_type'].unique())
    companies.insert(0,'Select Company')
    car_models.insert(0,'Select Car Model')

    return render_template('index.html',companies=companies,car_models=car_models,years=year,fuel_types=fuel_type)  
    
@app.route('/predict',methods=['POST'])
def predict():
    company=request.form.get('company')
    car_models=request.form.get('car_models')
    year=int(request.form.get('year'))
    fuel_type=request.form.get('fuel_type')
    driven_km=int(request.form.get('kms_driven'))
    
    prediction=model.predict(pd.DataFrame(columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'],data=[[car_model,company,year,driven_km,fuel_type]]))
    return 'The predicted price of the car is Rs. {}'.format(np.round(prediction[0],2))

if __name__ == '__main__':
    app.run(debug=True)
