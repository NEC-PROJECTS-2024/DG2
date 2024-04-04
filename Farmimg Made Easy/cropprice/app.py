from flask import Flask, render_template, request
import joblib
import json
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier


app = Flask(__name__)

with open('df_dict.json', 'r') as json_file:
    state_mapping = json.load(json_file)

with open('Crop.json','r') as crop_json:
    crop_mapping=json.load(crop_json)
    
with open('Fertilizer.json','r') as fertilizer_json:
    fertilizer_mapping=json.load(fertilizer_json)

def predict_price(state, district, market, variety, rainfall):
    # Process the input data and make prediction
    input_data = [[state, district, market, variety, rainfall]]  # Input data for the model
    print(input_data)
    prediction = model.predict(input_data)[0]  # Use the model to make prediction

    return prediction

@app.route('/')
def navbar():
    return render_template('navbar.html')
@app.route('/crop')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        state = request.form['state']
        district = request.form['district']
        market = request.form['market']
        variety = request.form['variety']
        rainfall = request.form['Rainfall']

        # Convert rainfall to float if it's not None
        rainfall = float(rainfall) if rainfall is not None and rainfall != '' else 0.0

        state_number = state_mapping['state'].get(state, None)
        district_number = state_mapping['district'].get(district, None)
        market_number = state_mapping['market'].get(market, None)
        variety_number = state_mapping['variety'].get(variety, None)

        prediction = predict_price(state_number,district_number,market_number,variety_number, rainfall)


        return render_template('result.html', prediction=prediction)

    # Handle the case where the method is not POST
    return render_template('result.html', prediction=None)
@app.route('/rem')
def rem():
    return render_template('recommend.html')
@app.route('/rempredict',methods=['POST','GET'])
def rempredict():
    N=float(request.form.get('N'))
    P = float(request.form.get('P'))
    K = float(request.form.get('K'))
    temperature = float(request.form.get('temperature'))
    humidity = float(request.form.get('humidity'))
    ph= float(request.form.get('ph'))
    rainfall = float(request.form.get('rainfall'))
    data = pd.read_csv('Crop_recommendation.csv')
    data.dropna()
    ss = StandardScaler()
    y = data['label']
    x = data.drop('label', axis=1)
    ss.fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75)
    model = GaussianNB()
    model.fit(x_train, y_train)
    a=[N, P, K, temperature, humidity, ph, rainfall]
    result=model.predict([a])
    data=data[data.label !=result[0]]
    y = data['label']
    x = data.drop('label', axis=1)
    ss.fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75)
    model.fit(x_train, y_train)
    result1 = model.predict([a])
    data = data[data.label != result1[0]]
    y = data['label']
    x = data.drop('label', axis=1)
    ss.fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75)
    model.fit(x_train, y_train)
    result2 = model.predict([a])


    return render_template('result1.html',**locals())
@app.route('/pre')
def home3():
    return render_template('yield.html')

@app.route('/yieldpredict', methods=['POST','GET'])
def yieldpredict():
    # Get the input fields from the form
    crop = request.form['crop']
    state = request.form['state']
    cost_of_cultivation = float(request.form['cost_of_cultivation'])
    cost_of_production = float(request.form['cost_of_production'])

    crop_number = crop_mapping['state'].get(crop, None)
    state_number = crop_mapping['district'].get(state, None)
    # Use the loaded model to make predictions
    predicted_yield = model1.predict([[crop_number, state_number, cost_of_cultivation, cost_of_production]])

    return render_template('result3.html', crop=crop, state=state, predicted_yield=predicted_yield[0])
@app.route('/ferti')
def home4():
    return render_template('fertilizer.html')

@app.route('/ferpredict', methods=['POST','GET'])
def ferpredict():
    
    Temparature= float(request.form['Temparature'])
    Humidity= float(request.form['Humidity'])
    Moisture= float(request.form['Moisture'])
    Soil_Type = request.form['Soil Type']
    Crop_Type= request.form['Crop Type']
    Nitrogen = float(request.form['Nitrogen'])
    Potassium = float(request.form['Potassium'])
    Phosphorous= float(request.form['Phosphorous'])


    soil_number = fertilizer_mapping['Siol_type'].get(Soil_Type, None)
    crop_number = fertilizer_mapping['Crop_type'].get(Crop_Type, None)
    # Use the loaded model to make predictions
    Fertilizer = model2.predict([[soil_number, crop_number, Temparature,Humidity,Moisture,Nitrogen,Potassium,Phosphorous]])

    return render_template('result4.html', Soil_Type=Soil_Type,  Crop_Type= Crop_Type, Fertilizer=Fertilizer[0])
if __name__ == '__main__':
    model=joblib.load("crop_price.pkl")
    model1=joblib.load("mymodel_for_cropprice.pkl")
    model2=joblib.load("Fertilizer.pkl")
    app.run(host='0.0.0.0', port=5000, debug=True)

