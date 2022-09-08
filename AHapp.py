
from flask import Flask, render_template, request
from AHhelpers.encoders import *
import numpy as np
import joblib


AHapp=Flask(__name__,template_folder="AhTemplates")

model = joblib.load('C:/Users/Abd AL-Rahman/Desktop/2nd graduation project/project dataset&notebook/model.ah',"readwrite")
scaler = joblib.load('C:/Users/Abd AL-Rahman/Desktop/2nd graduation project/project dataset&notebook/scaler.ah',"readwrite")

@AHapp.route('/')
def home():
    return render_template('ahpage.html')

@AHapp.route('/Indian_Used_Cars_Price_Prediction', methods=['POST'])
def Indian_Used_Cars_Price_Prediction():
    
    year = request.form["year"]
    kilometers= request.form["kilometers_driven"]
    mileage= request.form["mileage"]
    engine = request.form["engine"]
    power = request.form['power']
    seats  = request.form["seats"]
    name = carName_encoders[request.form["NameOfCar"]]
    location =Location_encoders[request.form["LocationOfCar"]]
    fuel_type = Fuel_encoders[request.form["FuelTypeOfCar"]]
    transmission = Transmission_encoders[request.form['TransmissionOfCar']]
    owner_type = OwnerType_encoders[request.form['OwnerType']]
    x=np.array([year,kilometers,mileage,engine,power,seats,name,location,fuel_type,transmission,owner_type])
    x2=scaler.transform([x])
    
    carprice=model.predict(x2)
    
    return render_template('ahpage.html',price_text= "Price of car is : {}$".format(carprice))
    
if __name__ == "__main__":
    
    
    AHapp.debug=True
    AHapp.run()