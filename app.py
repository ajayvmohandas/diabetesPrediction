import pickle
from flask import Flask, request, app, render_template
import numpy as np
import pandas as pd

app=Flask(__name__)

model=pickle.load(open("diabetesPrediction.pkl", "rb"))

@app.route("/")
def home():
    return render_template("API.html")

@app.route("/predict_api",methods=["POST"])
def predict():
    features=[float(x) for x in request.form.values()]
    final_features=[np.array(features)]
    prediction=model.predict(final_features)
    output=prediction[0]
    if output==1:
        return render_template("API.html",prediction_text="Oops! You have diabetes.")

    else:
        return render_template("API.html",prediction_text="Great! You don't have diabetes")

if __name__=="__main__":
    app.run(debug=True)

