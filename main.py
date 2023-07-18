import pandas as pd
from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)
data = pd.read_csv('cleaned_data.csv')
pipe = pickle.load(open('LinearModel.pkl', 'rb'))


@app.route('/')
def index():
    locations = sorted(data['site_location'].unique())
    return render_template('index.html', locations=locations)


@app.route('/predict', methods=['POST'])
def predict():
    location = request.form.get('location')
    bhk = request.form.get('bhk')
    bath = request.form.get('bath')
    balcony = request.form.get('balcony')
    sqft = request.form.get('sqft')

    # print(location, bhk, bath, balcony, sqft)

    input = pd.DataFrame([[sqft,bath,balcony,location,bhk]],columns=['total_sqft','bath','balcony','site_location','bhk'])
    prediction = pipe.predict(input)[0] * 1e5


    return str(np.round(prediction,2))


if __name__ == "__main__":
    app.run(debug=True)