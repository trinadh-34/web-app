import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

ufos = pd.read_csv('data/ufos.csv')

#CONVERT the ufos data to a small dataframe with fresh titles.
ufos = pd.DataFrame({'Seconds': ufos['duration (seconds)'], 'Country': ufos['country'],'Latitude': ufos['latitude'],'Longitude': ufos['longitude']})

#reduce the amount of data by dropping any null values and only importing sightings between 1-60 seconds
ufos.dropna(inplace=True)
ufos = ufos[(ufos['Seconds'] >= 1) & (ufos['Seconds'] <= 60)]

#LabelEncoder library to convert text values for countries to a number
from sklearn.preprocessing import LabelEncoder 
ufos['Country'] = LabelEncoder().fit_transform(ufos['Country'])

from sklearn.model_selection import train_test_split
Selected_features = ['Seconds','Latitude','Longitude']
X = ufos[Selected_features]
y = ufos['Country']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(solver="lbfgs",max_iter=50000)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

model_filename = 'ufo-model.pkl'#using pickled model ufo-model.pkl
pickle.dump(model, open(model_filename,'wb'))
model = pickle.load(open('ufo-model.pkl','rb'))

@app.route("/predict", methods=["POST"])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = prediction[0]

    countries = ["Australia", "Canada", "Germany", "UK", "US"]

    return render_template(
        "index.html", prediction_text="Likely country: {}".format(countries[output])
    )


if __name__ == "__main__":
    app.run(debug=True)