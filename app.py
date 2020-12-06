from flask import Flask, request, jsonify, render_template, redirect
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


@app.route("/")
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    
    features = [x for x in request.form.values()]
    final_features = [np.array(features)]
    result = model.predict(final_features)
    result = round(result[0], 2)

    return render_template('home.html', prediction_text= 'Price should be $ {}'.format(result))

if __name__ == '__main__':
    app.run(debug=True)