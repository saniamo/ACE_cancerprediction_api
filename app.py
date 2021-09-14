from flask import Flask, request, redirect, url_for, flash, jsonify
import numpy as np
import pickle
import json

app = Flask(__name__)


@app.route('/api/',methods=['POST'])
def predict():    
    data = request.get_json()
    prediction = np.array2string(model.predict(data))

    return jsonify(prediction)
        


if __name__ == '__main__':
    model = pickle.load(open('./final_model.sav', 'rb'))
    app.run(debug=True, host='localhost:5000/')
