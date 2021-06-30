import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__,template_folder='template')
model = pickle.load(open('C:/Users/reach/OneDrive/Desktop/Projects/model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('C:/Users/reach/OneDrive/Desktop/Projects/templates/index.html')
    

@app.route('/predict',methods=['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    output = prediction[0]
    return render_template('C:/Users/reach/OneDrive/Desktop/Projects/templates/index.html', prediction_text='Amount of beer consumed would be: {}L'.format(output))

@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])
    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)
