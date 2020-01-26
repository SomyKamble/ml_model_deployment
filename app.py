import pickle
import numpy as np
import flask
from flask import Flask,jsonify,render_template
from flask import request

app=Flask(__name__)

model=pickle.load(open('model.pkl','rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():

    '''for renderning the html input and output
    '''
    int_features=[int(x) for x in request.form.values()]
    final_features=[np.array(int_features)]
    prediction= model.predict(final_features)

    output= round(prediction[0],2)

    return render_template('index.html',prediction_text="flower is {}".format(output))

if __name__== "__main__":
    app.run(debug=True)
    
