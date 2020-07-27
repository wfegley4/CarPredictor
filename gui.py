import pickle
from flask import Flask, render_template, request, url_for
from predict import makePrediction, makeGraph

styles = pickle.load(open('pickles/styles.pkl', 'rb'))
drives = pickle.load(open('pickles/drivetrains.pkl', 'rb'))
transmissions = pickle.load(open('pickles/transmissions.pkl', 'rb'))

gui = Flask(__name__)

@gui.route('/')
def examples(name=None):
    return render_template('examples.html', predictions1=url_for('static', filename='good3.png'), predictions2=url_for('static', filename='good4.png'), predictions3=url_for('static', filename='good5.png'))

@gui.route('/predict')
def predict(name=None):
    return render_template('predict.html', styles=styles, drives=drives, transmissions=transmissions)

@gui.route('/predict/results', methods=['POST'])
def results(name=None):
    form = request.form.to_dict()
    specs = [int(form['Horsepower']), int(form['Torque']), int(form['Tires']), int(form['MPG']), int(form['Weight'])]
    style = str(form['Body'])
    drive = str(form['Drive'])
    trans = str(form['Trans'])

    prediction = makePrediction(specs, style, drive, trans)

    return render_template('results.html', prediction=prediction)

@gui.route('/predict/results/graph', methods=['POST'])
def graph(name=None):
    real = request.form.to_dict()['Price']

    path = makeGraph(int(real))

    return render_template('graph.html', graph=path)
