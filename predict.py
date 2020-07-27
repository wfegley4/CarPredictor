import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import pickle
import numpy as np
import time
import os
from tensorflow import keras

def makePrediction(specs, style, drive, trans):
    msrpScaler = pickle.load(open('pickles/msrp.pkl', 'rb'))
    hpScaler = pickle.load(open('pickles/hp.pkl', 'rb'))
    torqueScaler = pickle.load(open('pickles/torque.pkl', 'rb'))
    tireScaler = pickle.load(open('pickles/tire.pkl', 'rb'))
    mpgScaler = pickle.load(open('pickles/mpg.pkl', 'rb'))
    weightScaler = pickle.load(open('pickles/weight.pkl', 'rb'))

    bodyEncoder = pickle.load(open('pickles/body.pkl', 'rb'))
    driveEncoder = pickle.load(open('pickles/drive.pkl', 'rb'))
    transEncoder = pickle.load(open('pickles/trans.pkl', 'rb'))

    style = bodyEncoder.transform(np.array(style).reshape(1, -1)).toarray().reshape(1,-1,1)
    drive = driveEncoder.transform(np.array(drive).reshape(1, -1)).toarray().reshape(1,-1,1)
    trans = transEncoder.transform(np.array(trans).reshape(1, -1)).toarray().reshape(1,-1,1)

    specs[0] = hpScaler.transform(np.array(specs[0]).reshape(1, -1))
    specs[1] = torqueScaler.transform(np.array(specs[1]).reshape(1, -1))
    specs[2] = tireScaler.transform(np.array(specs[2]).reshape(1, -1))
    specs[3] = mpgScaler.transform(np.array(specs[3]).reshape(1, -1))
    specs[4] = weightScaler.transform(np.array(specs[4]).reshape(1, -1))

    specs = np.array(specs).reshape(1,-1,1)

    model = keras.models.load_model('model')

    prediction = model.predict({'specs' : specs, 'style' : style, 'drive' : drive, 'trans' : trans}).tolist()[0]

    prediction = msrpScaler.inverse_transform(np.array(prediction).reshape(-1,1)).tolist()[0][0]

    pickle.dump(int(prediction), open('pickles/prediction.pkl', 'wb'))

    prediction = '${:,}'.format(int(prediction))

    return prediction

def makeGraph(real):
    prediction = pickle.load(open('pickles/prediction.pkl', 'rb'))
    graph = pickle.load(open('pickles/graph.pkl', 'rb'))
    plt.scatter([real], [prediction], c='xkcd:bright blue', marker='*', s=150, label='Your Car')
    plt.legend()

    for filename in os.listdir('static/'):
        if filename.startswith('graph_'):
            os.remove('static/' + filename)

    name = f'graph_{str(time.time())}.png'
    path = f'static/{name}'

    plt.savefig(path)

    return name
