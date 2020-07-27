from datahandler import DataHandler
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Concatenate
import matplotlib.pyplot as plt
import numpy as np
import pickle

class ModelHandler():
    def __init__(self):
        plt.style.use('ggplot')
        size = 5000
        self.data = DataHandler(size, usePickle=False)

        #Training parameters
        self.epochs = 100
        self.batchSize = 32
        self.validationSplit = 0.1

        #Model parameters
        self.features = self.data.inputs.shape[1]
        self.styles = len(self.data.styles[0])
        self.drives = len(self.data.drivetrains[0])
        self.transmissions = len(self.data.speeds[0])

        self.activ = 'linear'

        #Compile parameters
        self.optimizer = 'nadam'
        self.loss = 'huber_loss'
        self.metrics = ['mean_absolute_error']

        self.makeModel()

        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)

        self.results = self.model.fit({'specs' : self.data.inputs, 'style' : self.data.styles, 'drive' : self.data.drivetrains, 'trans' : self.data.speeds}, self.data.targets, epochs=self.epochs, batch_size=self.batchSize, validation_split=self.validationSplit)

        self.graphTrainingResults()

        self.predicted = self.predictions()

        self.graphPredictions()

        self.model.save('model')


    def makeModel(self):
        specsIn = Input(shape=(self.features,), name='specs')
        styleIn = Input(shape=(self.data.styles.shape[1],), name='style')
        driveIn = Input(shape=(self.data.drivetrains.shape[1],), name='drive')
        transIn = Input(shape=(self.data.speeds.shape[1],), name='trans')

        x = Concatenate(axis=1)([specsIn, styleIn, driveIn, transIn])
        x = Dense(45, activation=self.activ)(x)
        x = Dense(45, activation=self.activ)(x)
        output = Dense(1, activation='linear')(x)

        self.model = keras.Model(inputs=[specsIn, styleIn, driveIn, transIn], outputs=output, name='model')
        self.model.summary()

    def predictions(self):
        predictSet = self.data.inputs
        styleSet = self.data.styles
        driveSet = self.data.drivetrains
        transSet = self.data.speeds

        bar = keras.utils.Progbar(len(predictSet), width=100)
        predicted = []

        for x in range(0, len(predictSet)):
            specs = predictSet[x].reshape(1,self.features,1)
            style = styleSet[x].reshape(1,self.styles,1)
            drive = driveSet[x].reshape(1,self.drives,1)
            trans = transSet[x].reshape(1,self.transmissions,1)

            predicted.append(self.model.predict({'specs' : specs, 'style' : style, 'drive' : drive, 'trans' : trans}).tolist()[0])

            bar.update(x + 1)

        predicted = self.data.msrpScaler.inverse_transform(np.array(predicted).reshape(-1,1)).tolist()

        return predicted

    def graphPredictions(self):
        target = np.linspace(self.data.minMsrp,self.data.maxMsrp,2)
        high = np.linspace(self.data.minMsrp * 1.2,self.data.maxMsrp * 1.2,2)
        low = np.linspace(self.data.minMsrp * 0.8,self.data.maxMsrp * 0.8,2)
        plt.figure(figsize=(16,9))
        graph = plt.subplot(111)
        plt.scatter(self.data.msrps, self.predicted, c='xkcd:coral', label='All Cars')
        plt.plot(target, target, '-', c='xkcd:pure blue', label='Target')
        plt.plot(target, high, '--', c='xkcd:water blue', label='20% High/Low')
        plt.plot(target, low, '--', c='xkcd:water blue')
        plt.title('Predicted MSRP vs. Actual MSRP')
        plt.xlabel('Actual MSRP')
        plt.ylabel('Predicted MSRP')
        plt.legend(loc='lower right')
        plt.savefig('predictions.png')

        pickle.dump(graph, open('pickles/graph.pkl', 'wb'))


    def graphTrainingResults(self):
        """
        Plots the loss and accuracy for the training and testing self.data
        """
        history = self.results.history
        plt.figure(figsize=(12,4))
        plt.plot(history['val_loss'])
        plt.plot(history['loss'])
        plt.legend(['val_loss', 'loss'])
        plt.title('Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.savefig("loss.png")

        plt.figure(figsize=(12,4))
        plt.plot(self.data.msrpScaler.inverse_transform(np.array(history['val_mean_absolute_error']).reshape(-1,1)).tolist())
        plt.plot(self.data.msrpScaler.inverse_transform(np.array(history['mean_absolute_error']).reshape(-1,1)).tolist())
        plt.legend(['val_mean_absolute_error', 'mean_absolute_error'])
        plt.title('Error')
        plt.xlabel('Epochs')
        plt.ylabel('Error')
        plt.savefig("error.png")

        plt.close('all')
