import pandas
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


#DataFrame indexing: [row, column] (swapped from csv)
class DataHandler():
    def __init__(self, size, usePickle=True):
        self.specs = ['MSRP', 'SAE Net Horsepower @ RPM', 'SAE Net Torque @ RPM', 'Rear Tire Size', 'Base Curb Weight (lbs)', 'Fuel Economy Est-Combined (MPG)', 'Body Style', 'Drivetrain', 'Trans Type']
        self.csv = 'short_specs.csv'

        if(usePickle):
            print('Using pickle')
            self.data = pandas.read_pickle('pickles/specs.pkl')

        else:
            print('Not using pickle')
            self.data = pandas.read_csv(self.csv, low_memory=False)
            self.data = self.data.convert_dtypes()
            self.data = self.data.set_index(self.data.iloc[:,0])
            self.data = self.data.drop(columns=[' '])
            self.data = self.data.swapaxes(0,1)
            self.data = self.data.loc[:,self.specs]
            self.data = self.data.mask(self.data == '- TBD -')
            self.data = self.data.dropna(axis='index')

            self.rows = self.data.index
            self.msrp()
            self.data = self.data.query('MSRP < 100000')
            self.data = self.data.iloc[0:(0 + size), :]
            self.data.to_pickle('pickles/specs.pkl')

        self.rows = self.data.index

        print(f'Cars in this set: {len(self.data.index)}')

        self.prepare()

        self.minMsrp = min(self.msrps)
        self.maxMsrp = max(self.msrps)

        self.inputs = []
        self.styles = []
        self.drivetrains = []
        self.speeds = []
        self.targets = []

        self.normalize()


    def prepare(self):
        self.msrps = self.msrp()
        self.hps = self.hp()
        self.torques = self.torque()
        self.tires = self.tire()
        self.mpgs = self.mpg()
        self.weights = self.weight()
        self.bodies = self.body()
        self.drives = self.drive()
        self.gears = self.transmission()

    def normalize(self):
        self.msrpScaler = MinMaxScaler()
        self.hpScaler = MinMaxScaler()
        self.torqueScaler = MinMaxScaler()
        self.tireScaler = MinMaxScaler()
        self.mpgScaler = MinMaxScaler()
        self.weightScaler = MinMaxScaler()
        self.bodyEncoder = OneHotEncoder()
        self.driveEncoder = OneHotEncoder()
        self.transEncoder = OneHotEncoder()

        normalizedMsrps = self.msrpScaler.fit_transform(self.msrps)
        normalizedHps = self.hpScaler.fit_transform(self.hps)
        normalizedTorques = self.torqueScaler.fit_transform(self.torques)
        normalizedTires = self.tireScaler.fit_transform(self.tires)
        normalizedMpgs = self.mpgScaler.fit_transform(self.mpgs)
        normalizedWeights = self.weightScaler.fit_transform(self.weights)

        encodedBodies = self.bodyEncoder.fit_transform(self.bodies).toarray()
        encodedDrives = self.driveEncoder.fit_transform(self.drives).toarray()
        encodedTrans = self.transEncoder.fit_transform(self.gears).toarray()

        inputs = np.array([normalizedHps, normalizedTorques, normalizedTires, normalizedMpgs, normalizedWeights])
        inputs = inputs.swapaxes(0,1)

        self.styles = encodedBodies
        self.drivetrains = encodedDrives
        self.speeds = encodedTrans


        self.inputs = inputs
        self.targets = normalizedMsrps

        bodyTypes = []
        for i in self.bodyEncoder.get_feature_names().squeeze().tolist():
            bodyTypes.append(i[3:])

        driveTypes = []
        for i in self.driveEncoder.get_feature_names().squeeze().tolist():
            driveTypes.append(i[3:])

        transTypes = []
        for i in self.transEncoder.get_feature_names().squeeze().tolist():
            transTypes.append(i[3:])

        pickle.dump(bodyTypes, open('pickles/styles.pkl', 'wb'))
        pickle.dump(driveTypes, open('pickles/drivetrains.pkl', 'wb'))
        pickle.dump(transTypes, open('pickles/transmissions.pkl', 'wb'))

        pickle.dump(self.msrpScaler, open('pickles/msrp.pkl', 'wb'))
        pickle.dump(self.hpScaler, open('pickles/hp.pkl', 'wb'))
        pickle.dump(self.torqueScaler, open('pickles/torque.pkl', 'wb'))
        pickle.dump(self.tireScaler, open('pickles/tire.pkl', 'wb'))
        pickle.dump(self.mpgScaler, open('pickles/mpg.pkl', 'wb'))
        pickle.dump(self.weightScaler, open('pickles/weight.pkl', 'wb'))
        pickle.dump(self.bodyEncoder, open('pickles/body.pkl', 'wb'))
        pickle.dump(self.driveEncoder, open('pickles/drive.pkl', 'wb'))
        pickle.dump(self.transEncoder, open('pickles/trans.pkl', 'wb'))

    def msrp(self):
        for row in self.rows:
                msrp = self.data.loc[row, 'MSRP']
                try:
                    msrp = msrp.strip('$').split(',')
                    msrp = msrp[0] + msrp[1]
                except AttributeError:
                    continue

                self.data.loc[row, 'MSRP'] = int(msrp)

        return self.data.loc[:, 'MSRP'].astype('int').to_numpy().reshape(-1,1)

    def hp(self):
        for row in self.rows:
                hp = self.data.loc[row, 'SAE Net Horsepower @ RPM']
                hp = hp.split(' @')[0].split(' ')[0]

                self.data.loc[row, 'SAE Net Horsepower @ RPM'] = int(hp)

        return self.data.loc[:, 'SAE Net Horsepower @ RPM'].astype('int').to_numpy().reshape(-1,1)

    def torque(self):
        for row in self.rows:
            torque = self.data.loc[row, 'SAE Net Torque @ RPM']
            torque = torque.split(' @')[0].split(' ')[0]

            self.data.loc[row, 'SAE Net Torque @ RPM'] = int(torque)

        return self.data.loc[:, 'SAE Net Torque @ RPM'].astype('int').to_numpy().reshape(-1,1)

    def tire(self):
        for row in self.rows:
            tire = self.data.loc[row, 'Rear Tire Size']
            tire = tire.split('/')[0].strip('PLTS').split('-')[0]

            self.data.loc[row, 'Rear Tire Size'] = int(tire)

        return self.data.loc[:, 'Rear Tire Size'].astype('int').to_numpy().reshape(-1,1)

    def mpg(self):
        for row in self.rows:
            mpg = self.data.loc[row, 'Fuel Economy Est-Combined (MPG)']
            mpg = mpg.split(' ')[0]

            self.data.loc[row, 'Fuel Economy Est-Combined (MPG)'] = int(mpg)

        return self.data.loc[:, 'Fuel Economy Est-Combined (MPG)'].astype('int').to_numpy().reshape(-1,1)

    def weight(self):
        for row in self.rows:
            weight = self.data.loc[row, 'Base Curb Weight (lbs)']
            weight = weight.split('-')[0].strip().split(' ')[0]

            self.data.loc[row, 'Base Curb Weight (lbs)'] = int(weight)

        return self.data.loc[:, 'Base Curb Weight (lbs)'].astype('int').to_numpy().reshape(-1,1)

    def body(self):
        return self.data.loc[:, 'Body Style'].astype('str').to_numpy().reshape(-1,1)

    def drive(self):
        return self.data.loc[:, 'Drivetrain'].astype('str').to_numpy().reshape(-1,1)

    def transmission(self):
        return self.data.loc[:, 'Trans Type'].astype('str').to_numpy().reshape(-1,1)

    def report(self):
        #Get report of Body Style and EPA Classification columns
        styles = {}
        classes = {}
        for row in self.rows:
            x = self.data.loc[row, 'Drivetrain']
            y = self.data.loc[row, 'Trans Type']
            if x not in styles.keys():
                styles.update({x : 1})
            else:
                styles.update({x : styles.get(x) + 1})

            if y not in classes.keys():
                classes.update({y : 1})
            else:
                classes.update({y : classes.get(y) + 1})

        print(f'\nDrivetrains: {len(styles)}')
        for i in styles:
            print(f'{i}: {styles.get(i)}')
        print(f'\nTransmissions: {len(classes)}')
        for i in classes:
            print(f'{i}: {classes.get(i)}')
