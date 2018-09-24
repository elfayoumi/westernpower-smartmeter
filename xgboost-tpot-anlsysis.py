import arcpy, os
import numpy as np
import logging

import pathlib
import errno
import pandas as pd
import csv
from tpot import TPOTRegressor
from sklearn.base import BaseEstimator, RegressorMixin

class WesternPowerSmartMeter(BaseEstimator, RegressorMixin):
    
    DAYS_AHEAD = 5
    LAGGED_DAYS = 0
    VALID_DAYS = 20
    def __init__(self, logger = None, verbose=1, batch_size=128, epochs=200,lr = 0.001, 
                sequence_length= 20,l2_reg= 0.01):
        """
            Called when initializing the classifier
        """
        self.log = logger
        self.current_directory = os.path.dirname(os.path.realpath(__file__))
        self.output_directory = os.path.join(self.current_directory,'data')
        self.pickle_filename = os.path.join(self.output_directory, f"all_data_view.pkl")
        self.lr = lr
        self.name = 'keras_multiInput'
        self.data_dir = os.path.join(self.current_directory, 'data')
        self.batch_size = batch_size
        self.epochs = epochs
        self.verbose = verbose
        self.window_size = sequence_length
        self.l2_reg = l2_reg

    def model_setup(self):
        cat_input = [Input(shape=(1,), dtype='int32', name=c) for c in self.categorical_columns]
        all_layers = []
        
        for i in range(len(cat_input)):
            emb = layers.Embedding(self.emb_szs[i][0], self.emb_szs[i][1])(cat_input[i])
            flat = layers.Flatten()(emb)
            all_layers.append(flat)

        contInput = Input(shape=(len(self.non_categorical_columns),), dtype='float32', name='continuouse')
        all_layers.append(contInput)
        lay = layers.concatenate(all_layers, axis =-1)
        #lay = BatchNormalization()(lay)
        lay = Dense(128, activation='tanh', kernel_regularizer=regularizers.l2(self.l2_reg))(lay)
        lay = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(self.l2_reg))(lay)
        lay = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(self.l2_reg))(lay)
        #lay = BatchNormalization()(lay)
        answer = layers.Dense(KerasMultiInputFitter.N_CAT, activation='softmax')(lay)
        inputs_all = cat_input
        inputs_all.append(contInput)
        self.model = Model(inputs_all, answer)
        self.log.info(self.model.summary())
        adam = Adam(lr=self.lr)
        self.model.compile(loss="sparse_categorical_crossentropy", optimizer=adam,metrics=['accuracy'])

    
    def fit(self, X, y=None):
        """
        This should fit classifier. All the "work" should be done here.

        Note: assert is not a good choice here and you should rather
        use try/except blog with exceptions. This is just for short syntax.
        """
        try:
            #X = pd.DataFrame(X, columns=['StatusXO', 'StatusBS', 'day', 'BP', 'Value', 'Low', 'High'])
            self.cat_sz = [(c, len(set(X[c]))+1) for c in self.categorical_columns]
            self.emb_szs = [(c, min(50, (c+1)//2)) for _,c in self.cat_sz]

            self.model_setup()
            train_values = {c:X[c] for c in self.categorical_columns}
            train_values['continuouse'] = X[self.non_categorical_columns]
            
            self.model.fit(train_values, y, epochs=self.epochs, batch_size = self.batch_size,
                                        #validation_data=(val_values, self.y_vali
                                        verbose=self.verbose)

        except Exception as ex:
            self.log.exception(ex)
        return self

    def _meaning(self, x):
        # returns True/False according to fitted classifier
        # notice underscore on the beginning
        return( True )

    def predict(self, X, y=None):
        try:
            test_values = {c:X[c] for c in self.categorical_columns}
            test_values['continuouse'] = X[self.non_categorical_columns]
            test_predict = self.model.predict(test_values)

            return np.argmax( test_predict, axis=1) 
        except AttributeError:
            raise RuntimeError("You must train classifer before predicting data!")

        return None

    def score(self, X, y=None):
        # counts number of values bigger than mean
        test_values = {c:X[c] for c in self.categorical_columns}
        test_values['continuouse'] = X[self.non_categorical_columns]
        t = self.model.evaluate(x=test_values,y=y, batch_size=self.batch_size)
        self.log.info(t)
        p = sum(np.equal(y.values[:].reshape(-1),  self.predict(X).reshape(-1)))/len(y)
        return(p) 

class smart_meter_data:
    DATA_ROOT = "smart-meters-in-london"
    def __init__(self, logger = None):
        self.logger = logger
        self.current_directory = os.path.dirname(os.path.realpath(__file__))
        self.output_directory = os.path.join(self.current_directory, 'output')
        self.data_directory = os.path.join(self.current_directory, 'data')
        self.halfhourly_dataset_feather = os.path.join(self.data_directory, 'half_hour_dataset.feather')

    def download_data(self):
        '''
            download data from csv files.
        '''
        data_root = os.path.join(self.current_directory, self.DATA_ROOT)
        self.block_path = os.path.join(data_root, 'halfhourly_dataset')
        self.blocks = os.listdir(self.block_path)
        df = pd.concat([pd.read_csv(os.path.join(self.block_path,block), parse_dates=['tstp'], index_col = ['tstp','LCLid']) 
                for block in self.blocks])
        df.to_feather(self.halfhourly_dataset_feather)
        hourly_weather = pd.read_csv(os.path.join(data_root, "weather_hourly_darksky.csv"))
        daily_weather  = pd.read_csv(os.path.join(data_root, "weather_daily_darksky.csv"))
        house          = pd.read_csv(os.path.join(data_root, "informations_households.csv"))
        holidays       = pd.read_csv(os.path.join(data_root, "uk_bank_holidays.csv"))



if __name__ == "__main__":
    base_directory = os.path.dirname(os.path.realpath(__file__))
    log = logging.getLogger("smart_meter")
    log.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(os.path.join(base_directory, "logging/smart_meter.log"))
    fh.setLevel(logging.INFO)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    log.addHandler(fh)
    log.addHandler(ch)

    try:
        smart_meter = smart_meter_data(logger=log)
        smart_meter.download_data()


    except Exception as ex:
        log.exception(ex)
    finally:
        log.handlers = []