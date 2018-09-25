import os
import numpy as np
import logging
import pathlib
import errno
import pandas as pd
import csv
from tpot import TPOTRegressor
from sklearn.base import BaseEstimator, RegressorMixin
import matplotlib.pyplot as plt
from datetime import datetime
import bcolz
import shutil
import pickle
import time

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
        # cat_input = [Input(shape=(1,), dtype='int32', name=c) for c in self.categorical_columns]
        # all_layers = []
        
        # for i in range(len(cat_input)):
        #     emb = layers.Embedding(self.emb_szs[i][0], self.emb_szs[i][1])(cat_input[i])
        #     flat = layers.Flatten()(emb)
        #     all_layers.append(flat)

        # contInput = Input(shape=(len(self.non_categorical_columns),), dtype='float32', name='continuouse')
        # all_layers.append(contInput)
        # lay = layers.concatenate(all_layers, axis =-1)
        # #lay = BatchNormalization()(lay)
        # lay = Dense(128, activation='tanh', kernel_regularizer=regularizers.l2(self.l2_reg))(lay)
        # lay = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(self.l2_reg))(lay)
        # lay = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(self.l2_reg))(lay)
        # #lay = BatchNormalization()(lay)
        # answer = layers.Dense(KerasMultiInputFitter.N_CAT, activation='softmax')(lay)
        # inputs_all = cat_input
        # inputs_all.append(contInput)
        # self.model = Model(inputs_all, answer)
        # self.log.info(self.model.summary())
        # adam = Adam(lr=self.lr)
        # self.model.compile(loss="sparse_categorical_crossentropy", optimizer=adam,metrics=['accuracy'])
        pass

    
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
    DATA_ROOT = "data"
    label_col = 'energy(kWh/hh)'
    def __init__(self, logger = None):
        self.logger = logger
        self.current_directory = os.path.dirname(os.path.realpath(__file__))
        
        self.output_directory = os.path.join(self.current_directory, 'output')
        self.data_directory = os.path.join(self.current_directory, 'data')
        self.cat_pickle = os.path.join(self.data_directory, 'cat_map.pkl')
        self.halfhourly_dataset_feather = os.path.join(self.data_directory, 'half_hour_dataset.feather')
        self.bcolz_persist_location = os.path.join(self.data_directory, 'processed_main')

        self.categorical_cols = ['LCLid', 'stdorToU', 'Acorn_grouped', 'Acorn', 'summary', 'h_summary', 'icon', 'h_icon', 'h_precipType']
        self.numeric_cols = ['h_visibility','h_windBearing','h_temperature','h_dewPoint','h_pressure','h_apparentTemperature'
            ,'h_windSpeed','h_humidity','month_x','dayofweek','isHoliday','halfhourofday','temperatureMax'
            ,'windBearing','dewPoint','cloudCover','windSpeed','pressure','apparentTemperatureHigh','visibility','humidity','apparentTemperatureLow'
            ,'apparentTemperatureMax','uvIndex','temperatureLow','temperatureMin','temperatureHigh','apparentTemperatureMin'
            ,'moonPhase','temperatureMinTime_mod','temperatureMaxTime_mod','apparentTemperatureMinTime_mod','apparentTemperatureMaxTime_mod'
            ,'temperatureHighTime_mod','temperatureLowTime_mod','apparentTemperatureHighTime_mod','apparentTemperatureLowTime_mod'
            ,'sunsetTime_mod','sunriseTime_mod','uvIndexTime_mod']
        self.allcols = np.concatenate([self.categorical_cols ,  self.numeric_cols ]).tolist()
        self.allcols.append(self.label_col)
        self.cat_map= dict()
        for c in self.categorical_cols:
            self.cat_map[c]=dict()

    def getvalue_for_column(self, colname,value):
        if not value in self.cat_map[colname].keys():
            td = self.cat_map[colname]
            cnt = len(td.values())
            td[value]=cnt+1
            self.cat_map[colname]=td
        return self.cat_map[colname][value]

    @staticmethod
    def half_hour_of_day(datevalue):
        
        return 2*(datevalue.hour + (datevalue.minute /60.0))

    @staticmethod
    def minute_of_day(datevalue):
        
        return ((datevalue.hour*60) + datevalue.minute )

    @staticmethod
    def get_ts_int(datestring, formatstr='%Y-%m-%d %H:%M:%S.%f', stripChars=6):
        if type(datestring)==float:
            return np.nan
        if stripChars>0:
            tmp = datetime.strptime(datestring[:-stripChars], formatstr)
        else:
            tmp = datetime.strptime(datestring, formatstr)
        start = datetime(2000,1, 1 )
        delta = tmp-start
        return delta.days*24*3600 + delta.seconds
    
    @staticmethod
    def convert_time_cols(data_frame, column_name):
        data_frame[column_name+ "_mod"] = data_frame[column_name].apply(lambda x: smart_meter_data.minute_of_day(x, formatstr='%Y-%m-%d %H:%M:%S', stripChars=0) )

    def persist_bcolz(self, chunk_number, data, processed_data_dir):
        if chunk_number == 0:
            if os.path.isdir(processed_data_dir):
                shutil.rmtree(processed_data_dir)
            self.da = bcolz.carray(data, rootdir=processed_data_dir)
            #da.flush()
        else: 
            #da = bcolz.open(rootdir=processed_data_dir, mode='w')
            self.da.append(data)
            #da.flush()    

    def feature_eng(self,df):
        
        df = pd.merge(df, self.house, on='LCLid', how='left')
        df = pd.merge(df, self.hourly_weather, left_on='tstp', right_on = 'time',  how='left')
        df['date'] = df['tstp'].dt.date
        df['month']= df['tstp'].dt.month
        df['dayofweek']= df['tstp'].dt.weekday
        df['isHoliday']= df['tstp'].dt.date in self.holidays.index
        
        df['halfhourofday'] = df.tstp.apply(lambda x: self.half_hour_of_day(x) )
         
        df['ts_int'] = df.tstp.apply(lambda x: time.mktime(x.timetuple()))
        df = pd.merge(df, self.daily_weather, on='date', how='left')
        df.sort_values(by='ts_int', ascending=True, inplace=True)
        hcols =['h_visibility', 'h_windBearing', 'h_temperature','h_dewPoint', 'h_pressure', 'h_apparentTemperature', 'h_windSpeed',
        'h_precipType', 'h_icon', 'h_humidity', 'h_summary']
        df[hcols]= df[hcols].ffill()
        df[self.label_col] = pd.to_numeric(df[self.label_col], errors='coerce')
        for c in self.categorical_cols:  # convert cat columns to ints
            df[c] = df[c].apply(lambda x: self.getvalue_for_column(c,x))
        return df[self.allcols]

    def download_data(self):
        '''
            download data from csv files.
            and persist into bcloz dataset
        '''
        BLOCK_PATH = os.path.join(self.data_directory,'halfhourly_dataset') 
        BLOCKS = os.listdir(BLOCK_PATH)
        self.hourly_weather = pd.read_csv(os.path.join(self.data_directory, "weather_hourly_darksky.csv"), parse_dates=['time'])
        self.daily_weather  = pd.read_csv(os.path.join(self.data_directory, "weather_daily_darksky.csv"), 
            parse_dates=['temperatureMaxTime','temperatureMinTime', 'apparentTemperatureMinTime','apparentTemperatureHighTime',
            'time','sunsetTime', 'sunriseTime','temperatureHighTime','uvIndexTime','temperatureLowTime','apparentTemperatureMaxTime',
            'apparentTemperatureLowTime'])
        self.house          = pd.read_csv(os.path.join(self.data_directory, "informations_households.csv"))
        self.holidays       = pd.read_csv(os.path.join(self.data_directory, "uk_bank_holidays.csv"), parse_dates=['Bank holidays'],
                 index_col = 'Bank holidays')

        self.hourly_weather.columns = ["h_"+ c for c in self.hourly_weather.columns]
        self.hourly_weather.rename(columns={"h_time":"time"}, inplace=True)
      

        self.daily_weather['month']= pd.DatetimeIndex(self.daily_weather['time'].values).month
        # for c in ['temperatureMinTime', 'temperatureMaxTime', 'apparentTemperatureMinTime','apparentTemperatureMaxTime','temperatureHighTime','temperatureLowTime','apparentTemperatureHighTime','apparentTemperatureLowTime','sunsetTime','sunriseTime','uvIndexTime']:
        #     self.convert_time_cols(self.daily_weather, c)
        self.daily_weather['date'] = self.daily_weather['temperatureMinTime'].dt.date
        self.daily_weather.fillna(method='ffill', inplace=True)    
        for i, block in enumerate (BLOCKS):
            self.logger.debug (f"starting  block {i}")
            ddf = pd.read_csv(os.path.join(BLOCK_PATH,block), parse_dates=['tstp'])
            df = self.feature_eng(ddf)
            self.persist_bcolz(i,df[self.allcols].values.astype(np.float32), self.bcolz_persist_location)
            del ddf
            del df
            self.logger.debug (f"completed block {i}")    
        self.da.flush()
        # save the string mapping dictionary    
        with open(self.cat_pickle, "wb") as output_file:
            pickle.dump(self.cat_map, output_file)
        
    def get_persisted_bcolz(self):
        '''
        return persisted bcloz set
        '''
        self.da = bcolz.open(rootdir=self.bcolz_persist_location, mode='r')
        self.logger.info(f'There are : {len(self.da)} records found!')
        return self.da

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
        #call first time only
        smart_meter.download_data()


    except Exception as ex:
        log.exception(ex)
    finally:
        log.handlers = []