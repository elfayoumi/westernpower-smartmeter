from keras.models import Sequential, Model
from keras import layers
from keras import Input
import pandas as pd
import numpy as np
from random import sample, randint
import os, datetime
from pathlib import Path
import itertools
from sklearn.preprocessing import StandardScaler
import logging
import  mysql.connector
from datetime import timedelta
import datetime
from keras.optimizers import Adam, Adadelta, SGD
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from keras.layers.core import Dense, Activation, Dropout
from pandas_ml import ConfusionMatrix
# import BatchNormalization
from keras.layers.normalization import BatchNormalization
from fastai.structured import proc_df
from keras import regularizers
from sklearn.base import BaseEstimator, ClassifierMixin

class KerasMultiInput:
    N_CAT = 3
    DAYS_AHEAD = 5
    LAGGED_DAYS = 0
    VALID_DAYS = 20
    def __enter__(self):
        return (self)
    def info(self, txt):
        if self.log:
            self.log.info(txt)
        else:
            print(txt)

    def debug(self, txt):
        if self.log:
            self.log.debug(txt)
        else:
            print(txt)


    def excption(self, ex):
        if self.log:
            self.log.exception(ex)
        else:
            print(ex)

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def best_weights(self, name=None):
        if not name:
            name = self.name
        return os.path.join(self.data_dir, name + '_weights.hdf5')

    def __init__(self, verbose=1, batch_size=128, epochs=200,lr = 0.001, sequence_length= 20,l2_reg= 0.01, logger = None):
        self.log = logger
        self.current_directory = os.path.dirname(os.path.realpath(__file__))
        self.output_directory = os.path.join(self.current_directory,'data')
        today_date = datetime.datetime.today().strftime('%Y%m%d')
        self.pickle_filename1 = os.path.join(self.output_directory, f"{today_date}_view.pkl")
        self.pickle_filename2 = os.path.join(self.output_directory, f"{today_date}_xjo.pkl")
        self.fig_name = os.path.join( self.output_directory, 'histo.png')
        self.feather_name = os.path.join( self.output_directory, f"{today_date}_xjo.feather")
        self.lr = lr
        self.name = 'keras_multiInput'
        self.data_dir = os.path.join(self.current_directory, 'data')
        self.batch_size = batch_size
        self.epochs = epochs
        self.verbose = verbose
        self.ts_scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
        self.window_size = sequence_length
        self.l2_reg = l2_reg
    def download_data(self):
        try:
            my_file = Path(self.pickle_filename1)
            if my_file.is_file():
                # file exists
                index_prices = pd.read_pickle(self.pickle_filename1)                
                
            else:
                cnx = mysql.connector.connect(user='ibrahim_', password='^a9_&kalman_', host='athe2014.il7ad.org', database="mystocks")
                query =  """SELECT ip.`Timestamp`,  StatusXO, StatusBS,BP, `Open`, High, Low, `Close`  FROM mystocks.IndexBulishPercent  ibp 
                            left join mystocks.IndexPrice ip on ibp.`Date` = ip.`Date` 
                            where ibp.indexId = 10 and ip.indexid = 3 order by ip.`Date`"""
                index_prices = pd.read_sql(query, con=cnx) #
                index_prices['Timestamp'] =  pd.to_datetime(index_prices['Timestamp'])
                index_prices = index_prices.set_index('Timestamp')
                index_prices.to_pickle(self.pickle_filename1)
                #drop duplicate
                index_prices.drop_duplicates(inplace =True)
                index_prices = index_prices[~index_prices.index.duplicated(keep='last')]
                cnx.close()
        
            my_file = Path(self.pickle_filename2)
            if my_file.is_file():
                # file exists
                xjo = pd.read_pickle(self.pickle_filename2)                
                
            else:
                cnx = mysql.connector.connect(user='ibrahim_', password='^a9_&kalman_', host='athe2014.il7ad.org', database="mystocks")
                query = "select * from `mystocks`.`IndexPrice` where IndexId = 3 order by `Date`"
                xjo = pd.read_sql(query, con=cnx) #
                xjo['Date'] =  pd.to_datetime(xjo['Date'], format='%Y%m%d')
                xjo = xjo.set_index('Date')
                xjo.to_pickle(self.pickle_filename2)
                cnx.close()
            

            df = index_prices.fillna(method='ffill').reset_index()
            df.to_feather(self.feather_name)
            index_prices['day'] = index_prices.index.weekday
            self.debug("\n{0}".format(index_prices.head()))
            self.debug(f'There are: {len(index_prices)} rows')
            self.xjo = xjo
            return index_prices
        except Exception as ex:
            self.excption(ex)
            raise ex


    def find_categorical(self, X):
        
        # Get list of categorical column names
        self.categorical_columns = ['StatusXO', 'StatusBS', 'day']
        self.debug(f"Columns found: {self.categorical_columns}")
        # Get list of non-categorical column names
        self.non_categorical_columns = ['BP', 'Value', 'Low', 'High']   
        #self.sequence_columns = ['Value']

    def create_dataset(self, df):
        df['Value'] = (df["High"] + df["Low"])/2.0
        y = df['Value']
        ahead = y.shift(self.DAYS_AHEAD)
        new_y = 100.0*(y/ahead-1)
        quantiles = new_y.quantile(np.linspace(0,1,self.N_CAT+1))
        self.info(f'The quantiles for fittings: {quantiles.values}')
        quantiles.iloc[-1] += 1.0
        quantiles.iloc[0] -= 1.0
        y = []
        for i in range(self.DAYS_AHEAD, len(new_y)):
            t = new_y.iloc[i]
            for j in range(1, self.N_CAT+1):
                
                if t >= quantiles.iloc[j-1] and t < quantiles.iloc[j]:
                    y.append(j-1)
        y = pd.DataFrame(y)
        y['Timestmap'] = df.index[:-self.DAYS_AHEAD]
        y = y.set_index('Timestmap')
        y.columns = ['Value']
        y = y['Value']

        for v in self.categorical_columns:
            df[v] = df[v].astype('category').cat.as_ordered()
        for v in self.non_categorical_columns:
            df[v] = df[v].astype('float32')

        df = df.fillna(method='ffill') #forward fill
        df = df.fillna(method='bfill') #backward fill
        #self.sequence_columns = ['Value']
        

        self.cat_sz = [(c, len(df[c].cat.categories)+1) for c in self.categorical_columns]
        self.emb_szs = [(c, min(50, (c+1)//2)) for _,c in self.cat_sz]
        df, _, self.nas, self.mapper = proc_df(df,  do_scale=True)
        df = df[self.categorical_columns + self.non_categorical_columns ]
        self.info(df.dtypes)
        self.ts = self.window_transform_series(self.window_size,df.index)
        #df[:max(y.index)].boxplot(by=y.values[:], column=['BP'])
        #plt.show()
        return (df,y.astype('int') )
    def window_transform_series(self,window_size, index):
        t = (self.xjo['High'] + self.xjo['Low'])/2.0
        t =self.ts_scaler.fit_transform(t.values.reshape(-1,1))
        v = pd.DataFrame(t[:,0])
        v['Timestamp'] = self.xjo.index
        v = v.set_index('Timestamp')
        
        # x values ends 1 before the end
        X = []
        
        # Create window_size columns of shiffted x values
        for id in index:
           
            r = v[:id].values[-window_size:]
            X.append(r)

        # reshape each
        X =np.asarray(X)
       
        return X
    
    def model_setup(self, X,y):
        valid_start_day = max(X.index) - timedelta(days=KerasMultiInput.VALID_DAYS)
        self.val_idx = X.index[np.flatnonzero( (X.index <= max(y.index)) & (X.index >= valid_start_day))]
        self.X_train = X[min(X.index):valid_start_day]
        self.ts_train = self.ts[0:len(self.X_train), :]
        self.y_train = y[self.X_train.index]
        self.X_valid = X[min(self.val_idx):max(self.val_idx)]
        self.ts_valid = self.ts[len(self.X_train):(len(self.X_train)+len(self.X_valid))]
        self.y_valid = y[min(self.val_idx):max(self.val_idx)]
        self.X_test = X[max(y.index):]
        self.ts_test = self.ts[-len(self.X_test):]

        cat_input = [Input(shape=(1,), dtype='int32', name=c) for c in self.categorical_columns]
        seq_input = Input(shape=( None, 1), dtype='float32', name='rnn_input')
        all_layers = []
        for i in range(len(cat_input)):
            emb = layers.Embedding(self.emb_szs[i][0], self.emb_szs[i][1])(cat_input[i])
            flat = layers.Flatten()(emb)
            all_layers.append(flat)

        contInput = Input(shape=(len(self.non_categorical_columns),), dtype='float32', name='continuouse')
        #continuousDense = layers.Dense(32)(contInput)

        #concatenated_embdding = layers.concatenate(cat_emb, axis=-1)
        #categDense = layers.Flatten()(concatenated_embdding)
        seq_lay = layers.GRU(16,return_sequences=False, activation='tanh', 
                kernel_regularizer=regularizers.l2(self.l2_reg))(seq_input)
        #seq_lay = layers.GRU(16,return_sequences=False, kernel_regularizer=regularizers.l2(self.l2_reg))(seq_lay)

        #concatenated = layers.concatenate([categDense, continuousDense, seq_lay1], axis =-1)
        all_layers.append(contInput)
        all_layers.append(seq_lay)
        lay = layers.concatenate(all_layers, axis =-1)
        #lay = BatchNormalization()(lay)
        # lay = Dense(64, kernel_regularizer=regularizers.l2(self.l2_reg))(lay)
        # lay = Dense(128, activation='tanh', kernel_regularizer=regularizers.l2(self.l2_reg))(lay)

        # #lay = Dropout(0.8)(lay)
        # lay = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(self.l2_reg))(lay)
        lay = Dense(32, activation='tanh', kernel_regularizer=regularizers.l2(self.l2_reg))(lay)
        #lay = BatchNormalization()(lay)
        answer = layers.Dense(KerasMultiInput.N_CAT, activation='softmax')(lay)
        inputs_all = cat_input
        inputs_all.append(contInput)
        inputs_all.append(seq_input)
        self.model = Model(inputs_all, answer)
        self.info(self.model.summary())
        adam = Adam(lr=self.lr)
        self.model.compile(loss="sparse_categorical_crossentropy", optimizer=adam,metrics=['accuracy'])

    def fit_data(self, show_figures = True):
        checkpointer = ModelCheckpoint(filepath=self.best_weights(self.name), verbose=self.verbose, save_best_only=True)
        train_values = {c:self.X_train[c] for c in self.categorical_columns}
        train_values['continuouse'] = self.X_train[self.non_categorical_columns]
        train_values['rnn_input'] = self.ts_train
        val_values = {c:self.X_valid[c] for c in self.categorical_columns}
        val_values['continuouse'] = self.X_valid[self.non_categorical_columns]
        val_values['rnn_input'] = self.ts_valid
        
        history = self.model.fit(train_values, self.y_train, epochs=self.epochs, batch_size = self.batch_size,
                                     #validation_data=(val_values, self.y_valid), 
                                     verbose=self.verbose, shuffle=True,
                                     validation_split=0.2,
                                     callbacks=[checkpointer])
        if show_figures:
            fig, ax = plt.subplots(figsize=(10, 5))
            # plot history
            ax.plot(history.history['loss'], label='train')
            ax.plot(history.history['val_loss'], label='test')
            ax.legend()
            figure_name = os.path.join(self.output_directory, self.name + "_history.png")

            plt.savefig(figure_name)
            plt.show()
        self.model.load_weights(self.best_weights(self.name))
        test_values = {c:self.X_test[c] for c in self.categorical_columns}
        test_values['continuouse'] = self.X_test[self.non_categorical_columns]
        test_values['rnn_input'] = self.ts_test
        test_predict = self.model.predict(test_values)
        p  = self.model.predict(val_values)
        t = np.sum(np.product(self.y_valid, np.log(p)))
        self.info(f"Validation Score: {t}")
        val_predict = np.argmax( p, axis=1)
        

        confusion_matrix = ConfusionMatrix(self.y_valid.values[:], val_predict)
        self.info (f'Confusion Marix: {confusion_matrix}')
        self.plot_confusion_matrix(confusion_matrix, ['Bearish', 'Neutral', 'Bullish'])
        self.info(f'Predicted Prop: {test_predict}')
        self.info(f"Predicted values: {np.argmax(test_predict, axis=1)}")

    @staticmethod
    def plot_confusion_matrix(cm, target_names,  title='Confusion matrix', cmap=plt.cm.Blues):
        cm.plot(cmap = cmap)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()
        #plt.savefig(file_name)


class KerasMultiInputFitter(BaseEstimator, ClassifierMixin):
    N_CAT = 3
    DAYS_AHEAD = 5
    LAGGED_DAYS = 0
    VALID_DAYS = 20
    def __init__(self, logger = None, verbose=1, batch_size=128, epochs=200,lr = 0.001, sequence_length= 20,l2_reg= 0.01):
        """
            Called when initializing the classifier
        """
        self.log = logger
        self.current_directory = os.path.dirname(os.path.realpath(__file__))
        self.output_directory = os.path.join(self.current_directory,'data')
        today_date = datetime.datetime.today().strftime('%Y%m%d')
        self.pickle_filename1 = os.path.join(self.output_directory, f"{today_date}_view.pkl")
        self.pickle_filename2 = os.path.join(self.output_directory, f"{today_date}_xjo.pkl")
        self.fig_name = os.path.join( self.output_directory, 'histo.png')
        self.feather_name = os.path.join( self.output_directory, f"{today_date}_xjo.feather")
        self.lr = lr
        self.name = 'keras_multiInput'
        self.data_dir = os.path.join(self.current_directory, 'data')
        self.batch_size = batch_size
        self.epochs = epochs
        self.verbose = verbose        
        self.window_size = sequence_length
        self.l2_reg = l2_reg
        self.find_categorical()

    def find_categorical(self):
        
        # Get list of categorical column names
        self.categorical_columns = ['StatusXO', 'StatusBS', 'day']
        # Get list of non-categorical column names
        self.non_categorical_columns = ['AXVI', 'SKEW', 'ISEE', 'aaii_bulish', 'aaii_bearish', 'BP', 'Value', 'Low', 'High']   
        #self.sequence_columns = ['Value']
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


if __name__ == "__main__":
    log_file = os.path.join(current_directory, 'data/wp.log')
    feather_file = os.path.join(current_directory, 'data/total_data.feather')

    # read in the prepared data set
    logger = logging.getLogger('wp')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    keras_multinput = KerasMultiInput(logger=logger)
