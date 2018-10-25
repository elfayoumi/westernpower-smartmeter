import pandas as pd
import numpy as np
from random import sample, randint
import os, datetime
from pathlib import Path
import itertools
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import logging
from datetime import timedelta
import datetime
import matplotlib.pyplot as plt
import matplotlib as mpl
import  warnings,sklearn
import pickle
# Import the email modules we'll need
from sklearn.base import BaseEstimator, ClassifierMixin
from email.message import EmailMessage
from pandas_ml import ConfusionMatrix
from sklearn.model_selection import train_test_split
from sklearn_pandas import DataFrameMapper
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator
from pandas.api.types import is_string_dtype, is_numeric_dtype
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from keras.models import Sequential, Model
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam, Adadelta, SGD
from keras.callbacks import ModelCheckpoint
from keras import layers
from keras import Input
from keras.layers.core import Dense, Activation, Dropout
from keras import regularizers
import random

def numericalize(df, col, name, max_n_cat):
    """ Changes the column col from a categorical type to it's integer codes.

    Parameters:
    -----------
    df: A pandas dataframe. df[name] will be filled with the integer codes from
        col.

    col: The column you wish to change into the categories.
    name: The column name you wish to insert into df. This column will hold the
        integer codes.

    max_n_cat: If col has more categories than max_n_cat it will not change the
        it to its integer codes. If max_n_cat is None, then col will always be
        converted.

    Examples:
    ---------
    >>> df = pd.DataFrame({'col1' : [1, 2, 3], 'col2' : ['a', 'b', 'a']})
    >>> df
       col1 col2
    0     1    a
    1     2    b
    2     3    a

    note the type of col2 is string

    >>> train_cats(df)
    >>> df

       col1 col2
    0     1    a
    1     2    b
    2     3    a

    now the type of col2 is category { a : 1, b : 2}

    >>> numericalize(df, df['col2'], 'col3', None)

       col1 col2 col3
    0     1    a    1
    1     2    b    2
    2     3    a    1
    """
    if not is_numeric_dtype(col) and ( max_n_cat is None or col.nunique()>max_n_cat):
        df[name] = col.cat.codes+1

def scale_vars(df, mapper):
    warnings.filterwarnings('ignore', category=sklearn.exceptions.DataConversionWarning)
    if mapper is None:
        map_f = [([n],StandardScaler()) for n in df.columns if is_numeric_dtype(df[n])]
        mapper = DataFrameMapper(map_f).fit(df)
    df[mapper.transformed_names_] = mapper.transform(df)
    return mapper

def fix_missing(df, col, name, na_dict):
    """ Fill missing data in a column of df with the median, and add a {name}_na column
    which specifies if the data was missing.

    Parameters:
    -----------
    df: The data frame that will be changed.

    col: The column of data to fix by filling in missing data.

    name: The name of the new filled column in df.

    na_dict: A dictionary of values to create na's of and the value to insert. If
        name is not a key of na_dict the median will fill any missing data. Also
        if name is not a key of na_dict and there is no missing data in col, then
        no {name}_na column is not created.


    Examples:
    ---------
    >>> df = pd.DataFrame({'col1' : [1, np.NaN, 3], 'col2' : [5, 2, 2]})
    >>> df
       col1 col2
    0     1    5
    1   nan    2
    2     3    2

    >>> fix_missing(df, df['col1'], 'col1', {})
    >>> df
       col1 col2 col1_na
    0     1    5   False
    1     2    2    True
    2     3    2   False


    >>> df = pd.DataFrame({'col1' : [1, np.NaN, 3], 'col2' : [5, 2, 2]})
    >>> df
       col1 col2
    0     1    5
    1   nan    2
    2     3    2

    >>> fix_missing(df, df['col2'], 'col2', {})
    >>> df
       col1 col2
    0     1    5
    1   nan    2
    2     3    2


    >>> df = pd.DataFrame({'col1' : [1, np.NaN, 3], 'col2' : [5, 2, 2]})
    >>> df
       col1 col2
    0     1    5
    1   nan    2
    2     3    2

    >>> fix_missing(df, df['col1'], 'col1', {'col1' : 500})
    >>> df
       col1 col2 col1_na
    0     1    5   False
    1   500    2    True
    2     3    2   False
    """
    if is_numeric_dtype(col):
        if pd.isnull(col).sum() or (name in na_dict):
            df[name+'_na'] = pd.isnull(col)
            filler = na_dict[name] if name in na_dict else col.median()
            df[name] = col.fillna(filler)
            na_dict[name] = filler
    return na_dict
def proc_df(df, y_fld=None, skip_flds=None, ignore_flds=None, do_scale=False, na_dict=None,
            preproc_fn=None, max_n_cat=None, subset=None, mapper=None):
    """ proc_df takes a data frame df and splits off the response variable, and
    changes the df into an entirely numeric dataframe.

    Parameters:
    -----------
    df: The data frame you wish to process.

    y_fld: The name of the response variable

    skip_flds: A list of fields that dropped from df.

    ignore_flds: A list of fields that are ignored during processing.

    do_scale: Standardizes each column in df. Takes Boolean Values(True,False)

    na_dict: a dictionary of na columns to add. Na columns are also added if there
        are any missing values.

    preproc_fn: A function that gets applied to df.

    max_n_cat: The maximum number of categories to break into dummy values, instead
        of integer codes.

    subset: Takes a random subset of size subset from df.

    mapper: If do_scale is set as True, the mapper variable
        calculates the values used for scaling of variables during training time (mean and standard deviation).

    Returns:
    --------
    [x, y, nas, mapper(optional)]:

        x: x is the transformed version of df. x will not have the response variable
            and is entirely numeric.

        y: y is the response variable

        nas: returns a dictionary of which nas it created, and the associated median.

        mapper: A DataFrameMapper which stores the mean and standard deviation of the corresponding continuous
        variables which is then used for scaling of during test-time.

    Examples:
    ---------
    >>> df = pd.DataFrame({'col1' : [1, 2, 3], 'col2' : ['a', 'b', 'a']})
    >>> df
       col1 col2
    0     1    a
    1     2    b
    2     3    a

    note the type of col2 is string

    >>> train_cats(df)
    >>> df

       col1 col2
    0     1    a
    1     2    b
    2     3    a

    now the type of col2 is category { a : 1, b : 2}

    >>> x, y, nas = proc_df(df, 'col1')
    >>> x

       col2
    0     1
    1     2
    2     1

    >>> data = DataFrame(pet=["cat", "dog", "dog", "fish", "cat", "dog", "cat", "fish"],
                 children=[4., 6, 3, 3, 2, 3, 5, 4],
                 salary=[90, 24, 44, 27, 32, 59, 36, 27])

    >>> mapper = DataFrameMapper([(:pet, LabelBinarizer()),
                          ([:children], StandardScaler())])

    >>>round(fit_transform!(mapper, copy(data)), 2)

    8x4 Array{Float64,2}:
    1.0  0.0  0.0   0.21
    0.0  1.0  0.0   1.88
    0.0  1.0  0.0  -0.63
    0.0  0.0  1.0  -0.63
    1.0  0.0  0.0  -1.46
    0.0  1.0  0.0  -0.63
    1.0  0.0  0.0   1.04
    0.0  0.0  1.0   0.21
    """
    if not ignore_flds: ignore_flds=[]
    if not skip_flds: skip_flds=[]
    if subset: df = get_sample(df,subset)
    else: df = df.copy()
    ignored_flds = df.loc[:, ignore_flds]
    df.drop(ignore_flds, axis=1, inplace=True)
    if preproc_fn: preproc_fn(df)
    if y_fld is None: y = None
    else:
        if not is_numeric_dtype(df[y_fld]): df[y_fld] = df[y_fld].cat.codes
        y = df[y_fld].values
        skip_flds += [y_fld]
    df.drop(skip_flds, axis=1, inplace=True)

    if na_dict is None: na_dict = {}
    else: na_dict = na_dict.copy()
    na_dict_initial = na_dict.copy()
    for n,c in df.items(): na_dict = fix_missing(df, c, n, na_dict)
    if len(na_dict_initial.keys()) > 0:
        df.drop([a + '_na' for a in list(set(na_dict.keys()) - set(na_dict_initial.keys()))], axis=1, inplace=True)
    if do_scale: mapper = scale_vars(df, mapper)
    for n,c in df.items(): numericalize(df, c, n, max_n_cat)
    df = pd.get_dummies(df, dummy_na=True)
    df = pd.concat([ignored_flds, df], axis=1)
    res = [df, y, na_dict]
    if do_scale: res = res + [mapper]
    return res
    
class KerasMultiInput(BaseEstimator, ClassifierMixin):

    DAYS_AHEAD = 1
    LAGGED_DAYS = 0
    VALID_DAYS = 15
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

    def __init__(self, verbose=1, batch_size=128, epochs=200,lr = 0.001, sequence_length= 20,l2_reg= 0.01, logger = None, ts_dim=1):
        self.log = logger
        self.current_directory = os.path.dirname(os.path.realpath(__file__))
        self.output_directory = os.path.join(self.current_directory,'data')
        self.D = ts_dim
        self.lr = lr
        self.name = 'keras_multiInput'
        self.data_dir = os.path.join(self.current_directory, 'data')
        self.batch_size = batch_size
        self.epochs = epochs
        self.verbose = verbose
        self.ts_scaler = MinMaxScaler(feature_range=(0,.9), copy=True)
        self.window_size = sequence_length
        self.l2_reg = l2_reg
        self.pickle_file = os.path.join(self.output_directory, 'ts.pickle')
    
    def find_categorical(self, X):        
        # Get list of categorical column names
        self.categorical_columns = ['LCLid','Acorn', 'Acorn_grouped', "icon", "stdorToU", "Type", "day.of.week", 'precipType',  'summary', 'before_holiday', 'after_holiday', 'month', 'year']
        # Get list of non-categorical column names
        self.non_categorical_columns =list(filter(lambda x: x not in self.categorical_columns, X.columns))
        self.sequence_columns = ['energy_sum']
        self.label_column = 'energy_sum'
    def proces_categorical_columns(self, df):
        for v in self.categorical_columns:
            df[v] = df[v].astype('category').cat.as_ordered()
        for v in self.non_categorical_columns:
            df[v] = df[v].astype('float32')
        self.cat_sz = [(c, len(df[c].cat.categories)+1) for c in self.categorical_columns]
        self.emb_szs = [(c, min(50, (c+1)//2)) for _,c in self.cat_sz]
        self.ts_scaler.fit(df[self.label_column].fillna(method = 'ffill').reshape(-1,1))

    def create_dataset(self, df):
        
        y = df[self.label_column]
        ahead = y.groupby(level = 0).shift(-self.DAYS_AHEAD)
        df['Value'] = ahead.to_frame()
        df = df.dropna()
        self.proces_categorical_columns(df)

        df, y, self.nas, self.mapper = proc_df(df, y_fld='Value',  do_scale=True)
        df = df[self.categorical_columns + self.non_categorical_columns ]
        self.info(df.dtypes)
        self.df = df
        self.y = self.ts_scaler.fit_transform(y.reshape(-1,1))
        self.ts = self.window_transform_series(self.window_size,df.index)
        
        return (self.df, self.ts,self.y )
    def window_transform_series(self,window_size, index):
        
        v = self.df[self.label_column]
        
        # x values ends 1 before the end
        X = []
        
        # Create window_size columns of shiffted x values
        for lclid, new_df in v.groupby(level = 0):
            d = np.asarray([x[1] for x in new_df.index])

            for id in d:
                r = new_df[lclid][:id].values[-window_size:]
                if(len(r) < window_size):
                    s = np.zeros(window_size - len(r))
                    r = np.insert(r, 0, s, axis = 0)
                X.append(r)

        # reshape each
        X =np.asarray(X)
       
        return X
    def load_pickled_data(self):
        self.X_train = pickle.load(open( os.path.join(self.output_directory, 'x_train_picle.pickle'), "rb" ) )
        self.X_valid = pickle.load(open( os.path.join(self.output_directory, 'x_valid_picle.pickle'), "rb" ) )
        self.ts_train = pickle.load( open( os.path.join(self.output_directory, 'ts_train_picle.pickle'), "rb" ) )
        self.ts_valid = pickle.load( open( os.path.join(self.output_directory, 'ts_valid_picle.pickle'), "rb" ) )
        self.y_train = pickle.load( open( os.path.join(self.output_directory, 'y_train_picle'), "rb" ) )
        self.y_valid = pickle.load( open( os.path.join(self.output_directory, 'y_valid_picle'), "rb" ) )
        return self.X_train, self.ts_train, self.y_train, self.X_valid, self.ts_valid, self.y_valid  

    def separate_train_valid(self, X, ts,y):
        max_date = max(X.index)[1]
        valid_start_day = max_date - timedelta(days=KerasMultiInput.VALID_DAYS)       
        train_idx = list(filter(lambda t: t[1] < valid_start_day, X.index))
        valid_idx = list(filter(lambda t: t[1] >= valid_start_day, X.index))
        ts = pd.DataFrame(ts, index = X.index)
        y_pd = pd.DataFrame(y, index = X.index)
        self.X_train = X.loc[train_idx]
        self.ts_train = ts.loc[train_idx].values.reshape(-1, self.window_size, self.D)
        self.y_train = y_pd.loc[train_idx].values.reshape(-1,1)
        self.X_valid = X.loc[valid_idx]
        self.ts_valid = ts.loc[valid_idx].values.reshape(-1, self.window_size, self.D)
        self.y_valid = y_pd.loc[valid_idx].values.reshape(-1,1)
        pickle.dump( self.X_train, open( os.path.join(self.output_directory, 'x_train_picle.pickle'), "wb" ) )
        pickle.dump( self.X_valid, open( os.path.join(self.output_directory, 'x_valid_picle.pickle'), "wb" ) )
        pickle.dump( self.ts_train, open( os.path.join(self.output_directory, 'ts_train_picle.pickle'), "wb" ) )
        pickle.dump( self.ts_valid, open( os.path.join(self.output_directory, 'ts_valid_picle.pickle'), "wb" ) )
        pickle.dump( self.y_train, open( os.path.join(self.output_directory, 'y_train_picle'), "wb" ) )
        pickle.dump( self.y_valid, open( os.path.join(self.output_directory, 'y_valid_picle'), "wb" ) )


        return self.X_train, self.ts_train, self.y_train, self.X_valid, self.ts_valid, self.y_valid  
    
    def model_setup(self):
        
        cat_input = [Input(shape=(1,), dtype='int32', name=c) for c in self.categorical_columns]
        seq_input = Input(shape=( None, 1), dtype='float32', name='rnn_input')
        all_layers = []
        for i in range(len(cat_input)):
            emb = layers.Embedding(self.emb_szs[i][0], self.emb_szs[i][1])(cat_input[i])
            flat = layers.Flatten()(emb)
            all_layers.append(flat)

        contInput = Input(shape=(len(self.non_categorical_columns),), dtype='float32', name='continuouse')
       
        # seq_lay = layers.LSTM(32,return_sequences=True, activation='tanh', 
        #         kernel_regularizer=regularizers.l2(self.l2_reg))(seq_input)

        # seq_lay = layers.LSTM(64, return_sequences=True, activation='tanh', 
        #          kernel_regularizer=regularizers.l2(self.l2_reg))(seq_lay)
        seq_lay = layers.LSTM(16, return_sequences=False, activation='tanh', 
                 kernel_regularizer=regularizers.l2(self.l2_reg))(seq_input)

        #concatenated = layers.concatenate([categDense, continuousDense, seq_lay1], axis =-1)
        all_layers.append(contInput)
        all_layers.append(seq_lay)
        lay = layers.concatenate(all_layers, axis =-1)
        lay = BatchNormalization()(lay)
        # lay = Dense(64, kernel_regularizer=regularizers.l2(self.l2_reg))(lay)
        # lay = Dense(128, activation='tanh', kernel_regularizer=regularizers.l2(self.l2_reg))(lay)

        #lay = Dropout(0.5)(lay)
        #lay = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(self.l2_reg))(lay)
        #lay = BatchNormalization()(lay)
        lay = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(self.l2_reg))(lay)
        lay = BatchNormalization()(lay)
        #lay = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(self.l2_reg))(lay)
        lay = Dense(4, activation='relu', kernel_regularizer=regularizers.l2(self.l2_reg))(lay)
        #lay = BatchNormalization()(lay)
        answer = layers.Dense(1, activation='sigmoid')(lay)
        inputs_all = cat_input
        inputs_all.append(contInput)
        inputs_all.append(seq_input)
        self.model = Model(inputs_all, answer)
        self.info(self.model.summary())
        adam = Adam(lr=self.lr)
        self.model.compile(loss='binary_crossentropy', optimizer=adam,metrics=['mse', 'mae'])
    def load_model(self):
        self.model.load_weights(self.best_weights(self.name))
        

    def fit_data(self, show_figures = True):
        checkpointer = ModelCheckpoint(filepath=self.best_weights(self.name), verbose=self.verbose, save_best_only=True)
        train_values = {c:self.X_train[c] for c in self.categorical_columns}
        train_values['continuouse'] = self.X_train[self.non_categorical_columns]
        train_values['rnn_input'] = self.ts_train
        val_values = {c:self.X_valid[c] for c in self.categorical_columns}
        val_values['continuouse'] = self.X_valid[self.non_categorical_columns]
        val_values['rnn_input'] = self.ts_valid
        
        history = self.model.fit(train_values, self.y_train, epochs=self.epochs, batch_size = self.batch_size,
                                     validation_data=(val_values, self.y_valid), 
                                     verbose=self.verbose, shuffle=True,
                                     #validation_split=0.2,
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
       
        p  = self.model.evaluate(val_values, self.y_valid)
        self.info(f"Validation Score: {p}")
        
    def _meaning(self, x):
        # returns True/False according to fitted classifier
        # notice underscore on the beginning
        return( True )

    def unscale_y_value(self, y):
        return self.ts_scaler.inverse_transform(y)

    def predict(self, X,ts, y=None):
        try:
            test_values = {c:X[c] for c in self.categorical_columns}
            test_values['continuouse'] = X[self.non_categorical_columns]
            test_values['rnn_input'] = ts
            test_predict = self.model.predict(test_values)

            return test_predict
        except AttributeError:
            raise RuntimeError("You must train classifer before predicting data!")

        return None

    def score(self, X,ts, y=None):
        # counts number of values bigger than mean
        test_values = {c:X[c] for c in self.categorical_columns}
        test_values['continuouse'] = X[self.non_categorical_columns]
        test_values['rnn_input'] = ts
        t = self.model.evaluate(x=test_values,y=y, batch_size=self.batch_size)
        self.log.info(t)
        return t

if __name__ == "__main__":
    current_directory = os.path.dirname(os.path.realpath(__file__))
    log_file = os.path.join(current_directory, 'data/wp.log')
    feather_file = os.path.join(current_directory, 'data/total_data_filled.feather')
    df = pd.read_feather(feather_file)
    acorn_Lclid = df[['Acorn', 'LCLid']].drop_duplicates()
    df = df.set_index([ 'index', 'day'])
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
    # date_fields = ["temperatureMaxTime", "temperatureMinTime", "apparentTemperatureMinTime",
    #                 "apparentTemperatureHighTime","sunsetTime", "uvIndexTime"  ,"sunriseTime","temperatureHighTime", "temperatureLowTime", 
    #                  "apparentTemperatureMaxTime",
    #                  "apparentTemperatureLowTime"]

    # for date_field in date_fields:
    #     name = date_field.replace('Time', 'Hour')
    #     df[name] = df[date_field].apply(lambda x: x.hour)
    df = df.drop([ 'energy_count', "temperatureMaxTime", "temperatureMinTime", "apparentTemperatureMinTime",
                    "apparentTemperatureHighTime","sunsetTime", "uvIndexTime"  ,"sunriseTime","temperatureHighTime", "temperatureLowTime", 
                     "apparentTemperatureMaxTime",
                     "apparentTemperatureLowTime"], axis = 1)
    
    logger.info(df.head())
    try:
        keras_multinput = KerasMultiInput(logger=logger, verbose=2, batch_size=2000, epochs=20, lr=0.0001, l2_reg=0.05, sequence_length =5)
        keras_multinput.find_categorical(df)
        logger.info(df.head())
        #df, ts, y = keras_multinput.create_dataset(df)
        #X_train, ts_train, y_train, X_valid, ts_valid, y_valid = keras_multinput.separate_train_valid(df, ts, y)
        keras_multinput.proces_categorical_columns(df)
        X_train, ts_train, y_train, X_valid, ts_valid, y_valid = keras_multinput.load_pickled_data()
        logger.info(X_train.head())
        keras_multinput.model_setup()
        #keras_multinput.fit_data(show_figures=True)
        keras_multinput.load_model()
        ts_valid = pd.DataFrame(ts_valid.reshape(-1, keras_multinput.window_size), index=X_valid.index)
        ts_train = pd.DataFrame(ts_train.reshape(-1, keras_multinput.window_size), index=X_train.index)
        y_train = pd.DataFrame(y_train, index = X_train.index, columns = ['Real Value'])
        y_valid = pd.DataFrame(y_valid, index = X_valid.index, columns = ['Real Value'])
        lclids = df.LCLid.unique()
        lclid_test = list( [ lclids[i] for i in sorted(random.sample(range(len(lclids)), 4)) ])
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        for lclid, clr in zip(lclid_test, colors[:len(lclid_test)]):
            logger.info(f'Predicting values for LCLID: {lclid}')
            X = X_train.loc[lclid,:]
            if(len(X) > 0):
                ts = ts_train.loc[lclid, :].values.reshape(-1, keras_multinput.window_size,keras_multinput.D)
                y = keras_multinput.unscale_y_value(keras_multinput.predict(X,ts).reshape(-1,1))
                y_pred = pd.DataFrame(y, index = y_train.loc[lclid,:].index, columns = ['Predicted Value'] )
                idx = list([t + datetime.timedelta(days=KerasMultiInput.DAYS_AHEAD) for t in y_train.loc[lclid,:].index])
                y_real = pd.DataFrame(keras_multinput.unscale_y_value(y_train.loc[lclid,:]), index = idx, columns= ['Real Values'])

                plt.plot(y_pred, label=f'Predicted value (kw) for LCLid: {lclid}', linestyle = '--', color=clr )
                plt.plot(y_real,label=f'Real value (kw) for LCLid: {lclid}', linestyle = '-', color = clr)

            X = X_valid.loc[lclid,:]
            if(len(X) > 0):
                ts = ts_valid.loc[lclid, :].values.reshape(-1, keras_multinput.window_size,keras_multinput.D)
                y = keras_multinput.unscale_y_value(keras_multinput.predict(X,ts).reshape(-1,1))
                y_pred = pd.DataFrame(y, index = y_valid.loc[lclid,:].index, columns = ['Predicted Value'] )
                idx = list([t + datetime.timedelta(days=KerasMultiInput.DAYS_AHEAD) for t in  y_valid.loc[lclid,:].index])
                y_real = pd.DataFrame(keras_multinput.unscale_y_value(y_valid.loc[lclid,:]), index =idx, columns= ['Real Values'])

                plt.plot(y_pred, linestyle = '--', color=clr , linewidth=2)
                plt.plot(y_valid.loc[lclid,:], linestyle = '-', color = clr, linewidth=2)
        
        plt.legend(loc='best')
        plt.xlabel('Date')
        plt.ylabel('Energy daily sum (kw.h)')
        plt.show()


        # predict_y_train =  keras_multinput.unscale_y_value(keras_multinput.predict(X_train,ts_train.values.reshape(-1, keras_multinput.window_size,keras_multinput.D)).reshape(-1,1))
        # predict_y_valid =  keras_multinput.unscale_y_value(keras_multinput.predict(X_valid,ts_valid.values.reshape(-1, keras_multinput.window_size,keras_multinput.D)).reshape(-1,1))

        # pickle.dump( predict_y_train, open( os.path.join('./data', 'predict_y_train.pickle'), "wb" ) )
        # pickle.dump( predict_y_valid, open( os.path.join('./data', 'predict_y_valid.pickle'), "wb" ) )
        yy =pickle.load(open( os.path.join('./data', 'predict_y_train.pickle'), "rb" ) )
        predict_y_train = pd.DataFrame(yy, columns = ['Predicted'], index = X_train.index)
        predict_y_train['Real'] = keras_multinput.unscale_y_value(y_train)
        yy = pickle.load(open( os.path.join('./data', 'predict_y_valid.pickle'), "rb" ) )
        predict_y_valid = pd.DataFrame(yy, columns = ['Predicted'], index = X_valid.index)
        predict_y_valid['Real'] = keras_multinput.unscale_y_value(y_valid)
        logger.info(predict_y_train.head())
        predict_y_train = predict_y_train.reset_index().merge(acorn_Lclid,how='inner', left_on=['index'], right_on=['LCLid'])
        predict_y_valid = predict_y_valid.reset_index().merge(acorn_Lclid,how='inner', left_on=['index'], right_on=['LCLid'])
        logger.info(predict_y_train.head())
        predict_y_train = predict_y_train.drop(['LCLid', 'index'],axis=1).groupby(['Acorn','day']).sum().dropna()
        predict_y_valid = predict_y_valid.drop(['LCLid', 'index'],axis=1).groupby(['Acorn','day']).sum().dropna()
        logger.info(predict_y_train.head())
        acrons = acorn_Lclid.Acorn.unique()
        acorn_tests = list( [ acrons[i] for i in sorted(random.sample(range(len(acrons)), 4)) ])
        acorn_tests.append('ACORN-A')
        for acorn, clr in zip(acorn_tests, colors[:len(lclid_test)]):
            logger.info(f'Predicting values for Acorn: {acorn}')

            y_pred = predict_y_train.loc[acorn,:]
            plt.plot(y_pred.Predicted, label=f'Predicted value (kw) for Acorn: {acorn}', linestyle = '--', color=clr )
            plt.plot(y_pred.Real,label=f'Real value (kw) for Acorn: {acorn}', linestyle = '-', color = clr)
            y_pred = predict_y_valid.loc[acorn,:]
            plt.plot(y_pred['Predicted'], linestyle = '--', color=clr , linewidth=2, label='Validation Predicted')
            plt.plot(y_pred['Real'], linestyle = '-', color = clr, linewidth=2, label = 'Validation Real')
    
        plt.legend(loc='best')
        plt.xlabel('Date')
        plt.ylabel('Energy daily sum (kw.h)')
        plt.show()



    


    except Exception as ex:
        logger.exception(ex)
    finally:
        logger.handlers = []
