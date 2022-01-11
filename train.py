import pandas as pd
from numpy import concatenate
# from matplotlib import pyplots
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.utils import shuffle
 
# convert series to supervised learning
def generate_supervised_dataset(dataset, n_shifts):
    '''
    This function converts time series data into supervised learning format.
    It creates new columns with shifted respiratory rate and heart rate values.
    Eg: it will create rr at t-1, t-2 and t-3 which are used to predict rr at time t.
    
    It also creates other columns like hour and minute.
    '''
    # filter dataset for in_room users only
    filter_df  = dataset[dataset["in_room"]==True]
    # filter -1.0 values
    filter_df = filter_df[filter_df['hr']!=-1.0]
    filter_df = filter_df[filter_df['rr']!=-1.0]
    # get hour and minute features
    filter_df['Datetime'] = pd.to_datetime(filter_df['ts'],format='%Y-%m-%d %H:%M:%S')
    filter_df['hour'] = filter_df['Datetime'].dt.hour 
    filter_df['minute'] = filter_df['Datetime'].dt.minute 
    # generate shifted dataset
    for i in range(n_shifts):
        filter_df['hr_t-'+str(i+1)] = filter_df.groupby(['user_id'])['hr'].shift(i+1)
        filter_df['rr_t-'+str(i+1)] = filter_df.groupby(['user_id'])['rr'].shift(i+1)
    # rolling mean at every time step
    filter_df['mean_hr'] = filter_df.groupby('user_id')['hr'].rolling(n_shifts).mean().reset_index(0,drop=True)
    filter_df['mean_rr'] = filter_df.groupby('user_id')['rr'].rolling(n_shifts).mean().reset_index(0,drop=True)
    # fill nan's with 0
    filter_df = filter_df.fillna(0)
    return filter_df

def train_test_split(dataframe):
    '''
    Function to split input data into train, validation and test sets.
    '''
    values = dataframe.values
    train_idx = int(dataframe.shape[0]*0.8)
    valid_idx = int(dataframe.shape[0]*0.8*0.2)
    # split into train, valid and test sets
    valid = values[:valid_idx, :]
    train = values[valid_idx:train_idx, :]
    test = values[train_idx:, :]
    # split dataframe into input and outputs
    train_X, train_y = train[:, :-1], train[:, -1]
    valid_X, valid_y = valid[:, :-1], valid[:, -1]
    test_X, test_y = test[:, :-1], test[:, -1]
    return train_X, train_y, valid_X, valid_y, test_X, test_y
 
def run_job(dataset):
    '''
    Function to convert input data to supervised learning format, 
    train DL model and return model artifact.
    '''
    # Change dataset from time series to supervised 
    filter_df  = generate_supervised_dataset(dataset, 3)

    reframed = filter_df[['rr_t-1','rr_t-2','rr_t-3','mean_rr','hour','minute','rr']]
    # Shuffle input data before feeding to the model
    reframed = shuffle(reframed)

    # split into train, valid and test sets
    train_X, train_y, valid_X, valid_y, test_X, test_y = train_test_split(reframed)

    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    valid_X = valid_X.reshape((valid_X.shape[0], 1, valid_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

    # design network
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True))
    model.add(LSTM(20, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True))
    model.add(LSTM(10, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    # fit network
    history = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(valid_X, valid_y), verbose=2, shuffle=False)
    # plot history
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.show()
    return model, test_X, test_y
    
def main():
    # load dataset
    dataset = pd.read_csv('respiratory_rate_data_2020_05.csv')
    # call run_job
    model, test_X, test_y = run_job(dataset)
    # save trained model
#     model.save('resp_rate.h5')
    # make a prediction on test data
    yhat = model.predict(test_X)
    # error on test data
    print ("Test mae: %.3f" %mean_absolute_error(test_y, yhat))

if __name__ == "__main__":
    main()
