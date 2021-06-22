import keras
import numpy as np 
import pandas as pd
from pandas import read_csv
from train import generate_supervised_dataset

# scoring
def run_job(dataset, model):
    '''
    This function predicts future 3 respiratory rates for a 
    series of respiratory rates. 
    '''
    # convert test dataframe to supervised dataset
    score_df = generate_supervised_dataset(dataset,3).tail(4)
    x_score = score_df[['rr_t-1','rr_t-2','rr_t-3','mean_rr','hour','minute']]
    score_arr = x_score.values
    score_arr = score_arr.reshape((score_arr.shape[0], 1, score_arr.shape[1]))
    # make predictions and append the results to the original dataset
    predictions  = []
    for test in score_arr:
        test = test.reshape((test.shape[0], 1, test.shape[1]))
        predictions.append(get_future_predictions(test, model))
    score_df['predictions'] = predictions
    return score_df[['rr','hr','in_room','ts','user_id','predictions']]
    
    
def get_future_predictions(score_arr, model):    
    '''
    This function takes one input vector for predicting time t value, 
    uses that prediction to predict t+1 value and so on.
    '''
    result = []
    for i in range(3):
        res = model.predict(np.array(score_arr))
        result.append(res[0][0])
        score_arr=np.insert(score_arr[0][0],0,res[0][0],axis = 0) 
        score_arr= np.delete(score_arr, 3, 0)
        score_arr[3]=np.mean(score_arr[:3])
        score_arr = np.reshape(score_arr,(1, 1,6))
    return result


def main():
    # load test data
    dataset = pd.read_csv('fake_test_data.csv')

    # load saved model
    model = keras.models.load_model('resp_rate.h5')
    
    # call run_job
    result_df = run_job(dataset, model)

    # if more than 1 abnormal value appears in the future 3 predictions 
    # alert for the user
    # check how many values are out of danger range for rr (12>rr>24)
    for res, user in np.array(result_df[['predictions','user_id']]):
        if len([1 for i in res if i<=12 or i>=24])>1:
            print (f"user {user} has abnormal respiratory rate")

if __name__ == "__main__":
    main()

