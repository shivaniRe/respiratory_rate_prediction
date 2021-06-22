# Respiratory Rate predictions
This project uses time series data along with other features like 'hour' and 'minute' to predict future respiratory rates. 
These future predictions are used to alert if the user has abnormal respiratory rate.

### Features used:
* rr(t-3)
* rr(t-2)
* rr(t-1)
* hour
* minute


### Predictions:
* rr(t)
* rr(t+1)
* rr(t+2)

Train.py file uses data provided to train a LSTM model and save it to the folder. 
Score.py file uses the saved model and fake data to make predictions and alert.

For this project, I created a fake dataset with 5 users and for 4 time steps. In my dataset user 1 
has consistantly low rr, user 5 has high rr and in_room for user 3 is set to False.
