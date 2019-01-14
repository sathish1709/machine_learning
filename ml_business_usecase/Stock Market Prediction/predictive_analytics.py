#Please import all the packages by pip install

import pandas_datareader as web
import pandas as pd
import numpy as np
import datetime as dt
from datetime import timedelta
import matplotlib.pyplot as mp
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import acf,pacf
from statsmodels.tsa.arima_model import ARIMA
import home_page as hm

#Predictive mode menu which contains Linear Regression, Auto Regressive Integrated Moving Average and a option to traverse back to previous menu and explore some other functionalities
def predictive_mode(dataset,start_date,end_date,company_name,company):
   print("*" *100)
   print(f"Thanks for Choosing Predictive Stock Analysis Option. You can choose \n1.Prediction using Linear Regression for {company_name} \n2.Prediction using ARIMA Model for {company_name} \n3.Main Menu\n")
   print("*" *100)
   option = input("Choose from the above option\nEnter your Option:\n")
   
   try:
       if option == "1":
           linear_mode(dataset,start_date,end_date,company_name,company)
           predictive_mode(dataset,start_date,end_date,company_name,company)
           
       elif option == "2":
           arima_model(dataset,start_date,end_date,company_name,company)
           predictive_mode(dataset,start_date,end_date,company_name,company)
           
       elif option == "3":
           hm.mainmenu(dataset,start_date,end_date,company_name,company)
           
       else:
           print(f"Guess, You have entered the wrong option, Please try again !!! You can choose \n1.Prediction using Linear Regression for {company_name} \n2.Prediction using ARIMA Model for {company_name} \n3.Main Menu\n")
           print("\n")
           predictive_mode(dataset,start_date,end_date,company_name,company)
           
   except ValueError:
       print("Please select from given below options")
       predictive_mode(dataset,start_date,end_date,company_name,company)

#This function is used to predict price using Linear Regression    
def linear_mode(data,start_date,end_date,company_name,company):   
        dataset = data.copy() 
        prediction_date=input("Enter the date for which you want to predict the stock price in YYYY-MM-DD format:") 
        predicted_year, predicted_month, predicted_day = map(int, prediction_date.split('-'))
        future_date = dt.datetime(predicted_year, predicted_month, predicted_day)
        future_end_date = input('Enter a date for window period in YYYY-MM-DD format:')
        year, month, day = map(int, future_end_date.split('-'))
        end_date = dt.datetime(year, month, day)
        difference = future_date - end_date
        z = difference.days
        print("The window period is: ",difference.days)
        dataset.reset_index(inplace=True,drop=False)
        dates = dataset.index.tolist()
          
         
        prices = dataset['Adj Close'].tolist()
        
        linear_mod = linear_model.LinearRegression() #defining the linear regression model
        dates = np.reshape(dates,(len(dates),1)) # converting to matrix of n X 1
        prices = np.reshape(prices,(len(prices),1))
        linear_mod.fit(dates,prices) #fitting the data points in the model
        mp.scatter(dates, prices, color='yellow', label= 'Actual Price') #plotting the initial datapoints
        mp.plot(dates, linear_mod.predict(dates), color='red', linewidth=3, label = 'Predicted Price') #plotting the line made by linear regression
        mp.title('Linear Regression : Time vs. Price')
        mp.legend()
        mp.xlabel('Date Integer')
        mp.show()
        
        predicted_price =linear_mod.predict(z)
        print(f"The stock predicted price of {company} for "+str(predicted_year) +"-"+str(predicted_month)+"-"+str(predicted_day)+" is: $",str(predicted_price))
        print("The regression coefficient is ",str(linear_mod.coef_[0][0]))
        
        startdate = dt.datetime.now()- timedelta(days=2)
        enddate = dt.datetime.now()- timedelta(days=2)
        dataset = web.DataReader(company,'yahoo',startdate,enddate)
        actualClose = dataset['Adj Close']
        rms = np.sqrt(mean_squared_error(actualClose,predicted_price))
        print(f"Root Mean square Value for {company_name} using Linear Regression: ",rms)
        print(f"Mean square Value for {company_name} using Linear Regression: ",np.sqrt(rms))
  
         
      
  
#This function is used to predict price using ARIMA Model          
def arima_model(data,start_date,end_date,company_name,company):
   dataset = data.copy()
   
#Creating a training DataSet
   initial_start_date = input("Enter a Start date which you want to take in testing window in YYYY-MM-DD format for training dataset:\n")
   year, month, day = map(int, initial_start_date.split('-'))
   end_date = dt.datetime(year, month, day)
   start_date = end_date - timedelta(days = 200)
   dataset= web.DataReader(company,'yahoo',start_date,end_date)
   close = dataset['Adj Close'].resample('MS').mean()
   close = close.fillna(close.bfill())
   closePrice_log = np.log(close)
   mp.plot(closePrice_log)
   mp.show()
   
   acf_value = acf(closePrice_log)[1:20]
   test_data = pd.DataFrame([acf_value]).T
   test_data.columns =["AutoCorrelation"]
   test_data.index +=1
   test_data.plot(kind= "bar")
   mp.show()
   
   closePrice_log_diff=closePrice_log-closePrice_log.shift()
   diff=closePrice_log_diff.dropna()
   acf_1_diff =  acf(diff)[1:20]
   test_df = pd.DataFrame([acf_1_diff]).T
   test_df.columns = ['First Difference Autocorrelation']
   test_df.index += 1
   test_df.plot(kind='bar')
   pacf_1_diff =  pacf(diff)[1:20]
   mp.plot(pacf_1_diff)
   mp.show()
    
   close_price_matrix=closePrice_log.as_matrix()
   model = ARIMA(close_price_matrix, order=(0,1,0))
   model_fit = model.fit(disp=0)
   print(model_fit.summary())
   mp.show()

   initial_start_date = input("Enter a Start date which you want to take in testing window in YYYY-MM-DD format:")
   year, month, day = map(int, initial_start_date.split('-'))
   start_date = dt.datetime(year, month, day)
   
   final_end_date = input("Enter a End date for which you want the price predicted in YYYY-MM-DD format:")
   year, month, day = map(int, final_end_date.split('-'))
   end_date_value = dt.datetime(year, month, day)
    
   start_date = start_date.toordinal()
   end_date = end_date_value.toordinal()
   predictions=model_fit.predict(start=start_date,end=end_date, typ='levels')
   predictionsadjusted=np.exp(predictions)
   print(f"The stock predicted price for {company_name} using ARIMA Model is: $",predictionsadjusted[1],"for date",final_end_date)
   predictions = pd.DataFrame(predictions,columns=['Prediction'])
   mp.plot(close_price_matrix)
   mp.show()

    
