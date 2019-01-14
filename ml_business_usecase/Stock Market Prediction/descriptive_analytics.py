#Please import all the packages by pip install

import numpy as np
import home_page as hm
import candlestick_timeseries as ct
import pandas_datareader as web

#Descriptive mode menu which contains Basic Stats, Advanced Stats, Also there is a option if user wants to compare multiple stocks at the same time
def descriptive_mode(dataset,start_date,end_date,company_name,company):
   print("\n"+"*" *100) 
   print(f"Thanks for Choosing Descriptive Stock Analysis Option. You can choose \n1.Basic Statistics Analysis for {company_name}\n2.Advanced Statistics Analysis for {company_name} \n3.Compare Stockes between different companies \n4.Main Menu\n")
   print("\n"+"*" *100)
   option = input("Choose from the above option\nEnter your Option:")
   try:
       if option == "1":
           basic_stats(dataset,start_date,end_date,company_name,company)
           
       elif option == "2":
           advanced_stats(dataset,start_date,end_date,company_name,company)
           
       elif option == "3":
           comparestocks(dataset,start_date,end_date,company_name,company)
           
       elif option == "4":
           hm.mainmenu(dataset,start_date,end_date,company_name,company)
           
       else:
           print(f"Guess, You have entered the wrong option, Please try again !!! You can choose \n1.For Stock Analysis for {company}\n 2.Stock Prediction for {company} \n3.Main Menu\n")
           print("\n")
           descriptive_mode(dataset,start_date,end_date,company_name,company)
           
   except ValueError:
      print("Enter the correct option from below menu")
      descriptive_mode(dataset,start_date,end_date,company_name,company)


 #The menu performs basic stats operation on all the attributes of the data set   
def basic_stats(dataset,start_date,end_date,company_name,company):
   print("\n"+"*" *100)
   option = input("Please enter the choice for which you need basic statistics i.e\n 1.For Open price\n 2.For Close price\n 3.For High\n 4.For Low\n 5.For Adjusted Close\n 6.For Volume\n 7.For Previous Menu\n Enter your Option: \n\n"+"*" *100+"\n\n")
   print("\n"+"*" *100)
   try:
       if option == "1":
           choice = "Open"
           basic_stats_info(choice,dataset,start_date,end_date,company_name,company)
       
       elif option == "2":
          choice = "Close"
          basic_stats_info(choice,dataset,start_date,end_date,company_name,company)
          
       elif option == "3":
          choice = "High"
          basic_stats_info(choice,dataset,start_date,end_date,company_name,company)
          
       elif option == "4":
          choice = "Low"
          basic_stats_info(choice,dataset,start_date,end_date,company_name,company)
          
       elif option == "5":
          choice = "Adj Close"
          basic_stats_info(choice,dataset,start_date,end_date,company_name,company)
           
       elif option == "6":
          choice = "Volume"
          basic_stats_info(choice,dataset,start_date,end_date,company_name,company)
          
       elif option == "7":
          descriptive_mode(dataset,start_date,end_date,company_name,company)
          
       else:
           print("Guess, You have entered the wrong option, Please try again !!!")
           print("\n")
           basic_stats(dataset,start_date,end_date,company_name,company)
           
   except ValueError:
       print("Enter from the below menu")
       basic_stats(dataset,start_date,end_date,company_name,company)

 #The function which prints all the all the basic info for all the attributes of data set from above menu  
def basic_stats_info(choice,dataset,start_date,end_date,company_name,company):
   print("\n"+"*" *100)
   print(f"\nBasic Summary stats_menu for {choice} Price of {company_name}")
   print(f"\nMean for {choice} Price of {company_name}                            : ",dataset[choice].mean())
   print(f"\nMedian for {choice} Price of {company_name}                          : ",dataset[choice].median())
   print(f"\nRange for {choice} Price of {company_name}                           : ",(dataset[choice].max() - dataset[choice].min()))
   print(f"\nFirst Quartile (25th Percentile) for {choice} Price of {company_name}: ", dataset[choice].quantile(0.25))
   print(f"\nThird Quartile (75th Percentile) for {choice} Price of {company_name}: ",dataset[choice].quantile(0.75))
   print(f"\nStandard deviation for {choice} Price of {company_name}              : ",np.std(dataset[choice]))
   print(f"\nCoefficient of Variation for {choice} Price of {company_name}        : ",np.std(dataset[choice])/dataset[choice].mean())  
   print("\n"+"*" *100)
   basic_stats(dataset,start_date,end_date,company_name,company)
   
  
#The function prints menu for advance Stats options    
def advanced_stats(dataset,start_date,end_date,company_name,company):
        print("\n"+"*" *100)
        print(f"\n1.Simple Moving Average for {company_name} \n2.Exponential Weighted Moving Average for {company_name} \n3.Moving Average Convergence Divergence (MACD for {company_name})\n4.Go back to previous menu")
        option = input("\nChoose option: ")
        print("\n"+"*" *100)
        try:
            if option == "1":
                
                simple_moving_avg(dataset,start_date,end_date,company_name)
                advanced_stats(dataset,start_date,end_date,company_name,company)

            elif option== "2":
                exponential_Weighted_MA(dataset,start_date,end_date,company_name)
                advanced_stats(dataset,start_date,end_date,company_name,company)
                
            elif option == "3":
                MACD(dataset,start_date,end_date,company_name)
                advanced_stats(dataset,start_date,end_date,company_name,company)
                
            elif option == "4":
                descriptive_mode(dataset,start_date,end_date,company_name,company)
        
            else:
                print("Wrong choice, Please select options 1,2,3 as per requirement")
                advanced_stats(dataset,start_date,end_date,company_name,company)
                
        except ValueError:
            print("Select from below option")
            descriptive_mode(dataset,start_date,end_date,company_name,company)


#Calculates simple moving average
def simple_moving_avg(dataset,start_date,end_date,company_name):
    try:
        window_size =int(input("Enter the n value for Moving Average i.e Window size\n"))
        if window_size >250:
                window_size =100
                
    #print(rolling.head(20))  
        rolling = dataset['Adj Close'].rolling(window=int(window_size)).mean()    
        ct.drawplot(rolling,title =f"Simple Moving Average of {company_name} for rolling window size {window_size}")
        print("Window size is not able to fit the graph")
        
    except ValueError:
      print("The window size should be a number ")
      simple_moving_avg(dataset,start_date,end_date,company_name)

    except:
        print("Window size is to small test expect result for greater range values")
        simple_moving_avg(dataset,start_date,end_date,company_name)
#Calculate Exponential Weight Average
def exponential_Weighted_MA(dataset,start_date,end_date,company_name):
    try:
    
      window_size =int(input("Enter the n value for Weighted Moving Average, i.e Window size  \n"))
      if window_size >250:
          window_size =100
          
      dataset['wm_avg'] = dataset['Adj Close'].ewm(span= window_size, min_periods=0,adjust=True, ignore_na=True).mean()
                
      ct.drawplot(dataset['wm_avg'],title =f"Exponential Weighted Moving Average  of {company_name} for window size {window_size}")
     
    except ValueError:
      print("The window size should be a number ")
      exponential_Weighted_MA(dataset,start_date,end_date,company_name)
    
    except:
        print("Window size is to small test expect result for greater range values")
        exponential_Weighted_MA(dataset,start_date,end_date,company_name)





#Calculate Moving average Convergence Divergence     
def MACD(dataset,start_date,end_date,company_name):
    close_26_ewma = dataset['Adj Close'].ewm(span=26, min_periods=0, adjust=True, ignore_na=True).mean()
    close_12_ewma = dataset['Adj Close'].ewm(span=12, min_periods=0, adjust=True, ignore_na=True).mean()
    dataset['26ema'] = close_26_ewma
    dataset['12ema'] = close_12_ewma
    dataset['MACD'] = (dataset['12ema'] - dataset['26ema'])
    ct.drawplot(dataset['MACD'],title =f"Moving Average Convergence Divergence (MACD) of {company_name}")
    
 #Compare 2 or 3 stocks and get entire analysis  
def comparestocks(dataset,start_date,end_date,company_name, company):
    tickerlist =[]
    tickerlist.append(company)
    MaxStockLimit = input("Please enter the number of companies you want to compare prices, Max limit is 3\n")
    try: 
        if(int(MaxStockLimit)<=3):
            for i in range(int(MaxStockLimit)):
                print("\n Please enter the ticker symbol for the stock(s) you want to proceed(Max 3)")
                ticker = input("(Press 1 to exit when you want to quit loading):\n").upper()
                all_symbols=  web.get_nasdaq_symbols()
                symbols = list(all_symbols.loc[:, "NASDAQ Symbol"])
                
                if ticker in symbols:
                    tickerlist.append(ticker)
                    
                else:
                    print("You have entered Invalid ticker")
                    comparestocks(dataset,start_date,end_date,company_name, company)  

        else:
          print("You have exceeded the maximum limit to compare stocks")
          comparestocks(dataset,start_date,end_date,company_name, company) 
          
    except ValueError:
        print("***Please enter integer value as 1,2 or 3***")
        comparestocks(dataset,start_date,end_date,company_name, company) 
        
    if(len(tickerlist)<=4):
        getdataForMultipleTickers(tickerlist,start_date,end_date,dataset,company_name, company)
    
 #Fetch data for  Multiple tickers one by one  
def getdataForMultipleTickers(tickerlist,start_date,end_date,dataset,company_name, company):
       
    combineddataset= web.DataReader(tickerlist,'yahoo',start_date,end_date)
    print("\n"+"*" *100)
    option = input("Choose\n1. For Basic Statistics View\n2. For Graphical View\n3. Previous Menu\n")
    print("\n"+"*" *100)
    try:
        if option == "1": 
            closeDataCombined = combineddataset['Close']
            print("Descriptive Analytics done on Close Price for Combined dataset:\n ",closeDataCombined.describe())
            getdataForMultipleTickers(tickerlist,start_date,end_date,dataset,company_name, company)
            
        elif option == "2":
            companyValues = "Multiple Companies"
            simple_moving_avg(combineddataset,start_date,end_date,companyValues)
            close = combineddataset['Close']
            close = close.fillna(method='ffill')
            ct.drawplot(close,title =f"Raw Time Series for closing price of {company}")
            getdataForMultipleTickers(tickerlist,start_date,end_date,dataset,company_name, company)
            
        elif option == "3":
            descriptive_mode(dataset,start_date,end_date,company_name,company)
            
        else:
            print("Please select correct option")
            getdataForMultipleTickers(tickerlist,start_date,end_date,dataset,company_name, company)
            
    except ValueError:
        print("Select from below the options")
        getdataForMultipleTickers(tickerlist,start_date,end_date,dataset,company_name, company)
