#Please import all the packages by pip install

import numpy as np
import matplotlib.pyplot as mp
from mpl_finance import candlestick_ohlc
import statsmodels.api as sma
import home_page as hm

#This function is basically used to plot data with different types of Matplotlib graph functionalities
def plotData(dataset,start_date,end_date,company_name,company):
   print("*" *100)
   print(f"Thanks for Choosing Descriptive Stock Analysis Option. You can choose \n1.Time Series \n2.Trend Lines for {company_name} \n3.Candle Stick View of Close price for {company_name}\n4.Histograms for High and Low \n5.Main Menu\n")
   option = input("Choose from the above option?\nEnter your Option:")
   print("*" *100)
   try: 
       if option == "1":
           timeSeries(dataset,start_date,end_date,company_name,company)
           plotData(dataset,start_date,end_date,company_name,company)
           
       elif option == "2":
           trendlines(dataset,start_date,end_date,company_name,company)
           plotData(dataset,start_date,end_date,company_name,company)
           
       elif option == "3":
           candle_stick_ohlc(dataset,start_date,end_date,company_name,company)
           plotData(dataset,start_date,end_date,company_name,company)
           
       elif option == "4":
           graph(dataset)
           plotData(dataset,start_date,end_date,company_name,company)
           
       elif option == "5":
           hm.mainmenu(dataset,start_date,end_date,company_name,company)

       else:
           print(f"Guess, You have entered the wrong option, Please try again !!! You can choose \n1.Time Series and Trend Lines for {company_name} \n2.Candle Stick View of Close price for {company_name}\n3.Main Menu\n")
           print("\n")
           plotData(dataset,start_date,end_date,company_name,company)
           
   except ValueError:
        print("***Wrong choice, Please try again....***\n\n")
        plotData(dataset,start_date,end_date,company_name,company)

         

#This function is used to plot Candle Stick graph 
def candle_stick_ohlc(data,start_date,end_date,company_name,company):
    dataset = data.copy()
    openData = dataset['Open']
    close = dataset['Close']
    high = dataset['High']
    low = dataset['Low']
    volume = dataset['Volume']
    dataset.reset_index(inplace=True,drop=False)
    try:
        dataset['Date'] = ((dataset['Date'] - dataset['Date'].min())/np.timedelta64(1,'D'))
       
    except TypeError:
        print("data format is not supported")
    date = dataset['Date']
    start_value = 0
    date_len = len(date)
    ohlc_list =[]
    ax1 = mp.subplot2grid((1,1),(0,0))
    
    while start_value<date_len:
         append_values =  date[start_value],openData[start_value], close[start_value], high[start_value], low[start_value], volume[start_value]
         ohlc_list.append(append_values)
         start_value+=1
        
    candlestick_ohlc(ax1,ohlc_list, width =0.5 , colorup='#77d879',colordown='#db3f3f')
    
    for label in ax1.xaxis.get_ticklabels():
        label.set_rotation(45)
   
    mp.xlabel('Date in Float')
    mp.ylabel(f'Open, High, Low, Close and Volume values for {company_name}')
    mp.title(f'Candle Stick Sketch for the {company_name} ')
    mp.tight_layout()
    mp.show()
    plotData(dataset,start_date,end_date,company_name,company)
    mp.close()
    mp.cla()
 
    
#Plot the timeseries graph
def timeSeries(dataset,start_date,end_date,company_name,company):    
        close = dataset['Close']
        close = close.fillna(method='ffill')
        drawplot(close,title =f"Raw Time Series for closing price of {company_name}")
        
#Plot trendline with comparison to actual values
def trendlines(dataset,start_date,end_date,company_name,company):
            newDataset = dataset.copy()
            data = sma.OLS(newDataset['Close'],sma.add_constant(range(len(newDataset['Close'].index)),prepend=True)).fit().fittedvalues
            newDataset['Close'].plot(grid =True)
            mp.plot(data,label="Trend lines of the stock_analysis")
            mp.xlabel('Date in Years')
            mp.ylabel(f'Closing Price of the stock for {company_name}')
            mp.title(f'Trend Line for Closing Price of {company_name}')
            mp.legend()
            mp.show()  

#Plot few basic histograms for hign and close values                                                     
def graph(dataset):
    mp.xlabel("Date")
    mp.ylabel("High")
    mp.hist(dataset.High)
    mp.legend()
    mp.show()
    mp.xlabel("Date")
    mp.ylabel("Low")
    mp.hist(dataset.Low)
    mp.show()

#Draw plot function plots all the graphs from any function
def drawplot(updateddataset, title):
     mp.clf()  
     mp.plot(updateddataset)
     mp.grid()
     mp.xlabel("Date")
     mp.ylabel(f"Closing Price of given company")
     mp.title(title)
     mp.legend()
     mp.show() 