#Please import all the packages by pip install

import pandas_datareader as web
import pandas as pd
import numpy as np
import datetime as dt
from datetime import timedelta
import calendar
import pandas_market_calendars as mcal
import descriptive_analytics as da
import predictive_analytics as pa
import candlestick_timeseries as ct
import twitter as tw
import warnings
from PIL import Image

#For ignoring the warnings
warnings.filterwarnings("ignore")



def getCompanyDetails(companyTicker):
    
#Checking if entered ticker is registered on NASDAQ   
    all_symbols=  web.get_nasdaq_symbols()
    company_details = pd.read_csv("http://www.nasdaq.com/screening/companies-by-name.aspx?letter=0&exchan0ge=nasdaq&render=download", usecols = [*range(0,7)])
    symbols = list(all_symbols.loc[:, "NASDAQ Symbol"])
    
# Input from the user for Ticker symbol
    if companyTicker in symbols:     
           company_details = company_details.loc[company_details['Symbol'] == companyTicker]
           company_details.index = range(len(company_details)) 
           symbol_details = all_symbols.loc[all_symbols['NASDAQ Symbol'] == companyTicker]
           symbol_details.index = range(len(symbol_details)) 
           company_name = company_details["Name"][0]
           print("*" *100)
           print("\nGeneral Information about the Company\n ")  
           print("Company  Name                  : ",company_details["Name"][0])    # Access elements such as Name, Market cap, etc from the csv for that company
           print("Last Sale Information          : ",company_details["LastSale"][0])
           print("Market Cap                     : ",company_details["MarketCap"][0])
           print("IPO years                      : ",company_details["IPOyear"][0])
           print("Sector of the company          : ",company_details["Sector"][0])
           print("Industry the company belong to : ",company_details["industry"][0])
           print("NextShares Information         : ",symbol_details["NextShares"][0])
           print("*" *100)
           return companyTicker,company_name
       
    else:
           print("Invalid Ticker Symbol.Please Re-enter Valid ticker symbol")
           main()
           
    
 #Default mode that is 52 weeks(1 Year)   
def default_mode(today_date,companyTicker,company_name):
   startDate =  today_date- timedelta(days=364)
   endDate = today_date
   getdata(startDate,endDate,companyTicker,company_name)

#Custom mode that is user can enter any range of dates
def custom_mode(companyTicker,company_name):
    dateValidation(companyTicker,company_name)

#Checks the date validation and convert from string to datetime format and pass to date range function        
def dateValidation(companyTicker,company_name):
    date_entry = input('Enter a start date in YYYY-MM-DD format:')
    isValidStartDate = validate_date_format(date_entry)
    if(isValidStartDate):
        year, month, day = map(int, date_entry.split('-'))
        date1 = dt.datetime(year, month, day)
        isValidStartDateRange = validate_date_range(date1)
        
        if(isValidStartDateRange):
            endDateValidation(date1,companyTicker,company_name)
            
        else:
           dateValidation(companyTicker,company_name)
           
    else:
       dateValidation(companyTicker,company_name)
       
   
 #It only checks user inputed end dates and validates them   
def endDateValidation(startDate,companyTicker,company_name):
    date_entry = input('Enter a end date in YYYY-MM-DD format:')
    isValidEndDate = validate_date_format(date_entry)
    year, month, day = map(int, date_entry.split('-'))
    endDate = dt.datetime(year, month, day)
    isValidEndDateRange = validate_date_range(endDate)
    if(isValidEndDate):
        year, month, day = map(int, date_entry.split('-'))
        endDate = dt.datetime(year, month, day)
        isValidEndDateRange = validate_date_range(endDate)
        
        if(isValidEndDateRange):
            
           if(startDate <= endDate):
             validateWorkingDays(startDate,endDate,companyTicker,company_name)
             
           else:
             print("Seems the start Date is greater than end Date ..!! Please enter end date past start Date")
             endDateValidation(startDate,companyTicker,company_name)
             
        else:
           endDateValidation(startDate,companyTicker,company_name)
           
    else:
        endDateValidation(startDate,companyTicker,company_name)
        
 
#It checks the format of the function which should be year-month-date   
def validate_date_format(date_string):
    try:
        if(dt.datetime.strptime(date_string, '%Y-%m-%d')):
            return True
        else:
            return False
    except ValueError:
        print("***Incorrect date format***\n")  # Catch int() exception
  

#It checks if the date range is a valid range
def validate_date_range(date_string):
   today_date = dt.datetime.now()
   #formatted_today_date = ('%s-%s-%s' % (today_date.year, today_date.month, today_date.day))
   try:
       if date_string <= today_date:
           return True
       else:
           print("Date entered seems to be future Date which is not a valid use case\n") 
           return False
   except TypeError:
        print("***Date entered seems to be future Date which is not a valid use case***\n")
   
#It checks for valid working days and skips saturdays, sundays and holiday dates
def validateWorkingDays(startDate, endDate,companyTicker,company_name):
    
   nyse = mcal.get_calendar('NYSE')
   isValidWorkingDays = False 
   isvaliddate = nyse.valid_days(startDate, endDate)
   
   if ((abs(endDate-startDate).days)) <= 1:
       start_Day = calendar.day_name[startDate.weekday()]
       
       end_Day = calendar.day_name[endDate.weekday()]
       
       if( (start_Day == "Saturday" and end_Day=="Sunday") or (start_Day == "Saturday" and end_Day=="Saturday") or (start_Day == "Sunday" and end_Day=="Sunday")):
           print("The Day of end Date:", end_Day) 
           print("The Day of start Date:", start_Day)
          
           isValidWorkingDays = False
       elif((pd.Timestamp(startDate) not in isvaliddate) and (pd.Timestamp(endDate) not in isvaliddate)):
           isValidWorkingDays = False
       else:
           isValidWorkingDays =  True
   else:
     isValidWorkingDays = True
  
   if(isValidWorkingDays):
       getdata(startDate,endDate,companyTicker,company_name)


#This function will fetch data from the website and prints it
def getdata(start_date,end_date,company,company_name):
       
    dataset= web.DataReader(company,'yahoo',start_date,end_date)
    print(f"The stock value for first 5 days {company_name} is : \n",dataset.head())
    print(f"The stock value for last 5 days {company_name} is : \n",dataset.tail())
   
    close = dataset['Close']
    print("*" *100)
    print(f"\nMaximum Close Price for {company_name}  : ",np.max(close))    # Access elements such as Name, Market cap, etc from the csv for that company
    print(f"\nMinimum Close Price for {company_name}  : ",np.min(close))
    print("*" *100)
 
    mainmenu(dataset,start_date,end_date,company_name,company)

#The function where Descriptive, Predective, Visulisation, Polarity detection works, also this menu has a option to go back to main menu so the user **
#** can traverse between menus fro performing any kind of operations
    
def mainmenu(dataset,start_date,end_date,company_name,company):
    print("*" *100)
    print("\nWelcome to Stock Market Analysis")
    print(f"1) Stock Analysis for {company_name}")
    print(f"2) Stock Prediction for {company_name}")
    print(f"3) Visualisation for {company_name}")
    print(f"4) Twitter sentiment Analysis for {company_name}")
    print("5) Home Menu to Change the Mode")
    print("*" *100)

    question = input("\nFrom the given options select any?\n")
    try:
        
        
        if question == "1":
            da.descriptive_mode(dataset,start_date,end_date,company_name,company)
           
        elif question == "2":   
            pa.predictive_mode(dataset,start_date,end_date,company_name,company) 
            
        elif question == "4":
            tw.semantic(company_name)
            mainmenu(dataset,start_date,end_date,company_name,company)
            
        elif question == "3":
            ct.plotData(dataset,start_date,end_date,company_name,company)
            mainmenu(dataset,start_date,end_date,company_name,company)
            
        elif question == "5":
            menu(company,company_name)
            
        else:
            print("Invalid Option. Please Re-enter the option you would like to choose. \n")
            print("\n")
            mainmenu(dataset,start_date,end_date,company_name,company)
            
    except ValueError:
        print("***Invalid Option. Please Re-enter the option you would like to choose.***\n")
        mainmenu(dataset,start_date,end_date,company_name,company)


#Enter ticker symbol for which you want the data to be printed or analyzed        
def main():
         tickerSymbol = input("Please enter Company Ticker: ").upper()
         companyTicker, company_name = getCompanyDetails(tickerSymbol)
         menu(companyTicker, company_name)
         
#Menu for default mode or custom mode and user can only exit the code from this menu if code does not break anywhere
def menu(companyTicker,company_name):
         print("*" *100)
         print("\t\tWe do have more functionalities to explore.")
         print("\nChoose from the below option")
         print("1. Analysis for 1 year(Default mode)\n2. Variable Date Range(Custom mode)\n3. Check for another Company\n4. Exit")
         print("*" *100)
         question = input("\nFrom the given options select any?\n")
         try:
             if question == "1":
                 print("Welcome to 1 year Analysis(Default mode)\n")
                 today_date = dt.datetime.now()
                 default_mode(today_date,companyTicker,company_name)
                 
             elif question == "2":
                 print("Welcome to Custom Mode(Variable Date Range)\n")
                 custom_mode(companyTicker,company_name)
                 
             elif question == "3":
                 main()
                 
             elif question == "4":
                 exit()
                 
             else:
                 print("Invalid Option. Please Re-enter the option you would like to choose. \n")
                 menu(companyTicker,company_name)
          
         except ValueError:
            print("***Invalid Option. Please Re-enter the option you would like to choose.***\n")
            menu(companyTicker,company_name)

       
   
#Beginning of the code        
if __name__ == "__main__":
    print("\n"+"*" *100)
    print("\t\tWelcome to The Data Whisperers Stock Market Analysis Project")
    print("*" *100)
    option = input("Do you want the flow chart for the entire project please Press 1 or press anything to Continue with the normal flow\n")
    try:
        if option == "1":
            img = Image.open('Flow_Chart.jpg')
            img.show()
    except ValueError:
        print("Please enter  correct choice Y/N")
    main()
    
