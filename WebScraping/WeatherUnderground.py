'''
Web-scrape historical weather data from Weather Underground and dump the data into database on laptop
'''

import pandas as pd
import sqlite3
from bs4 import BeautifulSoup
import csv
import datetime
import os
import requests
import numpy as np

def createTable():

	conn = sqlite3.connect('weather.db')
	c=conn.cursor()
	
	#This needs to correspond to the dataframe generated below in the getWeatherData function
    #The columns need to be the same as the columns below
	c.execute('''CREATE TABLE historical_weather2 
				(Time CHAR(10), Temp DECIMAL(3,1), Heat_Cool_Index DECIMAL(3,1), 
                Dew_Point DECIMAL(2,1), Humidity INT, Pressure DECIMAL(3,2), Visibility DECIMAL(3,1), 
                Wind_Dir CHAR(20), Wind_Speed  CHAR(20), Gust_Speed CHAR(20),  Precip CHAR(30),
                Events CHAR(30), Conditions CHAR(30), Date DATE, Key CHAR(4), Hour INT, df_index INT) ''')

	conn.commit()

	conn.close()
                  
def insertData(df):
    
    conn = sqlite3.connect('weather.db')
    c=conn.cursor()
    df.to_sql('historical_weather2', conn, if_exists = 'append',index=False)
    conn.commit()
    conn.close

def cleanData(weatherdf):
    #Want to clean up the data scraped from WU
    #All variables will need to be re-coded as missing if the entries are '-' or 'N/A'
    
    weatherdf.ix[weatherdf.ix[:,'Temp'].apply(lambda x: x == '\n  -\n'),'Temp'] = 999
    weatherdf.ix[weatherdf.ix[:,'Heat_Cool_Index'].apply(lambda x: x == '\n  -\n'),'Heat_Cool_Index'] = 999
    weatherdf.ix[weatherdf.ix[:,'Dew_Point'].apply(lambda x: x == '\n  -\n'),'Dew_Point'] = 999
    weatherdf.ix[weatherdf.ix[:,'Humidity'].apply(lambda x: x == 'N/A%'),'Humidity'] = 999
    weatherdf.ix[weatherdf.ix[:,'Pressure'].apply(lambda x: x == '\n  -\n'),'Pressure'] = 999
    weatherdf.ix[weatherdf.ix[:,'Visibility'].apply(lambda x: x == '\n  -\n'),'Visibility'] = 999
    weatherdf.ix[weatherdf.ix[:,'Wind_Dir'].apply(lambda x: x == '\n  -\n'),'Wind_Dir'] = '999'
    weatherdf.ix[weatherdf.ix[:,'Wind_Speed'].apply(lambda x: x == '\n  -\n'),'Wind_Speed'] = '999'
    weatherdf.ix[weatherdf.ix[:,'Gust_Speed'].apply(lambda x: x == '\n  -\n'),'Gust_Speed'] = '999'
    weatherdf.ix[weatherdf.ix[:,'Precip'].apply(lambda x: x == 'N/A'),'Precip'] = '999'
    
    #I don't know if a better way to do this. The above are quick vectorized methods
    for i in range(len(weatherdf)):
    
        if weatherdf.ix[i,'Events'].strip() == '':
            weatherdf.ix[i,'Events']='999'
        else:   
            dummy='do nothing'
            
            
    for i in range(len(weatherdf)):
        if weatherdf.ix[i,'Humidity']==999:
            dummy = 'do nothing'
        else:
            weatherdf.ix[i,'Humidity'] = int(weatherdf.ix[i,'Humidity'][:-1])
            
    return weatherdf
  
def getWeatherData(key, counter):

    weatherData = []
        
    #These need to be strings
    year  = str(counter.year)
    month = str(counter.month)
    day   = str(counter.day)
    date = year+"-"+month+"-"+day            
    
    #Contruct the url to direct us to the WU webpage
    url = 'http://www.wunderground.com/history/airport/'+key+'/'+year+'/'+month+'/'+day+'/DailyHistory.html'
    r = requests.get(url)
    c = r.content
    soup = BeautifulSoup(c, 'lxml') 
    
    #Get headers for the table: these are denoted as 'th' in the table
    table = soup.find("div", id="observations_details")
    headers = table.findAll('th')
    
    #WINDCHILL/HEAT-INDEX
    #If there's a column for windchill or heat-index, then our resulting dataframe result will be larger
    #Split these cases into two: one where we have these columns and then another case where we don't
    if headers[2].string == 'Windchill' or headers[2].string == 'Heat Index':
        
        table = soup.find("div", id="observations_details")
        table_body = table.find('tbody')
        rows = table_body.findAll('tr')
        row_len = len(rows)

        #This gets the rows into an array. Each element in the array is a 1x13 vector 
        row_data = []
        for i in range(len(rows)):
            row_data.append(rows[i].findAll('td'))
            
        #This gets the number data into an array. Each row in the array corresponds to the numerical data for that hour
        spans = []
        for i in range(len(rows)):
            spans.append(rows[i].findAll('span',attrs={'class':'wx-value'}))
            
        #need to create an 24x13 matrix where each row has the numerical data from spans and the other data from row_data
        #ACTUALLY this won't work because the data often has multiple entries per hour. So i need a way to dynamically
        #determine the length of the matrix. The width will always be 13 if there's a heat_index/wind_chill column
        matrix = [[0 for k in range(13)] for m in range(row_len)]
        
        #need this "counter" variable here due to the numerical data being included in the "span" objects in
        #this html code. The ind iterates through the number of numerical data in the spans array
        ind = 0
        for i in range(row_len):
            ind = 0
            for j in range(13):
                
                if row_data[i][j].string == None:
                    #this is numerical data.
                    #need to figure out how to access the spans array to pick out the right element
                    matrix[i][j] = spans[i][ind].string
                    ind += 1
                    
                else:
                    #this is non-numerical data (non-spans)
                    matrix[i][j] = row_data[i][j].string
                    
        cols = ['Time', 'Temp', 'Heat_Cool_Index', 'Dew_Point', 'Humidity', 'Pressure', 'Visibility', 'Wind_Dir', 'Wind_Speed', 'Gust_Speed', 'Precip', 'Events', 'Conditions']    
        df = pd.DataFrame(matrix, columns=cols)
        df['Date'] = date
        
        return df
              
    else:
        #put code here that adds NaN to the appropriate column
        table = soup.find("div", id="observations_details")
        table_body = table.find('tbody')
        rows = table_body.findAll('tr')
        row_len = len(rows)

        #This gets the rows into an array. Each element in the array is a 1x13 vector 
        row_data = []
        for i in range(len(rows)):
            row_data.append(rows[i].findAll('td'))
            
        #This gets the number data into an array. Each row in the array corresponds to the numerical data for that hour
        spans = []
        for i in range(len(rows)):
            spans.append(rows[i].findAll('span',attrs={'class':'wx-value'}))
            
        #need to create an 24x13 matrix where each row has the numerical data from spans and the other data from row_data
        #ACTUALLY this won't work because the data often has multiple entries per hour. So i need a way to dynamically
        #determine the length of the matrix. The width will always be 13 if there's a heat_index/wind_chill column
        matrix = [[0 for k in range(12)] for m in range(row_len)]
        
        #need this "counter" variable here due to the numerical data being included in the "span" objects in
        #this html code. The ind iterates through the number of numerical data in the spans array
        ind = 0
        for i in range(row_len):
            ind = 0
            for j in range(12):
                
                if row_data[i][j].string == None:
            
                    #this is numerical data.
                    #need to figure out how to access the spans array to pick out the right element
                    matrix[i][j] = spans[i][ind].string
                    ind += 1
                
                else:
                    #this is non-numerical data (non-spans)
                    matrix[i][j] = row_data[i][j].string
                        
        cols = ['Time', 'Temp', 'Dew_Point', 'Humidity', 'Pressure', 'Visibility', 'Wind_Dir', 'Wind_Speed', 'Gust_Speed', 'Precip', 'Events', 'Conditions']    
        df = pd.DataFrame(matrix, columns=cols)
        df['date'] = date
        
        new_cols = ['Time', 'Temp', 'Heat_Cool_Index', 'Dew_Point', 'Humidity', 'Pressure', 'Visibility', 'Wind_Dir', 'Wind_Speed', 'Gust_Speed', 'Precip', 'Events', 'Conditions','date']    
        df_reindex = df.reindex(columns=new_cols)
        
        return df_reindex    
    
def main():
    
    flag = 0
    while flag == 0:
        
        userBegDate = raw_input("Please enter beginning date (mm/dd/yyyy)  ")
        userEndDate = raw_input("Please enter ending date (mm/dd/yyyy)  ")
        locKey      = raw_input("Please enter the location Key. (e.g. KLAX)  ")
    
        begSplit = userBegDate.split("/")
        endSplit = userEndDate.split("/")

        userBegMonth = int(begSplit[0])
        userBegDay   = int(begSplit[1])
        userBegYear  = int(begSplit[2])
        userEndMonth = int(endSplit[0])
        userEndDay   = int(endSplit[1])
        userEndYear  = int(endSplit[2])

        d1 = datetime.date(userBegYear, userBegMonth, userBegDay)
        d2 = datetime.date(userEndYear, userEndMonth, userEndDay)
        
        one_day = datetime.timedelta(days=1)
        
        if d1 > d2:
            print "Date Input is invalid, try again"
        else:
            flag = 1
        
        ### Begin loop that gets data and appends it to the weatherData DF
        ### Loop over days that start with d1 (counter) and ends at d2 
        weatherData = []
        
        #counter is just a datetime object. But it is used to iterate through
        #the while loop below. Also, by copying d1 into another variable, I don't
        #mess with d1 which was entered by the user
        counter = d1
        while counter <= d2:

            #The getWeatherData function should return a dataframe that I then
            #dump into a database. 
            df_weather = getWeatherData(locKey, counter)
            
            #add the a Key column for easy lookup in Weather table
            df_weather["Key"] = locKey
            df_weather['Hour'] = 0
            for i in range(len(df_weather)):
                try:
                    df_weather.ix[i,'Hour'] = int(df_weather.ix[i,'Time'][:2])
                except ValueError:
                    df_weather.ix[i,'Hour'] = int(df_weather.ix[i,'Time'][:1])
                    
            df_weather['df_index'] = df_weather.index.values
            cleanData(df_weather)
            insertData(df_weather)
            
            counter+=one_day
             
main()
