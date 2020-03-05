import pandas as pd
import datetime as datetime
from twilio.rest import Client
import os

dir=os.environ['DOCPATH'] + '/Birthdays'
os.chdir(dir)

account_sid = os.environ['TWILIO_SID']
auth_token = os.environ['TWILIO_AUTH_TOKEN']
twilio_num = os.environ['TWILIO_NUM']
current_num = os.environ['CURRENT_NUM']

#Twilio creds
account_sid = account_sid
auth_token = auth_token
client = Client(account_sid, auth_token)

#Current Date
vToday = datetime.datetime.now()
vCurrentMonth = vToday.month
vCurrentDay = vToday.day

#Messages for different people
def bdayMessages(Person):
    message = Person +"""'s Birthday Today"""
    return message

#birthdayArr = []
#phoneNums = []
df = pd.read_csv(r'BirthdayDates.csv')
df['NumberWithPlus'] = df['Number'].map(lambda x: '+' + str(x))

for i in range(len(df)):
    if df.loc[i,'Month']==vCurrentMonth and df.loc[i,'Day']==vCurrentDay:
        #birthdayArr.append(df.loc[i,'Person'])
        #phoneNums.append(df.loc[i,'Numbers'])
        birthdayPerson = df.loc[i,'Person']
        phoneNum = df.loc[i,'NumberWithPlus']

        message = client.messages \
                        .create(
                             body=bdayMessages(birthdayPerson),
                             from_=twilio_num,
                             #to=phoneNum
                             to=current_num
                         )
