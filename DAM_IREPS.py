#!/usr/bin/env python3

# PyScript for extracting the required data from all the Members of the IREPS for Gurteen.
#Justin Rutherford dated 2.2.22

#####################################################################
####  IMPORT PACKAGES ###############################################

# For creating new directories
import os
# import shutil for removing directories
import shutil
# For dealing with arrays.
from ssl import MemoryBIO
import numpy as np
# For dealing with data frames.
import pandas as pd
# For dealing with IMAP email servers.
import imaplib
# For parsing emails.
import email
import email.mime.multipart
import email.mime.text
# For sending emails.
import smtplib
# For pretending strings are buffers.
import io
# For logging.
import logging as log
import sys
# For dates and times.
import datetime
from datetime import date, time
from datetime import timedelta
# to merge multiple dataframes efficiently
from functools import reduce
# To produce graphics
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
#To train our machine learning model
import sklearn
# For interpolating.
import scipy.interpolate

############################################################################################
########### SET GLOBAL PARAMETERS  #########################################################

# Hardcode the Power Capacity for graphic illustration
capacity_factor = 2310

#To switch on/off the header variable
HEADER_VAR = False

#To set the leadtime hour from the Issue time for the starting period
#LEAD = 23 # if using the FCT_Run(Z) = 00
LEAD = 17 # If using the FCT _Run(Z) = 06 - which is the closest forecast to the 11am gate closure.
#LEAD = 11 # if using the FCT_Run(12Z) = 00

# For saving files with today's date
today = datetime.date.today()
today = today.strftime("%d-%m-%Y")

#Include the relevant date on the graphics
tomorrow = datetime.date.today() + datetime.timedelta(days=1)
tomorrow = tomorrow.strftime("%d-%m-%Y")

# To format the dates on the graphics
myFmt = mdates.DateFormatter('%H:%M')

#set out the path for the saving of the day-only csv's
daily_path = ("/home/justi/Desktop/NewGurteenForecast/Daily/" + today)
if not os.path.exists(daily_path):
  os.makedirs(daily_path)
else:
  shutil.rmtree(daily_path)
  os.makedirs(daily_path)

#set out the path for the saving of the images
img_path = ("/home/justi/Desktop/NewGurteenForecast/img/" + today)
if not os.path.exists(img_path):
  os.makedirs(img_path)
else:
  shutil.rmtree(img_path)
  os.makedirs(img_path)


# os.chdir(daily_path)
# newfolder = tomorrow
# os.makedirs(newfolder)
# # Change back directory to the working directory
# os.chdir("/home/justi/Desktop/NewGurteenForecast/")

# Log to a file as well as the screen.
logger = log.getLogger('')
logger.setLevel(log.INFO)
fh = log.FileHandler('affiliate.log')
sh = log.StreamHandler(sys.stdout)
formatter = log.Formatter('%(asctime)s:%(filename)s:%(message)s',
                               datefmt='%a, %d %b %Y %H:%M:%S')
fh.setFormatter(formatter)
sh.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(sh)


#####################################################################
## WE ACCESS THE IREPS FORECAST FIRST ###############################

## Email username and password.

username = r'gurteenforecast1'
password = r'faahokgkcsbwgwxf'
#####################################################################


######################################################################
## Check for email ###################################################

# Connect to the server.
log.info("Connecting to IMAP server.")
mail = imaplib.IMAP4_SSL(r'imap.gmail.com')
mail.login(username, password)
log.info("Connected to IMAP server.")

######################################################################
## ## Select the Required Folder #####################################

mail.select(r'Member000')
log.info("Member000 Folder selected.")

# Get the list of email ids.
_, emaillist = mail.uid(r'search', None, r'(X-GM-RAW " has:attachment newer_than:1d")')

# print(emaillist)

# Pick the latest id.
latestid = emaillist[0].split()[-1]
# Fetch that email.
_, raw_email = mail.uid(r'fetch', latestid, r'(RFC822)')
raw_email = raw_email[0][1].decode('utf-8')
#log.info("Latest email selected.")

# ################################################################################
# ## Extract CSV from email.
# ################################################################################
csv_file = None
# Parse the email
met_email = email.message_from_string(raw_email)
for part in met_email.walk():
    # multipart/* are just containers
    if part.get_content_maintype() == 'multipart':
        continue
    filename = part.get_filename()
    if filename:
        log.info("Found filename '" + filename + "' in email.")
        csv_file = part.get_payload(decode=True)

# Make sure we got something.
assert csv_file, "Failed to extract CSV file from the email."
#log.info("Found CSV file.")

# ################################################################################
# ## Parse the CSV file.
# ################################################################################

# Pretend csv_file is a file, as opposed to a string.
with io.StringIO(csv_file.decode('utf-8')) as f:
  df =  pd.read_csv(f, usecols=[0,1,2,3,4],header = 0, names=['Date', 'FCT_Run(Z)', 'Param(10m)', 'Time(Hr)', 'Value(m/s)'])

#Ensure that the date is in the correct format
df['Date'] = pd.to_datetime(df['Date'])
#Identify the Issue Time
df['IssueTime'] = df.iloc[:, 1]
#As the time is in integer formate we need to change and convert
df['IssueTime'] = (df['IssueTime']/100).astype(int)
df['IssueTime'] = pd.to_datetime(df['IssueTime'],format= '%H')
#We only want the time element
df['IssueTime']= df['IssueTime'].dt.time
#We need to establish a relevant datetime variable for the Issue
df['DateIssueTime'] = pd.to_datetime(df['Date'].astype(str)+ ' ' + df['IssueTime'].astype(str))
#Now we need to establish the leadtime
df['Leadtime']= pd.to_timedelta(df['Time(Hr)'], 'h')
#Apply a relevant datetime to each leadtime
df['DateTime'] = df['DateIssueTime'] + df['Leadtime']


#################################################################################
# Select only the dates we need
#List out the unique dates
dates = df['DateTime'].unique()
# Get the first date and set the time to 11pm.
start_date = dates.astype('datetime64[ns]')[0] + np.timedelta64(LEAD, 'h')
print(start_date)
# Get the end time by adding 24hrs to start_date.
end_date = start_date + np.timedelta64(24, 'h')

print(end_date)

# Log the start and end datetimes.
log.info("Datetimes between " + str(start_date) + " and " + str(end_date))
# Create a mask to select dates from start_date (incl) to end_date (excl).
mask = (df['DateTime'] >= start_date) & (df['DateTime'] < end_date)
# Get the records.
df = df.loc[mask]

# Target the variables we want
dfU = df.loc[df['Param(10m)']== 'u']
dfU = dfU[['DateTime', 'Value(m/s)']]
dfU.columns = ['DateTime', 'U_10']
dfU.columns = ['DateTime', 'U_10']
dfU = dfU.set_index('DateTime')
#print(dfU.head())

dfV = df.loc[df['Param(10m)']== 'v']
dfV = dfV[['DateTime', 'Value(m/s)']]
dfV.columns = ['DateTime', 'V_10']
dfV= dfV.set_index('DateTime')
#print(dfV.head())

dfUG = df.loc[df['Param(10m)']== 'ugst']
dfUG = dfUG[['DateTime', 'Value(m/s)']]
dfUG.columns = ['DateTime', 'UG_10']
dfUG = dfUG.set_index('DateTime')
#print(dfUG.head())

dfVG = df.loc[df['Param(10m)']== 'vgst']
dfVG = dfVG[['DateTime', 'Value(m/s)']]
dfVG.columns = ['DateTime', 'VG_10']
dfVG = dfVG.set_index('DateTime')
#print(dfVG.head())

dfWDIR = df.loc[df['Param(10m)']== 'wdir']
dfWDIR = dfWDIR[['DateTime', 'Value(m/s)']]
dfWDIR.columns = ['DateTime', 'WDir_10']
dfWDIR = dfWDIR.set_index('DateTime')
#print(dfWDIR.head())

# Merge the subframes into one dataframe
df_merged = dfU.merge(dfV, how = 'outer', left_index=True, right_index=True)
df_merged = df_merged.merge(dfUG, how = 'outer', left_index=True, right_index=True)
df_merged = df_merged.merge(dfVG, how = 'outer', left_index=True, right_index=True)
df_merged = df_merged.merge(dfWDIR, how = 'outer', left_index=True, right_index=True)
#print(df_merged.head())

# Add the extra calculations.
df_merged['10mWS'] = np.sqrt(df_merged['U_10'].pow(2) + df_merged['V_10'].pow(2))
df_merged['10mWSg'] = np.sqrt(df_merged['UG_10'].pow(2) + df_merged['VG_10'].pow(2))
df_merged = df_merged.round(decimals=2)

#If running on Ubuntu in the Azure Cloud or Ubuntu on the local machine use the following for storing the members.
df_merged.to_csv(r"/home/justi/Desktop/NewGurteenForecast/IREPS/DAM/DAM_IREPS_Forecasts/Member000_Converted.csv", mode = 'a', index = True, header = HEADER_VAR)

log.info("Exported Member000 with Additional calculations as CSV file successfully.")

#print(df_merged)

M0 = df_merged[['U_10', 'V_10','10mWS']]
M0.rename(columns={'V_10':'M0_V10','U_10':'M0_U10','10mWS': 'M0_10mWS'}, inplace=True)
#print(M0)

# #######################################################################
# ## Select the Required Folder
# ################################################################################

mail.select(r'Member001')
log.info("Member001 Folder selected.")

# Get the list of email ids.
_, emaillist = mail.uid(r'search', None, r'(X-GM-RAW " has:attachment newer_than:1d")')

# print(emaillist)

# Pick the latest id.
latestid = emaillist[0].split()[-1]
# Fetch that email.
_, raw_email = mail.uid(r'fetch', latestid, r'(RFC822)')
raw_email = raw_email[0][1].decode('utf-8')
#log.info("Latest email selected.")

# ################################################################################
# ## Extract CSV from email.
# ################################################################################
csv_file = None
# Parse the email
met_email = email.message_from_string(raw_email)
for part in met_email.walk():
    # multipart/* are just containers
    if part.get_content_maintype() == 'multipart':
        continue
    filename = part.get_filename()
    if filename:
        log.info("Found filename '" + filename + "' in email.")
        csv_file = part.get_payload(decode=True)

# Make sure we got something.
assert csv_file, "Failed to extract CSV file from the email."
#log.info("Found CSV file.")

# ################################################################################
# ## Parse the CSV file.
# ################################################################################

# Pretend csv_file is a file, as opposed to a string.
with io.StringIO(csv_file.decode('utf-8')) as f:
  df =  pd.read_csv(f, usecols=[0,1,2,3,4],header = 0, names=['Date', 'FCT_Run(Z)', 'Param(10m)', 'Time(Hr)', 'Value(m/s)'])

#Ensure that the date is in the correct format
df['Date'] = pd.to_datetime(df['Date'])
#Identify the Issue Time
df['IssueTime'] = df.iloc[:, 1]
#As the time is in integer formate we need to change and convert
df['IssueTime'] = (df['IssueTime']/100).astype(int)
df['IssueTime'] = pd.to_datetime(df['IssueTime'],format= '%H')
#We only want the time element
df['IssueTime']= df['IssueTime'].dt.time
#We need to establish a relevant datetime variable for the Issue
df['DateIssueTime'] = pd.to_datetime(df['Date'].astype(str)+ ' ' + df['IssueTime'].astype(str))
#Now we need to establish the leadtime
df['Leadtime']= pd.to_timedelta(df['Time(Hr)'], 'h')
#Apply a relevant datetime to each leadtime
df['DateTime'] = df['DateIssueTime'] + df['Leadtime']

#################################################################################
# Select only the dates we need
#List out the unique dates
dates = df['DateTime'].unique()
# Get the first date and set the time to 11pm.
start_date = dates.astype('datetime64[ns]')[0] + np.timedelta64(LEAD, 'h')
print(start_date)
# Get the end time by adding 24hrs to start_date.
end_date = start_date + np.timedelta64(24, 'h')

#print(end_date)

# Log the start and end datetimes.
log.info("Datetimes between " + str(start_date) + " and " + str(end_date))
# Create a mask to select dates from start_date (incl) to end_date (excl).
mask = (df['DateTime'] >= start_date) & (df['DateTime'] < end_date)
# Get the records.
df = df.loc[mask]

# Target the variables we want
dfU = df.loc[df['Param(10m)']== 'u']
dfU = dfU[['DateTime', 'Value(m/s)']]
dfU.columns = ['DateTime', 'U_10']
dfU.columns = ['DateTime', 'U_10']
dfU = dfU.set_index('DateTime')
#print(dfU.head())

dfV = df.loc[df['Param(10m)']== 'v']
dfV = dfV[['DateTime', 'Value(m/s)']]
dfV.columns = ['DateTime', 'V_10']
dfV= dfV.set_index('DateTime')
#print(dfV.head())

dfUG = df.loc[df['Param(10m)']== 'ugst']
dfUG = dfUG[['DateTime', 'Value(m/s)']]
dfUG.columns = ['DateTime', 'UG_10']
dfUG = dfUG.set_index('DateTime')
#print(dfUG.head())

dfVG = df.loc[df['Param(10m)']== 'vgst']
dfVG = dfVG[['DateTime', 'Value(m/s)']]
dfVG.columns = ['DateTime', 'VG_10']
dfVG = dfVG.set_index('DateTime')
#print(dfVG.head())

dfWDIR = df.loc[df['Param(10m)']== 'wdir']
dfWDIR = dfWDIR[['DateTime', 'Value(m/s)']]
dfWDIR.columns = ['DateTime', 'WDir_10']
dfWDIR = dfWDIR.set_index('DateTime')
#print(dfWDIR.head())

# Merge the subframes into one dataframe
df_merged = dfU.merge(dfV, how = 'outer', left_index=True, right_index=True)
df_merged = df_merged.merge(dfUG, how = 'outer', left_index=True, right_index=True)
df_merged = df_merged.merge(dfVG, how = 'outer', left_index=True, right_index=True)
df_merged = df_merged.merge(dfWDIR, how = 'outer', left_index=True, right_index=True)
#print(df_merged.head()

# Add the extra calculations.
df_merged['10mWS'] = np.sqrt(df_merged['U_10'].pow(2) + df_merged['V_10'].pow(2))
df_merged['10mWSg'] = np.sqrt(df_merged['UG_10'].pow(2) + df_merged['VG_10'].pow(2))
df_merged = df_merged.round(decimals=2)

df_merged.to_csv(r"/home/justi/Desktop/NewGurteenForecast/IREPS/DAM/DAM_IREPS_Forecasts/Member001_Converted.csv", mode = 'a', index = True, header = HEADER_VAR)
#df_merged.to_csv(r"Data\Members_Aggregated\Member001_Converted.csv", mode = 'a', index = True, header = HEADER_VAR)

log.info("Exported Member001 with Additional calculations as CSV file successfully.")

M1 = df_merged[['U_10', 'V_10','10mWS']]
M1.rename(columns={'V_10':'M1_V10','U_10':'M1_U10','10mWS': 'M1_10mWS'}, inplace=True)


#######################################################################################
######## MEMBER002#####################################################################
#######################################################################################

mail.select(r'Member002')
log.info("Member002 Folder selected.")

# Get the list of email ids.
_, emaillist = mail.uid(r'search', None, r'(X-GM-RAW " has:attachment newer_than:1d")')

# print(emaillist)

# Pick the latest id.
latestid = emaillist[0].split()[-1]
# Fetch that email.
_, raw_email = mail.uid(r'fetch', latestid, r'(RFC822)')
raw_email = raw_email[0][1].decode('utf-8')
log.info("Latest email selected.")

csv_file = None
# Parse the email
met_email = email.message_from_string(raw_email)
for part in met_email.walk():
    # multipart/* are just containers
    if part.get_content_maintype() == 'multipart':
        continue
    filename = part.get_filename()
    if filename:
        log.info("Found filename '" + filename + "' in email.")
        csv_file = part.get_payload(decode=True)

# Make sure we got something.
assert csv_file, "Failed to extract CSV file from the email."
log.info("Found CSV file.")

###################################################################################

# Pretend csv_file is a file, as opposed to a string.
with io.StringIO(csv_file.decode('utf-8')) as f:
  df =  pd.read_csv(f, usecols=[0,1,2,3,4],header = 0, names=['Date', 'FCT_Run(Z)', 'Param(10m)', 'Time(Hr)', 'Value(m/s)'])

#Ensure that the date is in the correct format
df['Date'] = pd.to_datetime(df['Date'])
#Identify the Issue Time
df['IssueTime'] = df.iloc[:, 1]
#As the time is in integer formate we need to change and convert
df['IssueTime'] = (df['IssueTime']/100).astype(int)
df['IssueTime'] = pd.to_datetime(df['IssueTime'],format= '%H')
#We only want the time element
df['IssueTime']= df['IssueTime'].dt.time
#We need to establish a relevant datetime variable for the Issue
df['DateIssueTime'] = pd.to_datetime(df['Date'].astype(str)+ ' ' + df['IssueTime'].astype(str))
#Now we need to establish the leadtime
df['Leadtime']= pd.to_timedelta(df['Time(Hr)'], 'h')
#Apply a relevant datetime to each leadtime
df['DateTime'] = df['DateIssueTime'] + df['Leadtime']



#################################################################################
# Select only the dates we need
#List out the unique dates
dates = df['DateTime'].unique()
# Get the first date and set the time to 11pm.
start_date = dates.astype('datetime64[ns]')[0] + np.timedelta64(LEAD, 'h')
# Get the end time by adding 24hrs to start_date.
end_date = start_date + np.timedelta64(24, 'h')
# Log the start and end datetimes.
log.info("Datetimes between " + str(start_date) + " and " + str(end_date))
# Create a mask to select dates from start_date (incl) to end_date (excl).
mask = (df['DateTime'] >= start_date) & (df['DateTime'] < end_date)
# Get the records.
df = df.loc[mask]
#################################################################################

# Target the variables we want
dfU = df.loc[df['Param(10m)']== 'u']
dfU = dfU[['DateTime', 'Value(m/s)']]
dfU.columns = ['DateTime', 'U_10']
dfU.columns = ['DateTime', 'U_10']
dfU = dfU.set_index('DateTime')
#print(dfU.head())

dfV = df.loc[df['Param(10m)']== 'v']
dfV = dfV[['DateTime', 'Value(m/s)']]
dfV.columns = ['DateTime', 'V_10']
dfV= dfV.set_index('DateTime')
#print(dfV.head())

dfUG = df.loc[df['Param(10m)']== 'ugst']
dfUG = dfUG[['DateTime', 'Value(m/s)']]
dfUG.columns = ['DateTime', 'UG_10']
dfUG = dfUG.set_index('DateTime')
#print(dfUG.head())

dfVG = df.loc[df['Param(10m)']== 'vgst']
dfVG = dfVG[['DateTime', 'Value(m/s)']]
dfVG.columns = ['DateTime', 'VG_10']
dfVG = dfVG.set_index('DateTime')
#print(dfVG.head())

dfWDIR = df.loc[df['Param(10m)']== 'wdir']
dfWDIR = dfWDIR[['DateTime', 'Value(m/s)']]
dfWDIR.columns = ['DateTime', 'WDir_10']
dfWDIR = dfWDIR.set_index('DateTime')
#print(dfWDIR.head())

# Merge the subframes into one dataframe
df_merged = dfU.merge(dfV, how = 'outer', left_index=True, right_index=True)
df_merged = df_merged.merge(dfUG, how = 'outer', left_index=True, right_index=True)
df_merged = df_merged.merge(dfVG, how = 'outer', left_index=True, right_index=True)
df_merged = df_merged.merge(dfWDIR, how = 'outer', left_index=True, right_index=True)
#print(df_merged.head()

# Add the extra calculations.
df_merged['10mWS'] = np.sqrt(df_merged['U_10'].pow(2) + df_merged['V_10'].pow(2))
df_merged['10mWSg'] = np.sqrt(df_merged['UG_10'].pow(2) + df_merged['VG_10'].pow(2))
df_merged = df_merged.round(decimals=2)

df_merged.to_csv(r"/home/justi/Desktop/NewGurteenForecast/IREPS/DAM/DAM_IREPS_Forecasts/Member002_Converted.csv", mode = 'a', index = True, header = HEADER_VAR)
#df_merged.to_csv(r"Data\Members_Aggregated\Member002_Converted.csv", mode = 'a', index = True, header = HEADER_VAR)

log.info("Exported Member002 as CSV file successfully.")

M2 = df_merged[['U_10', 'V_10','10mWS']]
M2.rename(columns={'V_10':'M2_V10','U_10':'M2_U10','10mWS': 'M2_10mWS'}, inplace=True)

#######################################################################################
######## MEMBER003#####################################################################
#######################################################################################

mail.select(r'Member003')
log.info("Member003 Folder selected.")

# Get the list of email ids.
_, emaillist = mail.uid(r'search', None, r'(X-GM-RAW " has:attachment newer_than:1d")')

# print(emaillist)

# Pick the latest id.
latestid = emaillist[0].split()[-1]
# Fetch that email.
_, raw_email = mail.uid(r'fetch', latestid, r'(RFC822)')
raw_email = raw_email[0][1].decode('utf-8')
log.info("Latest email selected.")

csv_file = None
# Parse the email
met_email = email.message_from_string(raw_email)
for part in met_email.walk():
    # multipart/* are just containers
    if part.get_content_maintype() == 'multipart':
        continue
    filename = part.get_filename()
    if filename:
        log.info("Found filename '" + filename + "' in email.")
        csv_file = part.get_payload(decode=True)

# Make sure we got something.
assert csv_file, "Failed to extract CSV file from the email."
log.info("Found CSV file.")

##################################################################################
# Pretend csv_file is a file, as opposed to a string.
with io.StringIO(csv_file.decode('utf-8')) as f:
  df =  pd.read_csv(f, usecols=[0,1,2,3,4],header = 0, names=['Date', 'FCT_Run(Z)', 'Param(10m)', 'Time(Hr)', 'Value(m/s)'])

#Ensure that the date is in the correct format
df['Date'] = pd.to_datetime(df['Date'])
#Identify the Issue Time
df['IssueTime'] = df.iloc[:, 1]
#As the time is in integer formate we need to change and convert
df['IssueTime'] = (df['IssueTime']/100).astype(int)
df['IssueTime'] = pd.to_datetime(df['IssueTime'],format= '%H')
#We only want the time element
df['IssueTime']= df['IssueTime'].dt.time
#We need to establish a relevant datetime variable for the Issue
df['DateIssueTime'] = pd.to_datetime(df['Date'].astype(str)+ ' ' + df['IssueTime'].astype(str))
#Now we need to establish the leadtime
df['Leadtime']= pd.to_timedelta(df['Time(Hr)'], 'h')
#Apply a relevant datetime to each leadtime
df['DateTime'] = df['DateIssueTime'] + df['Leadtime']

#################################################################################
# Select only the dates we need
#List out the unique dates
dates = df['DateTime'].unique()
# Get the first date and set the time to 11pm.
start_date = dates.astype('datetime64[ns]')[0] + np.timedelta64(LEAD, 'h')
# Get the end time by adding 24hrs to start_date.
end_date = start_date + np.timedelta64(24, 'h')
# Log the start and end datetimes.
log.info("Datetimes between " + str(start_date) + " and " + str(end_date))
# Create a mask to select dates from start_date (incl) to end_date (excl).
mask = (df['DateTime'] >= start_date) & (df['DateTime'] < end_date)
# Get the records.
df = df.loc[mask]
#################################################################################

# Target the variables we want
dfU = df.loc[df['Param(10m)']== 'u']
dfU = dfU[['DateTime', 'Value(m/s)']]
dfU.columns = ['DateTime', 'U_10']
dfU.columns = ['DateTime', 'U_10']
dfU = dfU.set_index('DateTime')
#print(dfU.head())

dfV = df.loc[df['Param(10m)']== 'v']
dfV = dfV[['DateTime', 'Value(m/s)']]
dfV.columns = ['DateTime', 'V_10']
dfV= dfV.set_index('DateTime')
#print(dfV.head())

dfUG = df.loc[df['Param(10m)']== 'ugst']
dfUG = dfUG[['DateTime', 'Value(m/s)']]
dfUG.columns = ['DateTime', 'UG_10']
dfUG = dfUG.set_index('DateTime')
#print(dfUG.head())

dfVG = df.loc[df['Param(10m)']== 'vgst']
dfVG = dfVG[['DateTime', 'Value(m/s)']]
dfVG.columns = ['DateTime', 'VG_10']
dfVG = dfVG.set_index('DateTime')
#print(dfVG.head())

dfWDIR = df.loc[df['Param(10m)']== 'wdir']
dfWDIR = dfWDIR[['DateTime', 'Value(m/s)']]
dfWDIR.columns = ['DateTime', 'WDir_10']
dfWDIR = dfWDIR.set_index('DateTime')
#print(dfWDIR.head())

# Merge the subframes into one dataframe
df_merged = dfU.merge(dfV, how = 'outer', left_index=True, right_index=True)
df_merged = df_merged.merge(dfUG, how = 'outer', left_index=True, right_index=True)
df_merged = df_merged.merge(dfVG, how = 'outer', left_index=True, right_index=True)
df_merged = df_merged.merge(dfWDIR, how = 'outer', left_index=True, right_index=True)
#print(df_merged.head()

# Add the extra calculations.
df_merged['10mWS'] = np.sqrt(df_merged['U_10'].pow(2) + df_merged['V_10'].pow(2))
df_merged['10mWSg'] = np.sqrt(df_merged['UG_10'].pow(2) + df_merged['VG_10'].pow(2))
df_merged = df_merged.round(decimals=2)

df_merged.to_csv(r"/home/justi/Desktop/NewGurteenForecast/IREPS/DAM/DAM_IREPS_Forecasts/Member003_Converted.csv", mode = 'a', index = True, header = HEADER_VAR)
#df_merged.to_csv(r"Data\Members_Aggregated\Member003_Converted.csv", mode = 'a', index = True, header = HEADER_VAR)

log.info("Exported Member003 as CSV file successfully.")

M3 = df_merged[['U_10', 'V_10','10mWS']]
M3.rename(columns={'V_10':'M3_V10','U_10':'M3_U10','10mWS': 'M3_10mWS'}, inplace=True)

#######################################################################################
######## MEMBER004#####################################################################
#######################################################################################

mail.select(r'Member004')
log.info("Member004 Folder selected.")

# Get the list of email ids.
_, emaillist = mail.uid(r'search', None, r'(X-GM-RAW " has:attachment newer_than:1d")')

# print(emaillist)

# Pick the latest id.
latestid = emaillist[0].split()[-1]
# Fetch that email.
_, raw_email = mail.uid(r'fetch', latestid, r'(RFC822)')
raw_email = raw_email[0][1].decode('utf-8')
log.info("Latest email selected.")

csv_file = None
# Parse the email
met_email = email.message_from_string(raw_email)
for part in met_email.walk():
    # multipart/* are just containers
    if part.get_content_maintype() == 'multipart':
        continue
    filename = part.get_filename()
    if filename:
        log.info("Found filename '" + filename + "' in email.")
        csv_file = part.get_payload(decode=True)

# Make sure we got something.
assert csv_file, "Failed to extract CSV file from the email."
log.info("Found CSV file.")

#############################################################################
# Pretend csv_file is a file, as opposed to a string.
with io.StringIO(csv_file.decode('utf-8')) as f:
  df =  pd.read_csv(f, usecols=[0,1,2,3,4],header = 0, names=['Date', 'FCT_Run(Z)', 'Param(10m)', 'Time(Hr)', 'Value(m/s)'])

#Ensure that the date is in the correct format
df['Date'] = pd.to_datetime(df['Date'])
#Identify the Issue Time
df['IssueTime'] = df.iloc[:, 1]
#As the time is in integer formate we need to change and convert
df['IssueTime'] = (df['IssueTime']/100).astype(int)
df['IssueTime'] = pd.to_datetime(df['IssueTime'],format= '%H')
#We only want the time element
df['IssueTime']= df['IssueTime'].dt.time
#We need to establish a relevant datetime variable for the Issue
df['DateIssueTime'] = pd.to_datetime(df['Date'].astype(str)+ ' ' + df['IssueTime'].astype(str))
#Now we need to establish the leadtime
df['Leadtime']= pd.to_timedelta(df['Time(Hr)'], 'h')
#Apply a relevant datetime to each leadtime
df['DateTime'] = df['DateIssueTime'] + df['Leadtime']

#################################################################################
# Select only the dates we need
#List out the unique dates
dates = df['DateTime'].unique()
# Get the first date and set the time to 11pm.
start_date = dates.astype('datetime64[ns]')[0] + np.timedelta64(LEAD, 'h')
# Get the end time by adding 24hrs to start_date.
end_date = start_date + np.timedelta64(24, 'h')
# Log the start and end datetimes.
log.info("Datetimes between " + str(start_date) + " and " + str(end_date))
# Create a mask to select dates from start_date (incl) to end_date (excl).
mask = (df['DateTime'] >= start_date) & (df['DateTime'] < end_date)
# Get the records.
df = df.loc[mask]
#################################################################################

# Target the variables we want
dfU = df.loc[df['Param(10m)']== 'u']
dfU = dfU[['DateTime', 'Value(m/s)']]
dfU.columns = ['DateTime', 'U_10']
dfU.columns = ['DateTime', 'U_10']
dfU = dfU.set_index('DateTime')
#print(dfU.head())

dfV = df.loc[df['Param(10m)']== 'v']
dfV = dfV[['DateTime', 'Value(m/s)']]
dfV.columns = ['DateTime', 'V_10']
dfV= dfV.set_index('DateTime')
#print(dfV.head())

dfUG = df.loc[df['Param(10m)']== 'ugst']
dfUG = dfUG[['DateTime', 'Value(m/s)']]
dfUG.columns = ['DateTime', 'UG_10']
dfUG = dfUG.set_index('DateTime')
#print(dfUG.head())

dfVG = df.loc[df['Param(10m)']== 'vgst']
dfVG = dfVG[['DateTime', 'Value(m/s)']]
dfVG.columns = ['DateTime', 'VG_10']
dfVG = dfVG.set_index('DateTime')
#print(dfVG.head())

dfWDIR = df.loc[df['Param(10m)']== 'wdir']
dfWDIR = dfWDIR[['DateTime', 'Value(m/s)']]
dfWDIR.columns = ['DateTime', 'WDir_10']
dfWDIR = dfWDIR.set_index('DateTime')
#print(dfWDIR.head())

# Merge the subframes into one dataframe
df_merged = dfU.merge(dfV, how = 'outer', left_index=True, right_index=True)
df_merged = df_merged.merge(dfUG, how = 'outer', left_index=True, right_index=True)
df_merged = df_merged.merge(dfVG, how = 'outer', left_index=True, right_index=True)
df_merged = df_merged.merge(dfWDIR, how = 'outer', left_index=True, right_index=True)
#print(df_merged.head()

# Add the extra calculations.
df_merged['10mWS'] = np.sqrt(df_merged['U_10'].pow(2) + df_merged['V_10'].pow(2))
df_merged['10mWSg'] = np.sqrt(df_merged['UG_10'].pow(2) + df_merged['VG_10'].pow(2))
df_merged = df_merged.round(decimals=2)

df_merged.to_csv(r"/home/justi/Desktop/NewGurteenForecast/IREPS/DAM/DAM_IREPS_Forecasts/Member004_Converted.csv", mode = 'a', index = True, header = HEADER_VAR)
#df_merged.to_csv(r"Data\Members_Aggregated\Member004_Converted.csv", mode = 'a', index = True, header = HEADER_VAR)

log.info("Exported Member004 as CSV file successfully.")

M4 = df_merged[['U_10', 'V_10','10mWS']]
M4.rename(columns={'V_10':'M4_V10','U_10':'M4_U10','10mWS': 'M4_10mWS'}, inplace=True)

#######################################################################################
######## MEMBER005 #####################################################################
#######################################################################################

mail.select(r'Member005')
log.info("Member005 Folder selected.")

# Get the list of email ids.
_, emaillist = mail.uid(r'search', None, r'(X-GM-RAW " has:attachment newer_than:1d")')

# print(emaillist)

# Pick the latest id.
latestid = emaillist[0].split()[-1]
# Fetch that email.
_, raw_email = mail.uid(r'fetch', latestid, r'(RFC822)')
raw_email = raw_email[0][1].decode('utf-8')
log.info("Latest email selected.")

csv_file = None
# Parse the email
met_email = email.message_from_string(raw_email)
for part in met_email.walk():
    # multipart/* are just containers
    if part.get_content_maintype() == 'multipart':
        continue
    filename = part.get_filename()
    if filename:
        log.info("Found filename '" + filename + "' in email.")
        csv_file = part.get_payload(decode=True)

# Make sure we got something.
assert csv_file, "Failed to extract CSV file from the email."
log.info("Found CSV file.")

################################################################################

# Pretend csv_file is a file, as opposed to a string.
with io.StringIO(csv_file.decode('utf-8')) as f:
  df =  pd.read_csv(f, usecols=[0,1,2,3,4],header = 0, names=['Date', 'FCT_Run(Z)', 'Param(10m)', 'Time(Hr)', 'Value(m/s)'])

#Ensure that the date is in the correct format
df['Date'] = pd.to_datetime(df['Date'])
#Identify the Issue Time
df['IssueTime'] = df.iloc[:, 1]
#As the time is in integer formate we need to change and convert
df['IssueTime'] = (df['IssueTime']/100).astype(int)
df['IssueTime'] = pd.to_datetime(df['IssueTime'],format= '%H')
#We only want the time element
df['IssueTime']= df['IssueTime'].dt.time
#We need to establish a relevant datetime variable for the Issue
df['DateIssueTime'] = pd.to_datetime(df['Date'].astype(str)+ ' ' + df['IssueTime'].astype(str))
#Now we need to establish the leadtime
df['Leadtime']= pd.to_timedelta(df['Time(Hr)'], 'h')
#Apply a relevant datetime to each leadtime
df['DateTime'] = df['DateIssueTime'] + df['Leadtime']

#################################################################################
# Select only the dates we need
#List out the unique dates
dates = df['DateTime'].unique()
# Get the first date and set the time to 11pm.
start_date = dates.astype('datetime64[ns]')[0] + np.timedelta64(LEAD, 'h')
# Get the end time by adding 24hrs to start_date.
end_date = start_date + np.timedelta64(24, 'h')
# Log the start and end datetimes.
log.info("Datetimes between " + str(start_date) + " and " + str(end_date))
# Create a mask to select dates from start_date (incl) to end_date (excl).
mask = (df['DateTime'] >= start_date) & (df['DateTime'] < end_date)
# Get the records.
df = df.loc[mask]
#################################################################################

# Target the variables we want
dfU = df.loc[df['Param(10m)']== 'u']
dfU = dfU[['DateTime', 'Value(m/s)']]
dfU.columns = ['DateTime', 'U_10']
dfU.columns = ['DateTime', 'U_10']
dfU = dfU.set_index('DateTime')
#print(dfU.head())

dfV = df.loc[df['Param(10m)']== 'v']
dfV = dfV[['DateTime', 'Value(m/s)']]
dfV.columns = ['DateTime', 'V_10']
dfV= dfV.set_index('DateTime')
#print(dfV.head())

dfUG = df.loc[df['Param(10m)']== 'ugst']
dfUG = dfUG[['DateTime', 'Value(m/s)']]
dfUG.columns = ['DateTime', 'UG_10']
dfUG = dfUG.set_index('DateTime')
#print(dfUG.head())

dfVG = df.loc[df['Param(10m)']== 'vgst']
dfVG = dfVG[['DateTime', 'Value(m/s)']]
dfVG.columns = ['DateTime', 'VG_10']
dfVG = dfVG.set_index('DateTime')
#print(dfVG.head())

dfWDIR = df.loc[df['Param(10m)']== 'wdir']
dfWDIR = dfWDIR[['DateTime', 'Value(m/s)']]
dfWDIR.columns = ['DateTime', 'WDir_10']
dfWDIR = dfWDIR.set_index('DateTime')
#print(dfWDIR.head())

# Merge the subframes into one dataframe
df_merged = dfU.merge(dfV, how = 'outer', left_index=True, right_index=True)
df_merged = df_merged.merge(dfUG, how = 'outer', left_index=True, right_index=True)
df_merged = df_merged.merge(dfVG, how = 'outer', left_index=True, right_index=True)
df_merged = df_merged.merge(dfWDIR, how = 'outer', left_index=True, right_index=True)
#print(df_merged.head()

# Add the extra calculations.
df_merged['10mWS'] = np.sqrt(df_merged['U_10'].pow(2) + df_merged['V_10'].pow(2))
df_merged['10mWSg'] = np.sqrt(df_merged['UG_10'].pow(2) + df_merged['VG_10'].pow(2))
df_merged = df_merged.round(decimals=2)

df_merged.to_csv(r"/home/justi/Desktop/NewGurteenForecast/IREPS/DAM/DAM_IREPS_Forecasts/Member005_Converted.csv", mode = 'a', index = True, header = HEADER_VAR)
#df_merged.to_csv(r"Data\Members_Aggregated\Member005_Converted.csv", mode = 'a', index = True, header = HEADER_VAR)

log.info("Exported Member005 as CSV file successfully.")

M5 = df_merged[['U_10', 'V_10','10mWS']]
M5.rename(columns={'V_10':'M5_V10','U_10':'M5_U10','10mWS': 'M5_10mWS'}, inplace=True)

#######################################################################################
######## MEMBER006 #####################################################################
#######################################################################################

mail.select(r'Member006')
log.info("Member006 Folder selected.")

# Get the list of email ids.
_, emaillist = mail.uid(r'search', None, r'(X-GM-RAW " has:attachment newer_than:1d")')

# print(emaillist)

# Pick the latest id.
latestid = emaillist[0].split()[-1]
# Fetch that email.
_, raw_email = mail.uid(r'fetch', latestid, r'(RFC822)')
raw_email = raw_email[0][1].decode('utf-8')
log.info("Latest email selected.")

csv_file = None
# Parse the email
met_email = email.message_from_string(raw_email)
for part in met_email.walk():
    # multipart/* are just containers
    if part.get_content_maintype() == 'multipart':
        continue
    filename = part.get_filename()
    if filename:
        log.info("Found filename '" + filename + "' in email.")
        csv_file = part.get_payload(decode=True)

# Make sure we got something.
assert csv_file, "Failed to extract CSV file from the email."
log.info("Found CSV file.")

##################################################################################

# Pretend csv_file is a file, as opposed to a string.
with io.StringIO(csv_file.decode('utf-8')) as f:
  df =  pd.read_csv(f, usecols=[0,1,2,3,4],header = 0, names=['Date', 'FCT_Run(Z)', 'Param(10m)', 'Time(Hr)', 'Value(m/s)'])

#Ensure that the date is in the correct format
df['Date'] = pd.to_datetime(df['Date'])
#Identify the Issue Time
df['IssueTime'] = df.iloc[:, 1]
#As the time is in integer formate we need to change and convert
df['IssueTime'] = (df['IssueTime']/100).astype(int)
df['IssueTime'] = pd.to_datetime(df['IssueTime'],format= '%H')
#We only want the time element
df['IssueTime']= df['IssueTime'].dt.time
#We need to establish a relevant datetime variable for the Issue
df['DateIssueTime'] = pd.to_datetime(df['Date'].astype(str)+ ' ' + df['IssueTime'].astype(str))
#Now we need to establish the leadtime
df['Leadtime']= pd.to_timedelta(df['Time(Hr)'], 'h')
#Apply a relevant datetime to each leadtime
df['DateTime'] = df['DateIssueTime'] + df['Leadtime']

#################################################################################
# Select only the dates we need
#List out the unique dates
dates = df['DateTime'].unique()
# Get the first date and set the time to 11pm.
start_date = dates.astype('datetime64[ns]')[0] + np.timedelta64(LEAD, 'h')
# Get the end time by adding 24hrs to start_date.
end_date = start_date + np.timedelta64(24, 'h')
# Log the start and end datetimes.
log.info("Datetimes between " + str(start_date) + " and " + str(end_date))
# Create a mask to select dates from start_date (incl) to end_date (excl).
mask = (df['DateTime'] >= start_date) & (df['DateTime'] < end_date)
# Get the records.
df = df.loc[mask]
#################################################################################

# Target the variables we want
dfU = df.loc[df['Param(10m)']== 'u']
dfU = dfU[['DateTime', 'Value(m/s)']]
dfU.columns = ['DateTime', 'U_10']
dfU.columns = ['DateTime', 'U_10']
dfU = dfU.set_index('DateTime')
#print(dfU.head())

dfV = df.loc[df['Param(10m)']== 'v']
dfV = dfV[['DateTime', 'Value(m/s)']]
dfV.columns = ['DateTime', 'V_10']
dfV= dfV.set_index('DateTime')
#print(dfV.head())

dfUG = df.loc[df['Param(10m)']== 'ugst']
dfUG = dfUG[['DateTime', 'Value(m/s)']]
dfUG.columns = ['DateTime', 'UG_10']
dfUG = dfUG.set_index('DateTime')
#print(dfUG.head())

dfVG = df.loc[df['Param(10m)']== 'vgst']
dfVG = dfVG[['DateTime', 'Value(m/s)']]
dfVG.columns = ['DateTime', 'VG_10']
dfVG = dfVG.set_index('DateTime')
#print(dfVG.head())

dfWDIR = df.loc[df['Param(10m)']== 'wdir']
dfWDIR = dfWDIR[['DateTime', 'Value(m/s)']]
dfWDIR.columns = ['DateTime', 'WDir_10']
dfWDIR = dfWDIR.set_index('DateTime')
#print(dfWDIR.head())

# Merge the subframes into one dataframe
df_merged = dfU.merge(dfV, how = 'outer', left_index=True, right_index=True)
df_merged = df_merged.merge(dfUG, how = 'outer', left_index=True, right_index=True)
df_merged = df_merged.merge(dfVG, how = 'outer', left_index=True, right_index=True)
df_merged = df_merged.merge(dfWDIR, how = 'outer', left_index=True, right_index=True)
#print(df_merged.head()

# Add the extra calculations.
df_merged['10mWS'] = np.sqrt(df_merged['U_10'].pow(2) + df_merged['V_10'].pow(2))
df_merged['10mWSg'] = np.sqrt(df_merged['UG_10'].pow(2) + df_merged['VG_10'].pow(2))
df_merged = df_merged.round(decimals=2)

df_merged.to_csv(r"/home/justi/Desktop/NewGurteenForecast/IREPS/DAM/DAM_IREPS_Forecasts/Member006_Converted.csv", mode = 'a', index = True, header = HEADER_VAR)
#df_merged.to_csv(r"Data\Members_Aggregated\Member006_Converted.csv", mode = 'a', index = True, header = HEADER_VAR)

log.info("Exported Member006 as CSV file successfully.")

M6 = df_merged[['U_10', 'V_10','10mWS']]
M6.rename(columns={'V_10':'M6_V10','U_10':'M6_U10','10mWS': 'M6_10mWS'}, inplace=True)

#######################################################################################
######## MEMBER007 #####################################################################
#######################################################################################

mail.select(r'Member007')
log.info("Member007 Folder selected.")

# Get the list of email ids.
_, emaillist = mail.uid(r'search', None, r'(X-GM-RAW " has:attachment newer_than:1d")')

# print(emaillist)

# Pick the latest id.
latestid = emaillist[0].split()[-1]
# Fetch that email.
_, raw_email = mail.uid(r'fetch', latestid, r'(RFC822)')
raw_email = raw_email[0][1].decode('utf-8')
log.info("Latest email selected.")

csv_file = None
# Parse the email
met_email = email.message_from_string(raw_email)
for part in met_email.walk():
    # multipart/* are just containers
    if part.get_content_maintype() == 'multipart':
        continue
    filename = part.get_filename()
    if filename:
        log.info("Found filename '" + filename + "' in email.")
        csv_file = part.get_payload(decode=True)

# Make sure we got something.
assert csv_file, "Failed to extract CSV file from the email."
log.info("Found CSV file.")

##################################################################################

# Pretend csv_file is a file, as opposed to a string.
with io.StringIO(csv_file.decode('utf-8')) as f:
  df =  pd.read_csv(f, usecols=[0,1,2,3,4],header = 0, names=['Date', 'FCT_Run(Z)', 'Param(10m)', 'Time(Hr)', 'Value(m/s)'])

#Ensure that the date is in the correct format
df['Date'] = pd.to_datetime(df['Date'])
#Identify the Issue Time
df['IssueTime'] = df.iloc[:, 1]
#As the time is in integer formate we need to change and convert
df['IssueTime'] = (df['IssueTime']/100).astype(int)
df['IssueTime'] = pd.to_datetime(df['IssueTime'],format= '%H')
#We only want the time element
df['IssueTime']= df['IssueTime'].dt.time
#We need to establish a relevant datetime variable for the Issue
df['DateIssueTime'] = pd.to_datetime(df['Date'].astype(str)+ ' ' + df['IssueTime'].astype(str))
#Now we need to establish the leadtime
df['Leadtime']= pd.to_timedelta(df['Time(Hr)'], 'h')
#Apply a relevant datetime to each leadtime
df['DateTime'] = df['DateIssueTime'] + df['Leadtime']

#################################################################################
# Select only the dates we need
#List out the unique dates
dates = df['DateTime'].unique()
# Get the first date and set the time to 11pm.
start_date = dates.astype('datetime64[ns]')[0] + np.timedelta64(LEAD, 'h')
# Get the end time by adding 24hrs to start_date.
end_date = start_date + np.timedelta64(24, 'h')
# Log the start and end datetimes.
log.info("Datetimes between " + str(start_date) + " and " + str(end_date))
# Create a mask to select dates from start_date (incl) to end_date (excl).
mask = (df['DateTime'] >= start_date) & (df['DateTime'] < end_date)
# Get the records.
df = df.loc[mask]
#################################################################################

# Target the variables we want
dfU = df.loc[df['Param(10m)']== 'u']
dfU = dfU[['DateTime', 'Value(m/s)']]
dfU.columns = ['DateTime', 'U_10']
dfU.columns = ['DateTime', 'U_10']
dfU = dfU.set_index('DateTime')
#print(dfU.head())

dfV = df.loc[df['Param(10m)']== 'v']
dfV = dfV[['DateTime', 'Value(m/s)']]
dfV.columns = ['DateTime', 'V_10']
dfV= dfV.set_index('DateTime')
#print(dfV.head())

dfUG = df.loc[df['Param(10m)']== 'ugst']
dfUG = dfUG[['DateTime', 'Value(m/s)']]
dfUG.columns = ['DateTime', 'UG_10']
dfUG = dfUG.set_index('DateTime')
#print(dfUG.head())

dfVG = df.loc[df['Param(10m)']== 'vgst']
dfVG = dfVG[['DateTime', 'Value(m/s)']]
dfVG.columns = ['DateTime', 'VG_10']
dfVG = dfVG.set_index('DateTime')
#print(dfVG.head())

dfWDIR = df.loc[df['Param(10m)']== 'wdir']
dfWDIR = dfWDIR[['DateTime', 'Value(m/s)']]
dfWDIR.columns = ['DateTime', 'WDir_10']
dfWDIR = dfWDIR.set_index('DateTime')
#print(dfWDIR.head())

# Merge the subframes into one dataframe
df_merged = dfU.merge(dfV, how = 'outer', left_index=True, right_index=True)
df_merged = df_merged.merge(dfUG, how = 'outer', left_index=True, right_index=True)
df_merged = df_merged.merge(dfVG, how = 'outer', left_index=True, right_index=True)
df_merged = df_merged.merge(dfWDIR, how = 'outer', left_index=True, right_index=True)
#print(df_merged.head()

# Add the extra calculations.
df_merged['10mWS'] = np.sqrt(df_merged['U_10'].pow(2) + df_merged['V_10'].pow(2))
df_merged['10mWSg'] = np.sqrt(df_merged['UG_10'].pow(2) + df_merged['VG_10'].pow(2))
df_merged = df_merged.round(decimals=2)

df_merged.to_csv(r"/home/justi/Desktop/NewGurteenForecast/IREPS/DAM/DAM_IREPS_Forecasts/Member007_Converted.csv", mode = 'a', index = True, header = HEADER_VAR)
#df_merged.to_csv(r"Data\Members_Aggregated\Member007_Converted.csv", mode = 'a', index = True, header = HEADER_VAR)

log.info("Exported Member007 as CSV file successfully.")

M7 = df_merged[['U_10', 'V_10','10mWS']]
M7.rename(columns={'V_10':'M7_V10','U_10':'M7_U10','10mWS': 'M7_10mWS'}, inplace=True)


#######################################################################################
######## MEMBER008 #####################################################################
#######################################################################################

mail.select(r'Member008')
log.info("Member008 Folder selected.")

# Get the list of email ids.
_, emaillist = mail.uid(r'search', None, r'(X-GM-RAW " has:attachment newer_than:1d")')

# print(emaillist)

# Pick the latest id.
latestid = emaillist[0].split()[-1]
# Fetch that email.
_, raw_email = mail.uid(r'fetch', latestid, r'(RFC822)')
raw_email = raw_email[0][1].decode('utf-8')
log.info("Latest email selected.")

csv_file = None
# Parse the email
met_email = email.message_from_string(raw_email)
for part in met_email.walk():
    # multipart/* are just containers
    if part.get_content_maintype() == 'multipart':
        continue
    filename = part.get_filename()
    if filename:
        log.info("Found filename '" + filename + "' in email.")
        csv_file = part.get_payload(decode=True)

# Make sure we got something.
assert csv_file, "Failed to extract CSV file from the email."
log.info("Found CSV file.")

##################################################################################

# Pretend csv_file is a file, as opposed to a string.
with io.StringIO(csv_file.decode('utf-8')) as f:
  df =  pd.read_csv(f, usecols=[0,1,2,3,4],header = 0, names=['Date', 'FCT_Run(Z)', 'Param(10m)', 'Time(Hr)', 'Value(m/s)'])

#Ensure that the date is in the correct format
df['Date'] = pd.to_datetime(df['Date'])
#Identify the Issue Time
df['IssueTime'] = df.iloc[:, 1]
#As the time is in integer formate we need to change and convert
df['IssueTime'] = (df['IssueTime']/100).astype(int)
df['IssueTime'] = pd.to_datetime(df['IssueTime'],format= '%H')
#We only want the time element
df['IssueTime']= df['IssueTime'].dt.time
#We need to establish a relevant datetime variable for the Issue
df['DateIssueTime'] = pd.to_datetime(df['Date'].astype(str)+ ' ' + df['IssueTime'].astype(str))
#Now we need to establish the leadtime
df['Leadtime']= pd.to_timedelta(df['Time(Hr)'], 'h')
#Apply a relevant datetime to each leadtime
df['DateTime'] = df['DateIssueTime'] + df['Leadtime']

#################################################################################
# Select only the dates we need
#List out the unique dates
dates = df['DateTime'].unique()
# Get the first date and set the time to 11pm.
start_date = dates.astype('datetime64[ns]')[0] + np.timedelta64(LEAD, 'h')
# Get the end time by adding 24hrs to start_date.
end_date = start_date + np.timedelta64(24, 'h')
# Log the start and end datetimes.
log.info("Datetimes between " + str(start_date) + " and " + str(end_date))
# Create a mask to select dates from start_date (incl) to end_date (excl).
mask = (df['DateTime'] >= start_date) & (df['DateTime'] < end_date)
# Get the records.
df = df.loc[mask]
#################################################################################

# Target the variables we want
dfU = df.loc[df['Param(10m)']== 'u']
dfU = dfU[['DateTime', 'Value(m/s)']]
dfU.columns = ['DateTime', 'U_10']
dfU.columns = ['DateTime', 'U_10']
dfU = dfU.set_index('DateTime')
#print(dfU.head())

dfV = df.loc[df['Param(10m)']== 'v']
dfV = dfV[['DateTime', 'Value(m/s)']]
dfV.columns = ['DateTime', 'V_10']
dfV= dfV.set_index('DateTime')
#print(dfV.head())

dfUG = df.loc[df['Param(10m)']== 'ugst']
dfUG = dfUG[['DateTime', 'Value(m/s)']]
dfUG.columns = ['DateTime', 'UG_10']
dfUG = dfUG.set_index('DateTime')
#print(dfUG.head())

dfVG = df.loc[df['Param(10m)']== 'vgst']
dfVG = dfVG[['DateTime', 'Value(m/s)']]
dfVG.columns = ['DateTime', 'VG_10']
dfVG = dfVG.set_index('DateTime')
#print(dfVG.head())

dfWDIR = df.loc[df['Param(10m)']== 'wdir']
dfWDIR = dfWDIR[['DateTime', 'Value(m/s)']]
dfWDIR.columns = ['DateTime', 'WDir_10']
dfWDIR = dfWDIR.set_index('DateTime')
#print(dfWDIR.head())

# Merge the subframes into one dataframe
df_merged = dfU.merge(dfV, how = 'outer', left_index=True, right_index=True)
df_merged = df_merged.merge(dfUG, how = 'outer', left_index=True, right_index=True)
df_merged = df_merged.merge(dfVG, how = 'outer', left_index=True, right_index=True)
df_merged = df_merged.merge(dfWDIR, how = 'outer', left_index=True, right_index=True)
#print(df_merged.head()

# Add the extra calculations.
df_merged['10mWS'] = np.sqrt(df_merged['U_10'].pow(2) + df_merged['V_10'].pow(2))
df_merged['10mWSg'] = np.sqrt(df_merged['UG_10'].pow(2) + df_merged['VG_10'].pow(2))
df_merged = df_merged.round(decimals=2)

df_merged.to_csv(r"/home/justi/Desktop/NewGurteenForecast/IREPS/DAM/DAM_IREPS_Forecasts/Member008_Converted.csv", mode = 'a', index = True, header = HEADER_VAR)
#df_merged.to_csv(r"Data\Members_Aggregated\Member008_Converted.csv", mode = 'a', index = True, header = HEADER_VAR)

log.info("Exported Member008 as CSV file successfully.")

M8 = df_merged[['U_10', 'V_10','10mWS']]
M8.rename(columns={'V_10':'M8_V10','U_10':'M8_U10','10mWS': 'M8_10mWS'}, inplace=True)


#######################################################################################
######## MEMBER009 #####################################################################
#######################################################################################

mail.select(r'Member009')
log.info("Member009 Folder selected.")

# Get the list of email ids.
_, emaillist = mail.uid(r'search', None, r'(X-GM-RAW " has:attachment newer_than:1d")')

# print(emaillist)

# Pick the latest id.
latestid = emaillist[0].split()[-1]
# Fetch that email.
_, raw_email = mail.uid(r'fetch', latestid, r'(RFC822)')
raw_email = raw_email[0][1].decode('utf-8')
log.info("Latest email selected.")

csv_file = None
# Parse the email
met_email = email.message_from_string(raw_email)
for part in met_email.walk():
    # multipart/* are just containers
    if part.get_content_maintype() == 'multipart':
        continue
    filename = part.get_filename()
    if filename:
        log.info("Found filename '" + filename + "' in email.")
        csv_file = part.get_payload(decode=True)

# Make sure we got something.
assert csv_file, "Failed to extract CSV file from the email."
log.info("Found CSV file.")

##################################################################################

# Pretend csv_file is a file, as opposed to a string.
with io.StringIO(csv_file.decode('utf-8')) as f:
  df =  pd.read_csv(f, usecols=[0,1,2,3,4],header = 0, names=['Date', 'FCT_Run(Z)', 'Param(10m)', 'Time(Hr)', 'Value(m/s)'])

#Ensure that the date is in the correct format
df['Date'] = pd.to_datetime(df['Date'])
#Identify the Issue Time
df['IssueTime'] = df.iloc[:, 1]
#As the time is in integer formate we need to change and convert
df['IssueTime'] = (df['IssueTime']/100).astype(int)
df['IssueTime'] = pd.to_datetime(df['IssueTime'],format= '%H')
#We only want the time element
df['IssueTime']= df['IssueTime'].dt.time
#We need to establish a relevant datetime variable for the Issue
df['DateIssueTime'] = pd.to_datetime(df['Date'].astype(str)+ ' ' + df['IssueTime'].astype(str))
#Now we need to establish the leadtime
df['Leadtime']= pd.to_timedelta(df['Time(Hr)'], 'h')
#Apply a relevant datetime to each leadtime
df['DateTime'] = df['DateIssueTime'] + df['Leadtime']

#################################################################################
# Select only the dates we need
#List out the unique dates
dates = df['DateTime'].unique()
# Get the first date and set the time to 11pm.
start_date = dates.astype('datetime64[ns]')[0] + np.timedelta64(LEAD, 'h')
# Get the end time by adding 24hrs to start_date.
end_date = start_date + np.timedelta64(24, 'h')
# Log the start and end datetimes.
log.info("Datetimes between " + str(start_date) + " and " + str(end_date))
# Create a mask to select dates from start_date (incl) to end_date (excl).
mask = (df['DateTime'] >= start_date) & (df['DateTime'] < end_date)
# Get the records.
df = df.loc[mask]
#################################################################################

# Target the variables we want
dfU = df.loc[df['Param(10m)']== 'u']
dfU = dfU[['DateTime', 'Value(m/s)']]
dfU.columns = ['DateTime', 'U_10']
dfU.columns = ['DateTime', 'U_10']
dfU = dfU.set_index('DateTime')
#print(dfU.head())

dfV = df.loc[df['Param(10m)']== 'v']
dfV = dfV[['DateTime', 'Value(m/s)']]
dfV.columns = ['DateTime', 'V_10']
dfV= dfV.set_index('DateTime')
#print(dfV.head())

dfUG = df.loc[df['Param(10m)']== 'ugst']
dfUG = dfUG[['DateTime', 'Value(m/s)']]
dfUG.columns = ['DateTime', 'UG_10']
dfUG = dfUG.set_index('DateTime')
#print(dfUG.head())

dfVG = df.loc[df['Param(10m)']== 'vgst']
dfVG = dfVG[['DateTime', 'Value(m/s)']]
dfVG.columns = ['DateTime', 'VG_10']
dfVG = dfVG.set_index('DateTime')
#print(dfVG.head())

dfWDIR = df.loc[df['Param(10m)']== 'wdir']
dfWDIR = dfWDIR[['DateTime', 'Value(m/s)']]
dfWDIR.columns = ['DateTime', 'WDir_10']
dfWDIR = dfWDIR.set_index('DateTime')
#print(dfWDIR.head())

# Merge the subframes into one dataframe
df_merged = dfU.merge(dfV, how = 'outer', left_index=True, right_index=True)
df_merged = df_merged.merge(dfUG, how = 'outer', left_index=True, right_index=True)
df_merged = df_merged.merge(dfVG, how = 'outer', left_index=True, right_index=True)
df_merged = df_merged.merge(dfWDIR, how = 'outer', left_index=True, right_index=True)
#print(df_merged.head()

# Add the extra calculations.
df_merged['10mWS'] = np.sqrt(df_merged['U_10'].pow(2) + df_merged['V_10'].pow(2))
df_merged['10mWSg'] = np.sqrt(df_merged['UG_10'].pow(2) + df_merged['VG_10'].pow(2))
df_merged = df_merged.round(decimals=2)

df_merged.to_csv(r"/home/justi/Desktop/NewGurteenForecast/IREPS/DAM/DAM_IREPS_Forecasts/Member009_Converted.csv", mode = 'a', index = True, header = HEADER_VAR)
#df_merged.to_csv(r"Data\Members_Aggregated\Member009_Converted.csv", mode = 'a', index = True, header = HEADER_VAR)

log.info("Exported Member009 as CSV file successfully.")

M9 = df_merged[['U_10', 'V_10','10mWS']]
M9.rename(columns={'V_10':'M9_V10','U_10':'M9_U10','10mWS': 'M9_10mWS'}, inplace=True)

#######################################################################################
######## MEMBER010 #####################################################################
#######################################################################################

mail.select(r'Member010')
log.info("Member010 Folder selected.")

# Get the list of email ids.
_, emaillist = mail.uid(r'search', None, r'(X-GM-RAW " has:attachment newer_than:1d")')

# print(emaillist)

# Pick the latest id.
latestid = emaillist[0].split()[-1]
# Fetch that email.
_, raw_email = mail.uid(r'fetch', latestid, r'(RFC822)')
raw_email = raw_email[0][1].decode('utf-8')
log.info("Latest email selected.")

csv_file = None
# Parse the email
met_email = email.message_from_string(raw_email)
for part in met_email.walk():
    # multipart/* are just containers
    if part.get_content_maintype() == 'multipart':
        continue
    filename = part.get_filename()
    if filename:
        log.info("Found filename '" + filename + "' in email.")
        csv_file = part.get_payload(decode=True)

# Make sure we got something.
assert csv_file, "Failed to extract CSV file from the email."
log.info("Found CSV file.")

##################################################################################

# Pretend csv_file is a file, as opposed to a string.
with io.StringIO(csv_file.decode('utf-8')) as f:
  df =  pd.read_csv(f, usecols=[0,1,2,3,4],header = 0, names=['Date', 'FCT_Run(Z)', 'Param(10m)', 'Time(Hr)', 'Value(m/s)'])

#Ensure that the date is in the correct format
df['Date'] = pd.to_datetime(df['Date'])
#Identify the Issue Time
df['IssueTime'] = df.iloc[:, 1]
#As the time is in integer formate we need to change and convert
df['IssueTime'] = (df['IssueTime']/100).astype(int)
df['IssueTime'] = pd.to_datetime(df['IssueTime'],format= '%H')
#We only want the time element
df['IssueTime']= df['IssueTime'].dt.time
#We need to establish a relevant datetime variable for the Issue
df['DateIssueTime'] = pd.to_datetime(df['Date'].astype(str)+ ' ' + df['IssueTime'].astype(str))
#Now we need to establish the leadtime
df['Leadtime']= pd.to_timedelta(df['Time(Hr)'], 'h')
#Apply a relevant datetime to each leadtime
df['DateTime'] = df['DateIssueTime'] + df['Leadtime']

#################################################################################
# Select only the dates we need
#List out the unique dates
dates = df['DateTime'].unique()
# Get the first date and set the time to 11pm.
start_date = dates.astype('datetime64[ns]')[0] + np.timedelta64(LEAD, 'h')
# Get the end time by adding 24hrs to start_date.
end_date = start_date + np.timedelta64(24, 'h')
# Log the start and end datetimes.
log.info("Datetimes between " + str(start_date) + " and " + str(end_date))
# Create a mask to select dates from start_date (incl) to end_date (excl).
mask = (df['DateTime'] >= start_date) & (df['DateTime'] < end_date)
# Get the records.
df = df.loc[mask]
#################################################################################

# Target the variables we want
dfU = df.loc[df['Param(10m)']== 'u']
dfU = dfU[['DateTime', 'Value(m/s)']]
dfU.columns = ['DateTime', 'U_10']
dfU.columns = ['DateTime', 'U_10']
dfU = dfU.set_index('DateTime')
#print(dfU.head())

dfV = df.loc[df['Param(10m)']== 'v']
dfV = dfV[['DateTime', 'Value(m/s)']]
dfV.columns = ['DateTime', 'V_10']
dfV= dfV.set_index('DateTime')
#print(dfV.head())

dfUG = df.loc[df['Param(10m)']== 'ugst']
dfUG = dfUG[['DateTime', 'Value(m/s)']]
dfUG.columns = ['DateTime', 'UG_10']
dfUG = dfUG.set_index('DateTime')
#print(dfUG.head())

dfVG = df.loc[df['Param(10m)']== 'vgst']
dfVG = dfVG[['DateTime', 'Value(m/s)']]
dfVG.columns = ['DateTime', 'VG_10']
dfVG = dfVG.set_index('DateTime')
#print(dfVG.head())

dfWDIR = df.loc[df['Param(10m)']== 'wdir']
dfWDIR = dfWDIR[['DateTime', 'Value(m/s)']]
dfWDIR.columns = ['DateTime', 'WDir_10']
dfWDIR = dfWDIR.set_index('DateTime')
#print(dfWDIR.head())

# Merge the subframes into one dataframe
df_merged = dfU.merge(dfV, how = 'outer', left_index=True, right_index=True)
df_merged = df_merged.merge(dfUG, how = 'outer', left_index=True, right_index=True)
df_merged = df_merged.merge(dfVG, how = 'outer', left_index=True, right_index=True)
df_merged = df_merged.merge(dfWDIR, how = 'outer', left_index=True, right_index=True)
#print(df_merged.head()

# Add the extra calculations.
df_merged['10mWS'] = np.sqrt(df_merged['U_10'].pow(2) + df_merged['V_10'].pow(2))
df_merged['10mWSg'] = np.sqrt(df_merged['UG_10'].pow(2) + df_merged['VG_10'].pow(2))
df_merged = df_merged.round(decimals=2)

df_merged.to_csv(r"/home/justi/Desktop/NewGurteenForecast/IREPS/DAM/DAM_IREPS_Forecasts/Member010_Converted.csv", mode = 'a', index = True, header = HEADER_VAR)
#df_merged.to_csv(r"Data\Members_Aggregated\Member010_Converted.csv", mode = 'a', index = True, header = HEADER_VAR)

log.info("Exported Member010 as CSV file successfully.")

M10 = df_merged[['U_10', 'V_10','10mWS']]
M10.rename(columns={'V_10':'M10_V10','U_10':'M10_U10','10mWS': 'M10_10mWS'}, inplace=True)

#######################################################################################
######## MEMBER011 #####################################################################
#######################################################################################

mail.select(r'Member011')
log.info("Member011 Folder selected.")

# Get the list of email ids.
_, emaillist = mail.uid(r'search', None, r'(X-GM-RAW " has:attachment newer_than:1d")')

# print(emaillist)

# Pick the latest id.
latestid = emaillist[0].split()[-1]
# Fetch that email.
_, raw_email = mail.uid(r'fetch', latestid, r'(RFC822)')
raw_email = raw_email[0][1].decode('utf-8')
log.info("Latest email selected.")

csv_file = None
# Parse the email
met_email = email.message_from_string(raw_email)
for part in met_email.walk():
    # multipart/* are just containers
    if part.get_content_maintype() == 'multipart':
        continue
    filename = part.get_filename()
    if filename:
        log.info("Found filename '" + filename + "' in email.")
        csv_file = part.get_payload(decode=True)

# Make sure we got something.
assert csv_file, "Failed to extract CSV file from the email."
log.info("Found CSV file.")

##################################################################################

# Pretend csv_file is a file, as opposed to a string.
with io.StringIO(csv_file.decode('utf-8')) as f:
  df =  pd.read_csv(f, usecols=[0,1,2,3,4],header = 0, names=['Date', 'FCT_Run(Z)', 'Param(10m)', 'Time(Hr)', 'Value(m/s)'])

#Ensure that the date is in the correct format
df['Date'] = pd.to_datetime(df['Date'])
#Identify the Issue Time
df['IssueTime'] = df.iloc[:, 1]
#As the time is in integer formate we need to change and convert
df['IssueTime'] = (df['IssueTime']/100).astype(int)
df['IssueTime'] = pd.to_datetime(df['IssueTime'],format= '%H')
#We only want the time element
df['IssueTime']= df['IssueTime'].dt.time
#We need to establish a relevant datetime variable for the Issue
df['DateIssueTime'] = pd.to_datetime(df['Date'].astype(str)+ ' ' + df['IssueTime'].astype(str))
#Now we need to establish the leadtime
df['Leadtime']= pd.to_timedelta(df['Time(Hr)'], 'h')
#Apply a relevant datetime to each leadtime
df['DateTime'] = df['DateIssueTime'] + df['Leadtime']

#################################################################################
# Select only the dates we need
#List out the unique dates
dates = df['DateTime'].unique()
# Get the first date and set the time to 11pm.
start_date = dates.astype('datetime64[ns]')[0] + np.timedelta64(LEAD, 'h')
# Get the end time by adding 24hrs to start_date.
end_date = start_date + np.timedelta64(24, 'h')
# Log the start and end datetimes.
log.info("Datetimes between " + str(start_date) + " and " + str(end_date))
# Create a mask to select dates from start_date (incl) to end_date (excl).
mask = (df['DateTime'] >= start_date) & (df['DateTime'] < end_date)
# Get the records.
df = df.loc[mask]
#################################################################################

# Target the variables we want
dfU = df.loc[df['Param(10m)']== 'u']
dfU = dfU[['DateTime', 'Value(m/s)']]
dfU.columns = ['DateTime', 'U_10']
dfU.columns = ['DateTime', 'U_10']
dfU = dfU.set_index('DateTime')
#print(dfU.head())

dfV = df.loc[df['Param(10m)']== 'v']
dfV = dfV[['DateTime', 'Value(m/s)']]
dfV.columns = ['DateTime', 'V_10']
dfV= dfV.set_index('DateTime')
#print(dfV.head())

dfUG = df.loc[df['Param(10m)']== 'ugst']
dfUG = dfUG[['DateTime', 'Value(m/s)']]
dfUG.columns = ['DateTime', 'UG_10']
dfUG = dfUG.set_index('DateTime')
#print(dfUG.head())

dfVG = df.loc[df['Param(10m)']== 'vgst']
dfVG = dfVG[['DateTime', 'Value(m/s)']]
dfVG.columns = ['DateTime', 'VG_10']
dfVG = dfVG.set_index('DateTime')
#print(dfVG.head())

dfWDIR = df.loc[df['Param(10m)']== 'wdir']
dfWDIR = dfWDIR[['DateTime', 'Value(m/s)']]
dfWDIR.columns = ['DateTime', 'WDir_10']
dfWDIR = dfWDIR.set_index('DateTime')
#print(dfWDIR.head())

# Merge the subframes into one dataframe
df_merged = dfU.merge(dfV, how = 'outer', left_index=True, right_index=True)
df_merged = df_merged.merge(dfUG, how = 'outer', left_index=True, right_index=True)
df_merged = df_merged.merge(dfVG, how = 'outer', left_index=True, right_index=True)
df_merged = df_merged.merge(dfWDIR, how = 'outer', left_index=True, right_index=True)
#print(df_merged.head()

# Add the extra calculations.
df_merged['10mWS'] = np.sqrt(df_merged['U_10'].pow(2) + df_merged['V_10'].pow(2))
df_merged['10mWSg'] = np.sqrt(df_merged['UG_10'].pow(2) + df_merged['VG_10'].pow(2))
df_merged = df_merged.round(decimals=2)

df_merged.to_csv(r"/home/justi/Desktop/NewGurteenForecast/IREPS/DAM/DAM_IREPS_Forecasts/Member012_Converted.csv", mode = 'a', index = True, header = HEADER_VAR)
#df_merged.to_csv(r"Data\Members_Aggregated\Member011_Converted.csv", mode = 'a', index = True, header = HEADER_VAR)

log.info("Exported Member011 as CSV file successfully.")

M11 = df_merged[['U_10', 'V_10','10mWS']]
M11.rename(columns={'V_10':'M11_V10','U_10':'M11_U10','10mWS': 'M11_10mWS'}, inplace=True)

#######################################################################################
######## MEMBER012 #####################################################################
#######################################################################################

mail.select(r'Member012')
log.info("Member012 Folder selected.")

# Get the list of email ids.
_, emaillist = mail.uid(r'search', None, r'(X-GM-RAW " has:attachment newer_than:1d")')

# print(emaillist)

# Pick the latest id.
latestid = emaillist[0].split()[-1]
# Fetch that email.
_, raw_email = mail.uid(r'fetch', latestid, r'(RFC822)')
raw_email = raw_email[0][1].decode('utf-8')
log.info("Latest email selected.")

csv_file = None
# Parse the email
met_email = email.message_from_string(raw_email)
for part in met_email.walk():
    # multipart/* are just containers
    if part.get_content_maintype() == 'multipart':
        continue
    filename = part.get_filename()
    if filename:
        log.info("Found filename '" + filename + "' in email.")
        csv_file = part.get_payload(decode=True)

# Make sure we got something.
assert csv_file, "Failed to extract CSV file from the email."
log.info("Found CSV file.")

##################################################################################

# Pretend csv_file is a file, as opposed to a string.
with io.StringIO(csv_file.decode('utf-8')) as f:
  df =  pd.read_csv(f, usecols=[0,1,2,3,4],header = 0, names=['Date', 'FCT_Run(Z)', 'Param(10m)', 'Time(Hr)', 'Value(m/s)'])

#Ensure that the date is in the correct format
df['Date'] = pd.to_datetime(df['Date'])
#Identify the Issue Time
df['IssueTime'] = df.iloc[:, 1]
#As the time is in integer formate we need to change and convert
df['IssueTime'] = (df['IssueTime']/100).astype(int)
df['IssueTime'] = pd.to_datetime(df['IssueTime'],format= '%H')
#We only want the time element
df['IssueTime']= df['IssueTime'].dt.time
#We need to establish a relevant datetime variable for the Issue
df['DateIssueTime'] = pd.to_datetime(df['Date'].astype(str)+ ' ' + df['IssueTime'].astype(str))
#Now we need to establish the leadtime
df['Leadtime']= pd.to_timedelta(df['Time(Hr)'], 'h')
#Apply a relevant datetime to each leadtime
df['DateTime'] = df['DateIssueTime'] + df['Leadtime']

#################################################################################
# Select only the dates we need
#List out the unique dates
dates = df['DateTime'].unique()
# Get the first date and set the time to 11pm.
start_date = dates.astype('datetime64[ns]')[0] + np.timedelta64(LEAD, 'h')
# Get the end time by adding 24hrs to start_date.
end_date = start_date + np.timedelta64(24, 'h')
# Log the start and end datetimes.
log.info("Datetimes between " + str(start_date) + " and " + str(end_date))
# Create a mask to select dates from start_date (incl) to end_date (excl).
mask = (df['DateTime'] >= start_date) & (df['DateTime'] < end_date)
# Get the records.
df = df.loc[mask]
#################################################################################

# Target the variables we want
dfU = df.loc[df['Param(10m)']== 'u']
dfU = dfU[['DateTime', 'Value(m/s)']]
dfU.columns = ['DateTime', 'U_10']
dfU.columns = ['DateTime', 'U_10']
dfU = dfU.set_index('DateTime')
#print(dfU.head())

dfV = df.loc[df['Param(10m)']== 'v']
dfV = dfV[['DateTime', 'Value(m/s)']]
dfV.columns = ['DateTime', 'V_10']
dfV= dfV.set_index('DateTime')
#print(dfV.head())

dfUG = df.loc[df['Param(10m)']== 'ugst']
dfUG = dfUG[['DateTime', 'Value(m/s)']]
dfUG.columns = ['DateTime', 'UG_10']
dfUG = dfUG.set_index('DateTime')
#print(dfUG.head())

dfVG = df.loc[df['Param(10m)']== 'vgst']
dfVG = dfVG[['DateTime', 'Value(m/s)']]
dfVG.columns = ['DateTime', 'VG_10']
dfVG = dfVG.set_index('DateTime')
#print(dfVG.head())

dfWDIR = df.loc[df['Param(10m)']== 'wdir']
dfWDIR = dfWDIR[['DateTime', 'Value(m/s)']]
dfWDIR.columns = ['DateTime', 'WDir_10']
dfWDIR = dfWDIR.set_index('DateTime')
#print(dfWDIR.head())

# Merge the subframes into one dataframe
df_merged = dfU.merge(dfV, how = 'outer', left_index=True, right_index=True)
df_merged = df_merged.merge(dfUG, how = 'outer', left_index=True, right_index=True)
df_merged = df_merged.merge(dfVG, how = 'outer', left_index=True, right_index=True)
df_merged = df_merged.merge(dfWDIR, how = 'outer', left_index=True, right_index=True)
#print(df_merged.head()

# Add the extra calculations.
df_merged['10mWS'] = np.sqrt(df_merged['U_10'].pow(2) + df_merged['V_10'].pow(2))
df_merged['10mWSg'] = np.sqrt(df_merged['UG_10'].pow(2) + df_merged['VG_10'].pow(2))
df_merged = df_merged.round(decimals=2)

df_merged.to_csv(r"/home/justi/Desktop/NewGurteenForecast/IREPS/DAM/DAM_IREPS_Forecasts/Member012_Converted.csv", mode = 'a', index = True, header = HEADER_VAR)
#df_merged.to_csv(r"Data\Members_Aggregated\Member012_Converted.csv", mode = 'a', index = True, header = HEADER_VAR)

log.info("Exported Member012 as CSV file successfully.")

M12 = df_merged[['U_10', 'V_10','10mWS']]
M12.rename(columns={'V_10':'M12_V10','U_10':'M12_U10','10mWS': 'M12_10mWS'}, inplace=True)

#######################################################################################
######## MEMBER013 #####################################################################
#######################################################################################

mail.select(r'Member013')
log.info("Member013 Folder selected.")

# Get the list of email ids.
_, emaillist = mail.uid(r'search', None, r'(X-GM-RAW " has:attachment newer_than:1d")')

# print(emaillist)

# Pick the latest id.
latestid = emaillist[0].split()[-1]
# Fetch that email.
_, raw_email = mail.uid(r'fetch', latestid, r'(RFC822)')
raw_email = raw_email[0][1].decode('utf-8')
log.info("Latest email selected.")

csv_file = None
# Parse the email
met_email = email.message_from_string(raw_email)
for part in met_email.walk():
    # multipart/* are just containers
    if part.get_content_maintype() == 'multipart':
        continue
    filename = part.get_filename()
    if filename:
        log.info("Found filename '" + filename + "' in email.")
        csv_file = part.get_payload(decode=True)

# Make sure we got something.
assert csv_file, "Failed to extract CSV file from the email."
log.info("Found CSV file.")

##################################################################################

# Pretend csv_file is a file, as opposed to a string.
with io.StringIO(csv_file.decode('utf-8')) as f:
  df =  pd.read_csv(f, usecols=[0,1,2,3,4],header = 0, names=['Date', 'FCT_Run(Z)', 'Param(10m)', 'Time(Hr)', 'Value(m/s)'])

#Ensure that the date is in the correct format
df['Date'] = pd.to_datetime(df['Date'])
#Identify the Issue Time
df['IssueTime'] = df.iloc[:, 1]
#As the time is in integer formate we need to change and convert
df['IssueTime'] = (df['IssueTime']/100).astype(int)
df['IssueTime'] = pd.to_datetime(df['IssueTime'],format= '%H')
#We only want the time element
df['IssueTime']= df['IssueTime'].dt.time
#We need to establish a relevant datetime variable for the Issue
df['DateIssueTime'] = pd.to_datetime(df['Date'].astype(str)+ ' ' + df['IssueTime'].astype(str))
#Now we need to establish the leadtime
df['Leadtime']= pd.to_timedelta(df['Time(Hr)'], 'h')
#Apply a relevant datetime to each leadtime
df['DateTime'] = df['DateIssueTime'] + df['Leadtime']

#################################################################################
# Select only the dates we need
#List out the unique dates
dates = df['DateTime'].unique()
# Get the first date and set the time to 11pm.
start_date = dates.astype('datetime64[ns]')[0] + np.timedelta64(LEAD, 'h')
# Get the end time by adding 24hrs to start_date.
end_date = start_date + np.timedelta64(24, 'h')
# Log the start and end datetimes.
log.info("Datetimes between " + str(start_date) + " and " + str(end_date))
# Create a mask to select dates from start_date (incl) to end_date (excl).
mask = (df['DateTime'] >= start_date) & (df['DateTime'] < end_date)
# Get the records.
df = df.loc[mask]
#################################################################################

# Target the variables we want
dfU = df.loc[df['Param(10m)']== 'u']
dfU = dfU[['DateTime', 'Value(m/s)']]
dfU.columns = ['DateTime', 'U_10']
dfU.columns = ['DateTime', 'U_10']
dfU = dfU.set_index('DateTime')
#print(dfU.head())

dfV = df.loc[df['Param(10m)']== 'v']
dfV = dfV[['DateTime', 'Value(m/s)']]
dfV.columns = ['DateTime', 'V_10']
dfV= dfV.set_index('DateTime')
#print(dfV.head())

dfUG = df.loc[df['Param(10m)']== 'ugst']
dfUG = dfUG[['DateTime', 'Value(m/s)']]
dfUG.columns = ['DateTime', 'UG_10']
dfUG = dfUG.set_index('DateTime')
#print(dfUG.head())

dfVG = df.loc[df['Param(10m)']== 'vgst']
dfVG = dfVG[['DateTime', 'Value(m/s)']]
dfVG.columns = ['DateTime', 'VG_10']
dfVG = dfVG.set_index('DateTime')
#print(dfVG.head())

dfWDIR = df.loc[df['Param(10m)']== 'wdir']
dfWDIR = dfWDIR[['DateTime', 'Value(m/s)']]
dfWDIR.columns = ['DateTime', 'WDir_10']
dfWDIR = dfWDIR.set_index('DateTime')
#print(dfWDIR.head())

# Merge the subframes into one dataframe
df_merged = dfU.merge(dfV, how = 'outer', left_index=True, right_index=True)
df_merged = df_merged.merge(dfUG, how = 'outer', left_index=True, right_index=True)
df_merged = df_merged.merge(dfVG, how = 'outer', left_index=True, right_index=True)
df_merged = df_merged.merge(dfWDIR, how = 'outer', left_index=True, right_index=True)
#print(df_merged.head()

# Add the extra calculations.
df_merged['10mWS'] = np.sqrt(df_merged['U_10'].pow(2) + df_merged['V_10'].pow(2))
df_merged['10mWSg'] = np.sqrt(df_merged['UG_10'].pow(2) + df_merged['VG_10'].pow(2))
df_merged = df_merged.round(decimals=2)

df_merged.to_csv(r"/home/justi/Desktop/NewGurteenForecast/IREPS/DAM/DAM_IREPS_Forecasts/Member013_Converted.csv", mode = 'a', index = True, header = HEADER_VAR)
#df_merged.to_csv(r"Data\Members_Aggregated\Member013_Converted.csv", mode = 'a', index = True, header = HEADER_VAR)

log.info("Exported Member013 as CSV file successfully.")

M13 = df_merged[['U_10', 'V_10','10mWS']]
M13.rename(columns={'V_10':'M13_V10','U_10':'M13_U10','10mWS': 'M13_10mWS'}, inplace=True)

#######################################################################################
######## MEMBER014 #####################################################################
#######################################################################################

mail.select(r'Member014')
log.info("Member014 Folder selected.")

# Get the list of email ids.
_, emaillist = mail.uid(r'search', None, r'(X-GM-RAW " has:attachment newer_than:1d")')

# print(emaillist)

# Pick the latest id.
latestid = emaillist[0].split()[-1]
# Fetch that email.
_, raw_email = mail.uid(r'fetch', latestid, r'(RFC822)')
raw_email = raw_email[0][1].decode('utf-8')
log.info("Latest email selected.")

csv_file = None
# Parse the email
met_email = email.message_from_string(raw_email)
for part in met_email.walk():
    # multipart/* are just containers
    if part.get_content_maintype() == 'multipart':
        continue
    filename = part.get_filename()
    if filename:
        log.info("Found filename '" + filename + "' in email.")
        csv_file = part.get_payload(decode=True)

# Make sure we got something.
assert csv_file, "Failed to extract CSV file from the email."
log.info("Found CSV file.")

##################################################################################

# Pretend csv_file is a file, as opposed to a string.
with io.StringIO(csv_file.decode('utf-8')) as f:
  df =  pd.read_csv(f, usecols=[0,1,2,3,4],header = 0, names=['Date', 'FCT_Run(Z)', 'Param(10m)', 'Time(Hr)', 'Value(m/s)'])

#Ensure that the date is in the correct format
df['Date'] = pd.to_datetime(df['Date'])
#Identify the Issue Time
df['IssueTime'] = df.iloc[:, 1]
#As the time is in integer formate we need to change and convert
df['IssueTime'] = (df['IssueTime']/100).astype(int)
df['IssueTime'] = pd.to_datetime(df['IssueTime'],format= '%H')
#We only want the time element
df['IssueTime']= df['IssueTime'].dt.time
#We need to establish a relevant datetime variable for the Issue
df['DateIssueTime'] = pd.to_datetime(df['Date'].astype(str)+ ' ' + df['IssueTime'].astype(str))
#Now we need to establish the leadtime
df['Leadtime']= pd.to_timedelta(df['Time(Hr)'], 'h')
#Apply a relevant datetime to each leadtime
df['DateTime'] = df['DateIssueTime'] + df['Leadtime']

#################################################################################
# Select only the dates we need
#List out the unique dates
dates = df['DateTime'].unique()
# Get the first date and set the time to 11pm.
start_date = dates.astype('datetime64[ns]')[0] + np.timedelta64(LEAD, 'h')
# Get the end time by adding 24hrs to start_date.
end_date = start_date + np.timedelta64(24, 'h')
# Log the start and end datetimes.
log.info("Datetimes between " + str(start_date) + " and " + str(end_date))
# Create a mask to select dates from start_date (incl) to end_date (excl).
mask = (df['DateTime'] >= start_date) & (df['DateTime'] < end_date)
# Get the records.
df = df.loc[mask]
#################################################################################

# Target the variables we want
dfU = df.loc[df['Param(10m)']== 'u']
dfU = dfU[['DateTime', 'Value(m/s)']]
dfU.columns = ['DateTime', 'U_10']
dfU.columns = ['DateTime', 'U_10']
dfU = dfU.set_index('DateTime')
#print(dfU.head())

dfV = df.loc[df['Param(10m)']== 'v']
dfV = dfV[['DateTime', 'Value(m/s)']]
dfV.columns = ['DateTime', 'V_10']
dfV= dfV.set_index('DateTime')
#print(dfV.head())

dfUG = df.loc[df['Param(10m)']== 'ugst']
dfUG = dfUG[['DateTime', 'Value(m/s)']]
dfUG.columns = ['DateTime', 'UG_10']
dfUG = dfUG.set_index('DateTime')
#print(dfUG.head())

dfVG = df.loc[df['Param(10m)']== 'vgst']
dfVG = dfVG[['DateTime', 'Value(m/s)']]
dfVG.columns = ['DateTime', 'VG_10']
dfVG = dfVG.set_index('DateTime')
#print(dfVG.head())

dfWDIR = df.loc[df['Param(10m)']== 'wdir']
dfWDIR = dfWDIR[['DateTime', 'Value(m/s)']]
dfWDIR.columns = ['DateTime', 'WDir_10']
dfWDIR = dfWDIR.set_index('DateTime')
#print(dfWDIR.head())

# Merge the subframes into one dataframe
df_merged = dfU.merge(dfV, how = 'outer', left_index=True, right_index=True)
df_merged = df_merged.merge(dfUG, how = 'outer', left_index=True, right_index=True)
df_merged = df_merged.merge(dfVG, how = 'outer', left_index=True, right_index=True)
df_merged = df_merged.merge(dfWDIR, how = 'outer', left_index=True, right_index=True)
#print(df_merged.head()

# Add the extra calculations.
df_merged['10mWS'] = np.sqrt(df_merged['U_10'].pow(2) + df_merged['V_10'].pow(2))
df_merged['10mWSg'] = np.sqrt(df_merged['UG_10'].pow(2) + df_merged['VG_10'].pow(2))
df_merged = df_merged.round(decimals=2)

df_merged.to_csv(r"/home/justi/Desktop/NewGurteenForecast/IREPS/DAM/DAM_IREPS_Forecasts/Member014_Converted.csv", mode = 'a', index = True, header = HEADER_VAR)
#df_merged.to_csv(r"Data\Members_Aggregated\Member014_Converted.csv", mode = 'a', index = True, header = HEADER_VAR)

log.info("Exported Member014 as CSV file successfully.")

M14 = df_merged[['U_10', 'V_10','10mWS']]
M14.rename(columns={'V_10':'M14_V10','U_10':'M14_U10','10mWS': 'M14_10mWS'}, inplace=True)

#######################################################################################
######## MEMBER015 #####################################################################
#######################################################################################

mail.select(r'Member015')
log.info("Member015 Folder selected.")

# Get the list of email ids.
_, emaillist = mail.uid(r'search', None, r'(X-GM-RAW " has:attachment newer_than:1d")')

# print(emaillist)

# Pick the latest id.
latestid = emaillist[0].split()[-1]
# Fetch that email.
_, raw_email = mail.uid(r'fetch', latestid, r'(RFC822)')
raw_email = raw_email[0][1].decode('utf-8')
log.info("Latest email selected.")

csv_file = None
# Parse the email
met_email = email.message_from_string(raw_email)
for part in met_email.walk():
    # multipart/* are just containers
    if part.get_content_maintype() == 'multipart':
        continue
    filename = part.get_filename()
    if filename:
        log.info("Found filename '" + filename + "' in email.")
        csv_file = part.get_payload(decode=True)

# Make sure we got something.
assert csv_file, "Failed to extract CSV file from the email."
log.info("Found CSV file.")

##################################################################################

# Pretend csv_file is a file, as opposed to a string.
with io.StringIO(csv_file.decode('utf-8')) as f:
  df =  pd.read_csv(f, usecols=[0,1,2,3,4],header = 0, names=['Date', 'FCT_Run(Z)', 'Param(10m)', 'Time(Hr)', 'Value(m/s)'])

#Ensure that the date is in the correct format
df['Date'] = pd.to_datetime(df['Date'])
#Identify the Issue Time
df['IssueTime'] = df.iloc[:, 1]
#As the time is in integer formate we need to change and convert
df['IssueTime'] = (df['IssueTime']/100).astype(int)
df['IssueTime'] = pd.to_datetime(df['IssueTime'],format= '%H')
#We only want the time element
df['IssueTime']= df['IssueTime'].dt.time
#We need to establish a relevant datetime variable for the Issue
df['DateIssueTime'] = pd.to_datetime(df['Date'].astype(str)+ ' ' + df['IssueTime'].astype(str))
#Now we need to establish the leadtime
df['Leadtime']= pd.to_timedelta(df['Time(Hr)'], 'h')
#Apply a relevant datetime to each leadtime
df['DateTime'] = df['DateIssueTime'] + df['Leadtime']

#################################################################################
# Select only the dates we need
#List out the unique dates
dates = df['DateTime'].unique()
# Get the first date and set the time to 11pm.
start_date = dates.astype('datetime64[ns]')[0] + np.timedelta64(LEAD, 'h')
# Get the end time by adding 24hrs to start_date.
end_date = start_date + np.timedelta64(24, 'h')
# Log the start and end datetimes.
log.info("Datetimes between " + str(start_date) + " and " + str(end_date))
# Create a mask to select dates from start_date (incl) to end_date (excl).
mask = (df['DateTime'] >= start_date) & (df['DateTime'] < end_date)
# Get the records.
df = df.loc[mask]
#################################################################################

# Target the variables we want
dfU = df.loc[df['Param(10m)']== 'u']
dfU = dfU[['DateTime', 'Value(m/s)']]
dfU.columns = ['DateTime', 'U_10']
dfU.columns = ['DateTime', 'U_10']
dfU = dfU.set_index('DateTime')
#print(dfU.head())

dfV = df.loc[df['Param(10m)']== 'v']
dfV = dfV[['DateTime', 'Value(m/s)']]
dfV.columns = ['DateTime', 'V_10']
dfV= dfV.set_index('DateTime')
#print(dfV.head())

dfUG = df.loc[df['Param(10m)']== 'ugst']
dfUG = dfUG[['DateTime', 'Value(m/s)']]
dfUG.columns = ['DateTime', 'UG_10']
dfUG = dfUG.set_index('DateTime')
#print(dfUG.head())

dfVG = df.loc[df['Param(10m)']== 'vgst']
dfVG = dfVG[['DateTime', 'Value(m/s)']]
dfVG.columns = ['DateTime', 'VG_10']
dfVG = dfVG.set_index('DateTime')
#print(dfVG.head())

dfWDIR = df.loc[df['Param(10m)']== 'wdir']
dfWDIR = dfWDIR[['DateTime', 'Value(m/s)']]
dfWDIR.columns = ['DateTime', 'WDir_10']
dfWDIR = dfWDIR.set_index('DateTime')
#print(dfWDIR.head())

# Merge the subframes into one dataframe
df_merged = dfU.merge(dfV, how = 'outer', left_index=True, right_index=True)
df_merged = df_merged.merge(dfUG, how = 'outer', left_index=True, right_index=True)
df_merged = df_merged.merge(dfVG, how = 'outer', left_index=True, right_index=True)
df_merged = df_merged.merge(dfWDIR, how = 'outer', left_index=True, right_index=True)
#print(df_merged.head()

# Add the extra calculations.
df_merged['10mWS'] = np.sqrt(df_merged['U_10'].pow(2) + df_merged['V_10'].pow(2))
df_merged['10mWSg'] = np.sqrt(df_merged['UG_10'].pow(2) + df_merged['VG_10'].pow(2))
df_merged = df_merged.round(decimals=2)
#print(df_merged)

df_merged.to_csv(r"/home/justi/Desktop/NewGurteenForecast/IREPS/DAM/DAM_IREPS_Forecasts/Member015_Converted.csv", mode = 'a', index = True, header = HEADER_VAR)
#df_merged.to_csv(r"Data\Members_Aggregated\Member015_Converted.csv", mode = 'a', index = True, header = HEADER_VAR)

log.info("Exported Member015 as CSV file successfully.")

M15 = df_merged[['U_10', 'V_10','10mWS']]
M15.rename(columns={'V_10':'M15_V10','U_10':'M15_U10','10mWS': 'M15_10mWS'}, inplace=True)

#############################################################################################################
### NOW WE ACCESS THE HARMONIE FORECAST  ####################################################################
#############################################################################################################

# Select the inbox.
mail.select(r'inbox')
log.info("IMAP inbox selected.")

# Get the list of email ids.
_, emaillist = mail.uid(r'search', None, r'(X-GM-RAW "from: opmet@met.ie has:attachment newer_than:3d")')


# Pick the latest id.
latestid = emaillist[0].split()[-1]
# Fetch that email.
_, raw_email = mail.uid(r'fetch', latestid, r'(RFC822)')
raw_email = raw_email[0][1].decode('utf-8')
log.info("Latest email selected.")

################################################################################
## Extract CSV from email.
################################################################################
csv_file = None
# Parse the email
met_email = email.message_from_string(raw_email)
for part in met_email.walk():
    # multipart/* are just containers
    if part.get_content_maintype() == 'multipart':
        continue
    filename = part.get_filename()
    if filename:
        log.info("Found filename '" + filename + "' in email.")
        csv_file = part.get_payload(decode=True)

# Make sure we got something.
assert csv_file, "Failed to extract CSV file from the email."
log.info("Found CSV file.")

################################################################################
## Parse the CSV file.
################################################################################

# Clean the CSV file.
# Pretend csv_file is a file, as opposed to a string.
with io.StringIO(csv_file.decode('utf-8')) as f:
  # Seperate the file into lines and discard empty lines.
  csv_file = [l for l in f.readlines() if l.strip()]
  # Ignore the first 36 lines of the file.
  csv_file = csv_file[36:]
  csv_file = r''.join(csv_file)

# Pretend csv_file is a file, as opposed to a string.
with io.StringIO(csv_file) as f:
  # Read the file into a pandas data frame.
  df =  pd.read_csv(f)

################################################################################
## Select the correct data.
################################################################################
#print(df.columns)

# Add the extra calculations.
df['WS_10'] = np.sqrt(df['U10Wind'].pow(2) + df['V10Wind'].pow(2)).round(decimals=1)
df['WS_50'] = np.sqrt(df['U50Wind'].pow(2) + df['V50Wind'].pow(2)).round(decimals=1)
df['WS_100'] = np.sqrt(df['U100Wind'].pow(2) + df['V100Wind'].pow(2)).round(decimals=1)

df = df[['Day',' Hour', ' 2m Temp', ' SolRad', ' Rain', 'Press', 'RelHum', 'V10Wind', 'U10Wind', 'V50Wind', 'U50Wind', 'V100Wind', 'U100Wind', 'V305Wind', 'U305Wind', 'T50Temp', 'T100Temp', 'T305Temp', 'WS_10', 'WS_50', 'WS_100']]


################################################################################
#Add the latest Harmonie forcast to the training set
df.to_csv(r"/home/justi/Desktop/NewGurteenForecast/Aggregated_Met_Variables_for_training.csv", mode = 'a', index = False, header = False)
log.info("Latest meteorological variables added to training set")
################################################################################

# Select the appropriate datetimes.
# Create a new datetime field in the data frame.
df['Date'] = pd.to_datetime(df['Day'] + ' ' + df[' Hour'])
# Get a list of unique dates from the data frame.
dates = df['Day'].unique()
# Sort the dates (they're likely sorted already.)
dates.sort()
# Get the first date and set the time to 11pm.
start_date = dates.astype('datetime64[ns]')[0] + np.timedelta64(23, 'h')
# Get the end time by adding 24hrs to start_date.
end_date = start_date + np.timedelta64(24, 'h')
# Log the start and end datetimes.
log.info("Datetimes between " + str(start_date) + " and " + str(end_date))
# Create a mask to select dates from start_date (incl) to end_date (excl).
mask = (df['Date'] >= start_date) & (df['Date'] < end_date)
# Get the records.
df = df.loc[mask]
#print(df.columns)
# Select the Data we require.
df = df[['Date', ' 2m Temp', ' SolRad', ' Rain', 'Press', 'RelHum', 'V10Wind', 'U10Wind', 'V50Wind', 'U50Wind', 'V100Wind', 'U100Wind', 'V305Wind', 'U305Wind', 'T50Temp', 'T100Temp', 'T305Temp','WS_10', 'WS_50', 'WS_100']]
# Reset the index.
#df.reset_index()


################################################################################
## Forecast the power production.
################################################################################

mod = pd.read_csv('E70_power_curve_model.csv')
f = scipy.interpolate.interp1d(mod['Wind'].values, mod['Power'].values, kind='cubic')
df['Power'] = df['WS_100'].apply(f)
df = df.round(decimals=2)
#print(df)
df.to_csv(r"/home/justi/Desktop/NewGurteenForecast/HARMONIE/DAM/Harmonie.csv", mode = 'a', index = False, header = HEADER_VAR)

log.info("Exported Harmonie as CSV file successfully.")

Har = df[['Date', 'V10Wind', 'U10Wind', 'WS_10', 'V100Wind', 'U100Wind', 'WS_100','RelHum', 'Press', ' Rain', 'Power']]
Har =Har.set_index('Date')
Har.rename(columns={'V10Wind':'Har_V10','U10Wind':'Har_U10','WS_10': 'Har_10mWS','V100Wind': 'Har_V100', 'U100Wind' : 'Har_U100', 'WS_100' : 'Har_WS100','RelHum': 'Har_RelHum', 'Press': 'Har_Press', ' Rain' : 'Har_Rain', 'Power': 'Har_Power'}, inplace=True)

# Now we will colate the Members data into a single dataframe.

Members = [M0, M1, M2, M3, M4, M5, M6, M7, M8, M9, M10, M11, M12, M13, M14, M15, Har]

Members_Merged = reduce(lambda left,right: pd.merge(left, right, left_index = True, right_index=True, how='outer'), Members)

Members_Merged = Members_Merged.round(decimals=1)

Members_Merged.to_csv(r"/home/justi/Desktop/NewGurteenForecast/IREPS/DAM/AGGREGATE_DAM_IREPS_HARMONIE_All_Members.csv", mode = 'a', index = True, header = HEADER_VAR)


#####################################################################################
#####################################################################################
### NOW WE MOVE TO THE MACHINE LEARNING PHASE TO BUILD OUR ENSEMBLE 100M FORECASTS###
#####################################################################################

#Create the 'New' set of ensemble Forecasts at 10m to be used for training purposes.
For0_10m = Members_Merged[['M0_V10', 'M0_U10', 'M0_10mWS', 'Har_RelHum', 'Har_Press', 'Har_Rain']]
#print(For0_10m)
For1_10m = Members_Merged[['M1_V10', 'M1_U10', 'M1_10mWS', 'Har_RelHum', 'Har_Press', 'Har_Rain']]
For2_10m = Members_Merged[['M2_V10', 'M2_U10', 'M2_10mWS', 'Har_RelHum', 'Har_Press', 'Har_Rain']]
For3_10m = Members_Merged[['M3_V10', 'M3_U10', 'M3_10mWS', 'Har_RelHum', 'Har_Press', 'Har_Rain']]
For4_10m = Members_Merged[['M4_V10', 'M4_U10', 'M4_10mWS', 'Har_RelHum', 'Har_Press', 'Har_Rain']]
For5_10m = Members_Merged[['M5_V10', 'M5_U10', 'M5_10mWS', 'Har_RelHum', 'Har_Press', 'Har_Rain']]
For6_10m = Members_Merged[['M6_V10', 'M6_U10', 'M6_10mWS', 'Har_RelHum', 'Har_Press', 'Har_Rain']]
For7_10m = Members_Merged[['M7_V10', 'M7_U10', 'M7_10mWS', 'Har_RelHum', 'Har_Press', 'Har_Rain']]
For8_10m = Members_Merged[['M8_V10', 'M8_U10', 'M8_10mWS', 'Har_RelHum', 'Har_Press', 'Har_Rain']]
For9_10m = Members_Merged[['M9_V10', 'M9_U10', 'M9_10mWS', 'Har_RelHum', 'Har_Press', 'Har_Rain']]
For10_10m = Members_Merged[['M10_V10', 'M10_U10', 'M10_10mWS', 'Har_RelHum', 'Har_Press', 'Har_Rain']]
For11_10m = Members_Merged[['M11_V10', 'M11_U10', 'M11_10mWS', 'Har_RelHum', 'Har_Press', 'Har_Rain']]
For12_10m = Members_Merged[['M12_V10', 'M12_U10', 'M12_10mWS', 'Har_RelHum', 'Har_Press', 'Har_Rain']]
For13_10m = Members_Merged[['M13_V10', 'M13_U10', 'M13_10mWS', 'Har_RelHum', 'Har_Press', 'Har_Rain']]
For14_10m = Members_Merged[['M14_V10', 'M14_U10', 'M14_10mWS', 'Har_RelHum', 'Har_Press', 'Har_Rain']]
For15_10m = Members_Merged[['M15_V10', 'M15_U10', 'M15_10mWS', 'Har_RelHum', 'Har_Press', 'Har_Rain']]
#print(For15_10m)

print('The algorithm will now run the machine learning on the training set and may take up to 60s - so dont worry about a flashing cursor!!!')

train_set= pd.read_csv('Met_Variables_for_Training.csv')
X = train_set.iloc[:,[7,8,18,6,5,4]]
#print(X.head())

y = train_set['WS_100']
from sklearn.preprocessing import Normalizer
X = Normalizer().fit_transform(X)

# To avoid overfitting, we use a random split of the dataset to divide the variables into a training set and a testing or validation set.
# For the purpose of this model we have selected to divide the dataset into a 7:3 training-testing ratio.
from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split(X, y,train_size=0.9, test_size=0.1, random_state=1)
# Note, the random argument controls the randomness of the estimator - but its value is a an integer, wich is used to seed the random generator.
# The random generator therefore starts at the same position every time, so producing the same results.

# As this is a Regression problem, we us an appropriate Regression algorithm.
from sklearn.ensemble import RandomForestRegressor
RFR_model = RandomForestRegressor(random_state=1)
# We now fit the input data to the output data fro the training set.
RFR_model.fit(train_X, train_y)

y_pred = RFR_model.predict(test_X)

from sklearn.metrics import r2_score
#Check the score
score = r2_score(test_y, y_pred)*100
print("R2 Score", score)

For0_10m = Normalizer().fit_transform(For0_10m)
For1_10m = Normalizer().fit_transform(For1_10m)
For2_10m = Normalizer().fit_transform(For2_10m)
For3_10m = Normalizer().fit_transform(For3_10m)
For4_10m = Normalizer().fit_transform(For4_10m)
For5_10m = Normalizer().fit_transform(For5_10m)
For6_10m = Normalizer().fit_transform(For6_10m)
For7_10m = Normalizer().fit_transform(For7_10m)
For8_10m = Normalizer().fit_transform(For8_10m)
For9_10m = Normalizer().fit_transform(For9_10m)
For10_10m =Normalizer().fit_transform(For10_10m)
For11_10m = Normalizer().fit_transform(For11_10m)
For12_10m = Normalizer().fit_transform(For12_10m)
For13_10m = Normalizer().fit_transform(For13_10m)
For14_10m = Normalizer().fit_transform(For14_10m)
For15_10m = Normalizer().fit_transform(For15_10m)


For0_100m = RFR_model.predict(For0_10m)
For1_100m = RFR_model.predict(For1_10m)
For2_100m = RFR_model.predict(For2_10m)
For3_100m = RFR_model.predict(For3_10m)
For4_100m = RFR_model.predict(For4_10m)
For5_100m = RFR_model.predict(For5_10m)
For6_100m = RFR_model.predict(For6_10m)
For7_100m = RFR_model.predict(For7_10m)
For8_100m = RFR_model.predict(For8_10m)
For9_100m = RFR_model.predict(For9_10m)
For10_100m = RFR_model.predict(For10_10m)
For11_100m = RFR_model.predict(For11_10m)
For12_100m = RFR_model.predict(For12_10m)
For13_100m = RFR_model.predict(For13_10m)
For14_100m = RFR_model.predict(For14_10m)
For15_100m = RFR_model.predict(For15_10m)
#print(For15_100m)

WForecasts = [For0_100m, For10_100m, For2_100m, For3_100m, For4_100m, For5_100m, For6_100m, For7_100m, For8_100m, For9_100m, For10_100m, For11_100m, For12_100m, For13_100m, For14_100m, For15_100m]
#print('OK')
#print(Members_Merged.index)

#We use the datetime from Members_Merged to rebuild the dataset.
newTime = Members_Merged.index.values

WS_100 = pd.DataFrame(WForecasts, index = ['WF0_100', 'WF1_100', 'WF2_100', 'WF3_100', 'WF4_100', 'WF5_100', 'WF6_100', 'WF7_100', 'WF8_100', 'WF9_100', 'WF10_100', 'WF11_100', 'WF12_100', 'WF13_100', 'WF14_100', 'WF15_100'], 
columns = (newTime))

WS_100 = WS_100.round(1)
#print(WS_100)
# Transpose for Aggregation but not for converting to power
trans_WS_100 = WS_100.transpose()
trans_WS_100.to_csv(r"/home/justi/Desktop/NewGurteenForecast/IREPS/DAM/Aggregated_WS_100m_Members.csv", mode = 'a', index = True, header = HEADER_VAR)
# Save day_only file
trans_WS_100.to_csv(daily_path + "/WS_Forcast_100m_ for_" + tomorrow + ".csv", mode = 'w', index= True, header = True)


# Plot the comparative wind speeds for reference
plt.rcParams["figure.figsize"] = [16,9]
plt.plot(Har[['Har_WS100']])
plt.plot(trans_WS_100[['WF0_100']])
plt.gca().xaxis.set_major_formatter(myFmt)
plt.xticks(rotation='horizontal')
plt.xlabel('Time', color='r', fontsize=15)
plt.ylabel('Wind m/s', color='r', fontsize=15)
plt.title('Day-Ahead Control Member WS Forecast Vs Trained Member at 100m ' + tomorrow, color='r', fontsize = 20)
plt.legend(['Control Member', 'Trained Member'])
plt.grid(linestyle='--', linewidth=0.5)
img_name = "/Day_Ahead_Control_vs_Trained_100m_Wind_Forecast_for_"
plt.savefig(img_path + img_name + tomorrow + ".png" )
plt.close()


# Plot the Modelled Ensemble 100m Wind Forecast
plt.rcParams["figure.figsize"] = [16,9]
plt.plot(trans_WS_100)
plt.gca().xaxis.set_major_formatter(myFmt)
plt.xticks(rotation='horizontal')
plt.xlabel('Time', color='r', fontsize=15)
plt.ylabel('Wind m/s', color='r', fontsize=15)
plt.title('Day-Ahead Ensemble Wind Forecast at 100m for ' + tomorrow, color='r', fontsize = 20)
plt.legend(['Member 0','Member 1', 'Member 2', 'Member 3', 'Member 4', 'Member 5','Member 6', 'Member 7', 'Member 8', 'Member 9', 'Member 10','Member 11', 'Member 12', 'Member 13', 'Member 14', 'Member 15'])
plt.grid(linestyle='--', linewidth=0.5)
img_name = "/Day_Ahead_Ensemble_Wind_Forecast_for_"
plt.savefig(img_path + img_name + tomorrow + ".png" )
plt.close()


#Interpret the values from the power curve and apply to the wind speeds
mod = pd.read_csv('E70_power_curve_model.csv')
f = scipy.interpolate.interp1d(mod['Wind'].values, mod['Power'].values, kind='cubic')
EnergyForecast = WS_100.apply(f)
EnergyForecast = EnergyForecast.round(1)
#print(EnergyForecast)

# Transpose to plot the datetime on the y-axis
EnergyForecast = EnergyForecast.transpose()
EnergyForecast = EnergyForecast.rename(columns={'WF0_100':'EF0_100', 'WF1_100':'EF1_100', 'WF2_100':'EF2_100', 'WF3_100':'EF3_100', 'WF4_100':'EF4_100', 'WF5_100':'EF5_100', 'WF6_100':'EF6_100', 'WF7_100':'EF7_100', 'WF8_100':'EF8_100', 'WF9_100':'EF9_100', 'WF10_100':'EF10_100', 'WF11_100':'EF11_100', 'WF12_100':'EF12_100', 'WF13_100':'EF13_100', 'WF14_100':'EF14_100', 'WF15_100':'EF15_100'})
#print('This is the energy forecast')
#print(EnergyForecast)
EnergyForecast.to_csv(r"/home/justi/Desktop/NewGurteenForecast/IREPS/DAM/Aggregated_EnergyForecast_All_Members.csv", mode = 'a', index = True, header = HEADER_VAR)
# Save day_only file
EnergyForecast.to_csv(daily_path + "/Energy_Forcast_100m_ for_" + tomorrow + ".csv", mode = 'w', index= True, header = True)


# Plot the Ensemble Energy Forecast
plt.rcParams["figure.figsize"] = [16,9]
plt.plot(EnergyForecast)
plt.gca().xaxis.set_major_formatter(myFmt)
plt.xticks(rotation='horizontal')
plt.xlabel('Time', color='r', fontsize=15)
plt.ylabel('Power kWh', color='r', fontsize=15)
plt.title('Day-Ahead Ensemble Power Forecast at 100m for ' + tomorrow, color='r', fontsize = 20)
plt.legend(['Member 0','Member 1', 'Member 2', 'Member 3', 'Member 4', 'Member 5','Member 6', 'Member 7', 'Member 8', 'Member 9', 'Member 10','Member 11', 'Member 12', 'Member 13', 'Member 14', 'Member 15'])
plt.grid(linestyle='--', linewidth=0.5)
img_name = "/Day_Ahead_Ensemble_Power_Forecast_for_"
plt.savefig(img_path + img_name + tomorrow + ".png" )
plt.close()

#Convert the Power to the Capacity Factor 
EnergyForecast_CF = EnergyForecast.divide(capacity_factor)
#To Convert to a percentage CF
EnergyForecast_CFP = EnergyForecast_CF.multiply(100)
EnergyForecast_CFP = EnergyForecast_CFP.round(3)
#print('This is the EnergyForecast_CF_by Percentage')
#print(EnergyForecast_CFP)
EnergyForecast_CF.to_csv(r"/home/justi/Desktop/NewGurteenForecast/IREPS/DAM/Aggregated_EnergyForecast_by_Cap_Factor_as%_All_Members.csv", mode = 'a', index = True, header = HEADER_VAR)
# Save day_only file
EnergyForecast_CF.to_csv(daily_path + "/Energy_Forcast_by_Capacity_100m_ for_" + tomorrow + ".csv", mode = 'w', index= True, header = True)

# Plot the Ensemble Energy Forecast by Capacity Factor
plt.rcParams["figure.figsize"] = [16,9]
plt.plot(EnergyForecast_CFP)
plt.gca().xaxis.set_major_formatter(myFmt)
plt.xticks(rotation='horizontal')
plt.xlabel('Time', color='r', fontsize=15)
plt.ylabel('Percentage Rated Capacity', color='r', fontsize=15)
plt.title('Day-Ahead Ensemble Power Forecast at 100m for ' + tomorrow, color='r', fontsize = 20)
plt.legend(['Member 0','Member 1', 'Member 2', 'Member 3', 'Member 4', 'Member 5','Member 6', 'Member 7', 'Member 8', 'Member 9', 'Member 10','Member 11', 'Member 12', 'Member 13', 'Member 14', 'Member 15'])
plt.grid(linestyle='--', linewidth=0.5)
img_name = "/Day_Ahead_Ensemble_Power_Forecast_by_Rated_Capacity_for_"
plt.savefig(img_path + img_name + tomorrow + ".png" )
plt.close()

#Create a Quantile Forecast for Energy
Quantile_Forecast = EnergyForecast.quantile([0.9,0.75,0.5,0.25,0.1], axis=1, interpolation = 'midpoint')
#print(Quantile_Forecast)

#Transpose to facilitate graphics with datetime as index
Quantile_Forecast1 = Quantile_Forecast.transpose()

# Merge with the Power forecast from the Control Member of the Harmonie Forecast
Quantile_Forecast1 = Quantile_Forecast1.join(Members_Merged['Har_Power'])

#Rename the columns
Quantile_Forecast1 = Quantile_Forecast1.rename(columns={0.9:'90th%', 0.75:'75th%', 0.5:'50th%', 0.25:'25th%', 0.1:'10th%', 'Har_Power': 'Control_Power'})

#print(Quantile_Forecast1)

# Create a graphic of the Quantile Forecast and include the Harmonie Power Forecast.
node=Quantile_Forecast1.index
plt.plot(Quantile_Forecast1['Control_Power'], 'go-')
plt.plot(Quantile_Forecast1['90th%'], '--')
plt.plot(Quantile_Forecast1['75th%'], '--')
plt.plot(Quantile_Forecast1['50th%'], 'ro-', linewidth=2.5)
plt.plot(Quantile_Forecast1['25th%'], '--')
plt.plot(Quantile_Forecast1['10th%'], '--')
plt.gca().xaxis.set_major_formatter(myFmt)
plt.xticks(rotation='horizontal')
plt.xlabel('Time', color='r', fontsize=15)
plt.ylabel('Power kWh', color='r', fontsize=15)
plt.legend(['Deterministic Forecast','90th%', '75th%', '50th%', '25th%', '10th%'])
plt.title('Day-Ahead Quantile & Deterministic Power Forecast for ' + tomorrow, color='r', fontsize = 20)
plt.fill_between(node, Quantile_Forecast1['90th%'], Quantile_Forecast1['75th%'],facecolor='cornflowerblue', alpha=0.5)
plt.fill_between(node, Quantile_Forecast1['75th%'], Quantile_Forecast1['50th%'],facecolor='cornflowerblue', alpha=1)
plt.fill_between(node, Quantile_Forecast1['50th%'], Quantile_Forecast1['25th%'],facecolor='cornflowerblue', alpha=1)
plt.fill_between(node, Quantile_Forecast1['25th%'], Quantile_Forecast1['10th%'],facecolor='cornflowerblue', alpha=0.5)
plt.grid(linestyle='--', linewidth=0.5)
img_name = "/Day_Ahead_Quantile_Deterministic_Power_Forecast_for_"
plt.savefig(img_path + img_name + tomorrow + ".png" )
plt.close()

# Do a Graphic without including the Deterministic Member
node=Quantile_Forecast1.index
plt.plot(Quantile_Forecast1['90th%'], '--')
plt.plot(Quantile_Forecast1['75th%'], '--')
plt.plot(Quantile_Forecast1['50th%'], 'ro-', linewidth=2.5)
plt.plot(Quantile_Forecast1['25th%'], '--')
plt.plot(Quantile_Forecast1['10th%'], '--')
plt.gca().xaxis.set_major_formatter(myFmt)
plt.xticks(rotation='horizontal')
plt.xlabel('Time', color='r', fontsize=15)
plt.ylabel('Power kWh', color='r', fontsize=15)
plt.legend(['90th%', '75th%', '50th%', '25th%', '10th%'])
plt.title('Day-Ahead Quantile Power Forecast for ' + tomorrow, color='r', fontsize = 20)
plt.fill_between(node, Quantile_Forecast1['90th%'], Quantile_Forecast1['75th%'],facecolor='cornflowerblue', alpha=0.5)
plt.fill_between(node, Quantile_Forecast1['75th%'], Quantile_Forecast1['50th%'],facecolor='cornflowerblue', alpha=1)
plt.fill_between(node, Quantile_Forecast1['50th%'], Quantile_Forecast1['25th%'],facecolor='cornflowerblue', alpha=1)
plt.fill_between(node, Quantile_Forecast1['25th%'], Quantile_Forecast1['10th%'],facecolor='cornflowerblue', alpha=0.5)
plt.grid(linestyle='--', linewidth=0.5)
img_name = "/Day_Ahead_Quantile_Power_Forecast_for_"
plt.savefig(img_path + img_name + tomorrow + ".png" )
plt.close()

#pd.DataFrame(Quantile_Forecast1)
Quantile_Forecast1 = (Quantile_Forecast1).round(2)
Quantile_Forecast1.to_csv(r"/home/justi/Desktop/NewGurteenForecast/Quantiles/Aggregated_DAM_Quantile_Energy_Forecast.csv", mode = 'a', index = True, header = HEADER_VAR)
# Save day_only file
Quantile_Forecast1.to_csv(daily_path + "/Quantile_Power_Forcast_100m_ for_" + tomorrow + ".csv", mode = 'w', index= True, header = True)

#Convert the Power from the Quantile Forecast to a Percentage of Rated Capacity 
Quantile_Forecast1_CF = Quantile_Forecast1.divide(capacity_factor)
#To Convert to a percentage CF
Quantile_Forecast1_CFP = Quantile_Forecast1_CF.multiply(100)
Quantile_Forecast1_CFP = Quantile_Forecast1_CFP.round(1)
#print('This is the Quantile Power Forecast_CF_by Percentage')
#print(Quantile_Forecast1_CFP)
Quantile_Forecast1_CFP.to_csv(r"/home/justi/Desktop/NewGurteenForecast/Quantiles/Aggregated_Quantile_Power_Forecast_by_Rated_Capacity_All_Members.csv", mode = 'a', index = True, header = HEADER_VAR)
# Save day_only file
Quantile_Forecast1_CFP.to_csv(daily_path + "/Quantile_Power_Forcast_by_Rated_Capacity_100m_ for_" + tomorrow + ".csv", mode = 'w', index= True, header = True)


# Create a graphic of the Quantile Forecast and include the Harmonie Power Forecast.
node=Quantile_Forecast1_CFP.index
plt.plot(Quantile_Forecast1_CFP['Control_Power'], 'go-')
plt.plot(Quantile_Forecast1_CFP['90th%'], '--')
plt.plot(Quantile_Forecast1_CFP['75th%'], '--')
plt.plot(Quantile_Forecast1_CFP['50th%'], 'ro-', linewidth=2.5)
plt.plot(Quantile_Forecast1_CFP['25th%'], '--')
plt.plot(Quantile_Forecast1_CFP['10th%'], '--')
plt.gca().xaxis.set_major_formatter(myFmt)
plt.xticks(rotation='horizontal')
plt.xlabel('Time', color='r', fontsize=15)
plt.ylabel('Percentage Rated Capacity', color='r', fontsize=15)
plt.legend(['Deterministic Forecast','90th%', '75th%', '50th%', '25th%', '10th%'])
plt.title('Day-Ahead Quantile & Deterministic Power Forecast for ' + tomorrow, color='r', fontsize = 20)
plt.fill_between(node, Quantile_Forecast1_CFP['90th%'], Quantile_Forecast1_CFP['75th%'],facecolor='cornflowerblue', alpha=0.5)
plt.fill_between(node, Quantile_Forecast1_CFP['75th%'], Quantile_Forecast1_CFP['50th%'],facecolor='cornflowerblue', alpha=1)
plt.fill_between(node, Quantile_Forecast1_CFP['50th%'], Quantile_Forecast1_CFP['25th%'],facecolor='cornflowerblue', alpha=1)
plt.fill_between(node, Quantile_Forecast1_CFP['25th%'], Quantile_Forecast1_CFP['10th%'],facecolor='cornflowerblue', alpha=0.5)
plt.grid(linestyle='--', linewidth=0.5)
img_name = "/Day_Ahead_Quantile_Deterministic_Power_Forecast_by_Rated_Power_for_"
plt.savefig(img_path + img_name + tomorrow + ".png" )
plt.close()

# Do a Graphic without including the Deterministic Member
node=Quantile_Forecast1_CFP.index
plt.plot(Quantile_Forecast1_CFP['90th%'], '--')
plt.plot(Quantile_Forecast1_CFP['75th%'], '--')
plt.plot(Quantile_Forecast1_CFP['50th%'], 'ro-', linewidth=2.5)
plt.plot(Quantile_Forecast1_CFP['25th%'], '--')
plt.plot(Quantile_Forecast1_CFP['10th%'], '--')
plt.gca().xaxis.set_major_formatter(myFmt)
plt.xticks(rotation='horizontal')
plt.xlabel('Time', color='r', fontsize=15)
plt.ylabel('Percentage Rated Capacity', color='r', fontsize=15)
plt.legend(['90th%', '75th%', '50th%', '25th%', '10th%'])
plt.title('Day-Ahead Quantile Power Forecast for ' + tomorrow, color='r', fontsize = 20)
plt.fill_between(node, Quantile_Forecast1_CFP['90th%'], Quantile_Forecast1_CFP['75th%'],facecolor='cornflowerblue', alpha=0.5)
plt.fill_between(node, Quantile_Forecast1_CFP['75th%'], Quantile_Forecast1_CFP['50th%'],facecolor='cornflowerblue', alpha=1)
plt.fill_between(node, Quantile_Forecast1_CFP['50th%'], Quantile_Forecast1_CFP['25th%'],facecolor='cornflowerblue', alpha=1)
plt.fill_between(node, Quantile_Forecast1_CFP['25th%'], Quantile_Forecast1_CFP['10th%'],facecolor='cornflowerblue', alpha=0.5)
plt.grid(linestyle='--', linewidth=0.5)
img_name = "/Day_Ahead_Quantile_Power_Forecast_by_Rated_Power_for_"
plt.savefig(img_path + img_name + tomorrow + ".png" )
plt.close()
###############################################################################
# Email the results
###############################################################################

# Email details.
email_from = r'gurteenforecast1'
email_to = r'repowersupplyltd@gmail.com'
email_subject = r'Quantile Energy Forecast at 100m for Gurteen'
email_body = r'This is the latest Quantile Energy Forecast for Gurteen.'

# Create the email itself.
email_msg =  email.mime.multipart.MIMEMultipart()
email_msg['From'] = email_from
email_msg['To'] = email_to
email_msg['Subject'] = email_subject
email_msg.attach(email.mime.text.MIMEText(email_body, 'plain'))

# Convert the pandas dataframe to string, pretend it's a file.
Quantile_Forecast1_as_string = io.StringIO()
Quantile_Forecast1.to_csv(Quantile_Forecast1_as_string, index=True)

# Attach the file to the email.
email_attachment = email.mime.text.MIMEText(Quantile_Forecast1_as_string.getvalue())
email_attachment.add_header('Content-Disposition', 'attachment', filename=f'Gurteen_DAM_Quantile_Forecast_at_HH-{datetime.datetime.now():%Y-%m-%d-%H-%M}.csv')           
email_msg.attach(email_attachment)

# Send the email.
server = smtplib.SMTP('smtp.gmail.com:587')
server.ehlo()
server.starttls()
server.login(username, password)
server.sendmail(email_from, email_to, email_msg.as_string())
server.quit()

print(' IF YOU ARE READING THIS THE CODE HAS RUN SUCCESSFULLY - THE END...FOR NOW ANYWAY')
print('##### END #####')
