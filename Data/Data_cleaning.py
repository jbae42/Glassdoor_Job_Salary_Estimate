# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 22:01:00 2021

@author: Andrew
"""

from os import listdir
from os.path import isfile,join
import pandas as pd


### Step 1. Reading the datasets and combine datasets by job type

path2 = 'C:/Users/Andrew/Desktop/EdX/Self Projects/Classification/Data_entry/'
path3 = 'C:/Users/Andrew/Desktop/EdX/Self Projects/Classification/Data_entry/test/'

DS = pd.read_csv(path2+'DataScientist.csv')
DS2 = pd.read_csv(path2+'DataScientist_extra.csv')
DS3 = pd.read_csv(path2+'Data Scientist_20210122.csv')
SE = pd.read_csv(path2+'SoftwareEngineer.csv')
SE2 = pd.read_csv(path2+'SoftwareEngineer_extra.csv')

DS = pd.concat([DS, DS2, DS3]).reset_index(drop=True)
DS.drop_duplicates()

SE = pd.concat([SE, SE2]).reset_index(drop=True)
SE = SE.drop_duplicates()


### Step 2. Selecting only Data Scientist, Data Analyst, andn Software Engineer Job Posts

DS['Job Title'] = DS['Job Title'].apply(lambda x: 'Data Scientist' if 'data scientist' in x.lower() else ("Data Analyst" if "data analyst" in x.lower() else 0))
SE['Job Title'] = SE['Job Title'].apply(lambda x: x if 'software engineer' in x.lower() else -1)
SE = SE[SE['Job Title'] != -1]

AN = DS[DS['Job Title'] == 'Data Analyst']
DS = DS[DS['Job Title'] == 'Data Scientist']

### Step 3. Create Job_Code column by job type
AN['Job_Code'] = 'AN'
DS['Job_Code'] = 'DS'
SE['Job_Code'] = 'SE'

### Step 4. Drop unnecessary columns in each dataset
DS = DS.drop(columns = ['Headquarters','Size','Founded','Type of ownership','Industry','Sector','Revenue','Competitors'])
AN = AN.drop(columns = ['Headquarters','Size','Founded','Type of ownership','Industry','Sector','Revenue','Competitors'])
SE = SE.drop(columns = ['Headquarters','Size','Founded','Type of ownership','Industry','Sector','Revenue','Competitors'])

### Step 5. Export the datasets for the record
AN.to_csv('DataAnalyst_test.csv', index = False)
DS.to_csv('DataScientist_test.csv', index = False)
SE.to_csv('SoftwareEngineer_test.csv', index = False)


### Step 6. Combine all datasets and export the master dataset
# =============================================================================
colnames = DS.columns

filenames = [f for f in listdir(path3) if isfile(join(path3,f))]
# # 
data = pd.DataFrame(columns = colnames)
# # 
# 
for file in filenames:
     dat1 = pd.read_csv(path3+'{}'.format(file))
     data = pd.concat([data, dat1]).reset_index(drop=True)
data.to_csv('full_data_entry_added.csv', index = False)
# =============================================================================


df = pd.read_csv('full_data_entry_added.csv')
df = df.drop_duplicates()


### Step 7. Parse and extract the salary data
df['hourly'] = df['Salary Estimate'].apply(lambda x: 1 if 'per hour' in x.lower() else 0)
df['employer_provided'] = df['Salary Estimate'].apply(lambda x: 1 if 'employer provided salary:' in x.lower() else 0)

df = df[df['Salary Estimate'] != "-1"]
salary = df['Salary Estimate'].apply(lambda x: x.split('(')[0])
minus_kd = salary.apply(lambda x: x.replace('K',"").replace('$',""))

min_hr = minus_kd.apply(lambda x: x.lower().replace('per hour',''))

df['min_salary'] = min_hr.apply(lambda x: int(x.split('-')[0]))
df['max_salary'] = min_hr.apply(lambda x: int(x.split('-')[1]))
df['avg_salary'] = (df.min_salary+df.max_salary) / 2

### Step 8. Extract the company name
df['company_text'] = df.apply(lambda x: x['Company Name'] if x['Rating'] < 0 else x['Company Name'][:-3], axis = 1)

### Step 9. Extract the job location (state) 
df['job_state'] = df['Location'].apply(lambda x: x if ',' not in x else x.split(',')[1])


### Step 10. Parse the job description: Factoring out the keywords
df['python'] = df['Job Description'].apply(lambda x: 1 if 'python' in x.lower() else 0)
df['masters'] = df['Job Description'].apply(lambda x: 1 if 'master' in x.lower() else 0)
df['statistic'] = df['Job Description'].apply(lambda x: 1 if 'statistic' in x.lower() else 0)
df['SQL'] = df['Job Description'].apply(lambda x: 1 if 'sql' in x.lower() else 0)
df['spark'] = df['Job Description'].apply(lambda x: 1 if 'spark' in x.lower() else 0)
df['AWS'] = df['Job Description'].apply(lambda x: 1 if 'aws' in x.lower() else 0)
df['Tableau'] = df['Job Description'].apply(lambda x: 1 if 'tableau' in x.lower() else 0)
df['Hadoop'] = df['Job Description'].apply(lambda x: 1 if 'hadoop' in x.lower() else 0)
df['experience'] = df['Job Description'].apply(lambda x: 1 if 'year' in x.lower() else 0)
df['C_lang'] = df['Job Description'].apply(lambda x: 1 if 'C#' in x or 'C++' in x else 0)
df['Java'] = df['Job Description'].apply(lambda x: 1 if 'java' in x.lower() else 0)
df['app'] = df['Job Description'].apply(lambda x: 1 if 'application' in x.lower() else 0)
df['debug'] = df['Job Description'].apply(lambda x: 1 if 'debug' in x.lower() else 0)
df['HTML'] = df['Job Description'].apply(lambda x: 1 if 'html' in x.lower() else 0)
df['object'] = df['Job Description'].apply(lambda x: 1 if 'object-oriented' in x.lower() else 0)
df['experience'] = df['Job Description'].apply(lambda x: 1 if 'year' in x.lower() else 0)


### Step 11. Export the cleaned dataset
df.to_csv("entryLevel_cleaned.csv", index = False)