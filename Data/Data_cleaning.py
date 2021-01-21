# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 19:10:32 2021

@author: Andrew
"""
from os import listdir
from os.path import isfile,join
import pandas as pd


# =============================================================================
path = 'C:/Users/Andrew/Desktop/EdX/Self Projects/Classification/Data/'
path2 = 'C:/Users/Andrew/Desktop/EdX/Self Projects/Classification/Data_entry/'
path3 = 'C:/Users/Andrew/Desktop/EdX/Self Projects/Classification/Data_entry/added/'

DS = pd.read_csv(path2+'DataScientist.csv')
DS2 = pd.read_csv(path2+'DataScientist_extra.csv')
SE = pd.read_csv(path2+'SoftwareEngineer.csv')
SE2 = pd.read_csv(path2+'SoftwareEngineer_extra.csv')

DS = pd.concat([DS, DS2]).reset_index(drop=True)

DS['Job_Code'] = 'DS'
#print(DS.columns)
DS = DS.drop(columns = ['Headquarters','Size','Founded','Type of ownership','Industry','Sector','Revenue','Competitors'])
DS.head()
DS['Job Title'] = DS['Job Title'].apply(lambda x: x if 'data' in x.lower() else -1)
DS = DS[DS['Job Title'] != -1]
DS.head()
DS = DS.drop_duplicates()

SE = pd.concat([SE,SE2]).reset_index(drop=True)

SE['Job_Code'] = 'SE'
SE = SE.drop(columns = ['Headquarters','Size','Founded','Type of ownership','Industry','Sector','Revenue','Competitors'])
SE.head()
SE['Job Title'] = SE['Job Title'].apply(lambda x: x if 'software engineer' in x.lower() else -1)
SE = SE[SE['Job Title'] != -1]
SE = SE.drop_duplicates()

DS.to_csv('DataScientist2.csv', index = False)
SE.to_csv('SoftwareEngineer2.csv', index = False)

# 
#ATL1 = pd.read_csv(path+'ATL1.csv')
# 
colnames = DS.columns
# 
filenames = [f for f in listdir(path3) if isfile(join(path3,f))]
# 
data = pd.DataFrame(columns = colnames)
# 

for file in filenames:
    dat1 = pd.read_csv(path3+'{}'.format(file))
    data = pd.concat([data, dat1]).reset_index(drop=True)
data.to_csv('full_data_entry_added.csv', index = False)
# =============================================================================


df = pd.read_csv('full_data_entry_added.csv')
df = df.drop_duplicates()
#salary parsing

df['hourly'] = df['Salary Estimate'].apply(lambda x: 1 if 'per hour' in x.lower() else 0)
df['employer_provided'] = df['Salary Estimate'].apply(lambda x: 1 if 'employer provided salary:' in x.lower() else 0)

df = df[df['Salary Estimate'] != "-1"]
salary = df['Salary Estimate'].apply(lambda x: x.split('(')[0])
minus_kd = salary.apply(lambda x: x.replace('K',"").replace('$',""))

min_hr = minus_kd.apply(lambda x: x.lower().replace('per hour',''))

df['min_salary'] = min_hr.apply(lambda x: int(x.split('-')[0]))
df['max_salary'] = min_hr.apply(lambda x: int(x.split('-')[1]))
df['avg_salary'] = (df.min_salary+df.max_salary) / 2

#company name text only
df['company_text'] = df.apply(lambda x: x['Company Name'] if x['Rating'] < 0 else x['Company Name'][:-3], axis = 1)

#state field
df['job_state'] = df['Location'].apply(lambda x: x if ',' not in x else x.split(',')[1])
#df.job_city.value_counts()

#df['same_state'] = df.apply(lambda x: 1 if x.loation == x.headquarters else 0, axis = 1)


#age of company
#df['age'] = df.Founded.apply(lambda x: x if x < 1 else 2021 - x)

#parsing of job description (python, etc.)
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
#df['develop'] = df['Job Description'].apply(lambda x: 1 if 'development' in x.lower() else 0)
#df['software'] = df['Job Description'].apply(lambda x: 1 if 'software' in x.lower() else 0)
df['debug'] = df['Job Description'].apply(lambda x: 1 if 'debug' in x.lower() else 0)
df['HTML'] = df['Job Description'].apply(lambda x: 1 if 'html' in x.lower() else 0)
df['object'] = df['Job Description'].apply(lambda x: 1 if 'object-oriented' in x.lower() else 0)
df['experience'] = df['Job Description'].apply(lambda x: 1 if 'year' in x.lower() else 0)

#words = df['Job Title'].split(" ")
words = str(df['Job Title'])
ind = str(df['Job Title']).index("Data")
#print(str(df['Job Title']))
print(words[ind-5:ind+4])

#print(list(df['Job Description']).index('on'))

df['Job_Code'].value_counts()

cols = df.columns
#print(new_colnames[0])
names = df.columns.to_list()
#for col in cols:
#    print(col, df[col].value_counts())

#df.columns
#df_out = df.drop(['Tableu'],axis = 1)

df.to_csv("entryLevel_cleaned.csv", index = False)
