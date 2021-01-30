# Glassdoor_Job_Salary_Estimate
# Glassdoor_Job_Salary_Estimate
## Project Overview
* Objective: The objective of this project is to build a classification model and a regression model that estimate the best fitting job type among data scientist, data analyst, and software engineering and the respective salary based on the set of skill set (input).
* Potential Use: The product of this project (job type classification model and salary regression model) can be used to assist any candidate seeking an entry level position in data analytics or software engineering field and provide the best fitting job type and estimated salary based on the candidate's skill sets.
* Model Performance: 93% accuracy achieved with classification model | salary estimate regression model (MAE ~ $18.06K)
* Comments on Model Performance: Low performance on salary estimate regression model may be due to the salary data from glassdoor estimates. Glassdoor estimates may be based on the job location (state), rather than the listed skill set.

## Used Resources
-**Python Version**: 3.8\
-**Python Packages**: pandas, numpy, sklearn, matplotlib, seaborn, selenium, flask, json, pickle\
-**Overall Project Reference**: https://www.youtube.com/channel/UCiT9RITQ9PW6BhXK0y2jaeg \
-**Web Scrapping**: https://github.com/arapfaik/scraping-glassdoor-selenium | https://towardsdatascience.com/selenium-tutorial-scraping-glassdoor-com-in-10-minutes-3d0915c6d905 \
-**Productionization**: https://towardsdatascience.com/productionize-a-machine-learning-model-with-flask-and-heroku-8201260503d2

## 1. Data Collection - Web Scrapping
Modified the web scrapping script (referenced above) to collect 1000+ job posts on entry level data scientist, data analyst, and software engineer from Glassdoor. From each job post, the collected data include:

Job title | Salary estimate | Job description | Rating | Company Name | Location | Company headquarter | Company Size | Company founded date | Type of ownership | Industry | Sector | Revenue | Competitors

## 2. Data Cleaning
Collected data were cleaned to be used as input data for models. The cleaning process is described as follows:
1. Numerical salary values extracted from the glassdoor estimate salary range
2. Rows without salary data removed
3. Extracted location data (state) by removing city names
4. Scaled annual salary data and hourly rates
5. Created columns for each significant job skills listed on job description (ex: python, AWS, Hadoop, Tableau, etc.)

## 3. Exploratory Data Analysis (EDA)
Cleaned dataset was then used to perform EDA. Few highlights are presented:

![alt text](https://github.gatech.edu/jbae42/Glassdoor_Job_Salary_Estimate/blob/master/Exploratory%20Data%20Analysis/images/Histogram%20on%20Entry%20Level%20Data%20Analyst%20Salary.png)
![alt text](https://github.gatech.edu/jbae42/Glassdoor_Job_Salary_Estimate/blob/master/Exploratory%20Data%20Analysis/images/Histogram%20on%20Entry%20Level%20Data%20Scientist%20Salary.png)
![alt text](https://github.gatech.edu/jbae42/Glassdoor_Job_Salary_Estimate/blob/master/Exploratory%20Data%20Analysis/images/Histogram%20on%20Entry%20Level%20Software%20Engineer%20Salary.png)

![alt text](https://github.gatech.edu/jbae42/Glassdoor_Job_Salary_Estimate/blob/master/Exploratory%20Data%20Analysis/images/Top%2020%20States%20with%20Highest%20Average%20Salary%20in%20Entry%20Data%20Analyst%20Positions.png)
![alt text](https://github.gatech.edu/jbae42/Glassdoor_Job_Salary_Estimate/blob/master/Exploratory%20Data%20Analysis/images/Top%2020%20States%20with%20Highest%20Average%20Salary%20in%20Entry%20Data%20Scientist%20Position.png)
![alt text](https://github.gatech.edu/jbae42/Glassdoor_Job_Salary_Estimate/blob/master/Exploratory%20Data%20Analysis/images/Top%2020%20States%20with%20Highest%20Average%20Salary%20in%20Entry%20Software%20Engineer%20Positions.png)

![alt text](https://github.gatech.edu/jbae42/Glassdoor_Job_Salary_Estimate/blob/master/Exploratory%20Data%20Analysis/images/wordcloud_dataAnalyst.png)
![alt text](https://github.gatech.edu/jbae42/Glassdoor_Job_Salary_Estimate/blob/master/Exploratory%20Data%20Analysis/images/wordcloud_datascientist.png)
![alt text](https://github.gatech.edu/jbae42/Glassdoor_Job_Salary_Estimate/blob/master/Exploratory%20Data%20Analysis/images/wordcloud_softwareEngineer.png)

## 4. Model Building
With the cleaned dataset, dummy variables were created. The dataset was then split into train and test sets with the ratio of 0.8:0.2, respectively.

**Classification Model** \
Two techniques were tried: Random Forest Classifier with Gridsearch CV and K Nearest Neighbors \
As a result, Random Forest Classifier gave better performance with 93%.

**Regression Model** \
Three techniques were tried: Linear Regression, Random Forest Regressor, XGBoost. \
As a result, Random Forest Regressor with hyperparameter tuning gave the best performance with MAE = 18.06.

## 5. Productionization
A flask API was built on a local webserver (followed steps are referenced above). The product takes in a request with a list of values that indicate whether a job skill is present or absent and returns best fitting job type among Data Scientist, Data Analyst, and Software Engineer with respective estimated salary. 
