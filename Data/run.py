# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 21:48:51 2021

@author: Andrew
"""
from selenium import webdriver
import glassdoor_scraper as gs
import pandas as pd

#path = webdriver.Chrome(executable_path=r"C:/Users/Andrew/Desktop/EdX/Self Projects/Classification/chromedriver.exe")
path = r"C:/Users/Andrew/Desktop/EdX/Self Projects/Classification/chromedriver.exe"
file_path = r"C:/Users/Andrew/Desktop/EdX/Self Projects/Classification/Data_entry/"
df = gs.get_jobs('data scientist',970,False,path,5)


#<span class="css-1uyte9r css-hca4ks e1wijj242">$100K - $160K <span class="css-0 e1wijj240">(Glassdoor est.)</span><span class="SVGInline greyInfoIcon"><svg class="SVGInline-svg greyInfoIcon-svg" height="14" viewBox="0 0 14 14" width="14" xmlns="http://www.w3.org/2000/svg"><path d="M7 14A7 7 0 117 0a7 7 0 010 14zm0-.7A6.3 6.3 0 107 .7a6.3 6.3 0 000 12.6zm-.7-7a.7.7 0 011.4 0v4.2a.7.7 0 01-1.4 0zM7 4.2a.7.7 0 110-1.4.7.7 0 010 1.4z" fill="#505863" fill-rule="evenodd"></path></svg></span><div class="hidden"></div></span>

#css-1uyte9r css-hca4ks e1wijj242


#<a data-test="pagination-next" href="/Job/data-scientist-jobs-SRCH_KO0,14.htm?p=2"><span class=""><i></i></span></a>

           # driver.find_element_by_xpath('.//li[@class="next"]//a').click()


df.to_csv('Data Scientist_20210122.csv',index=False)

#df_Michigan4 = df

# =============================================================================
# df_ATL1.to_csv('ATL1.csv',index=False)
# df_ATL2.to_csv('ATL2.csv',index=False)
# df_ATL3.to_csv('ATL3.csv',index=False)
# 
# df_Austin1.to_csv('Austin1.csv', index=False)
# df_Austin2.to_csv('Austin2.csv', index=False)
# df_Austin3.to_csv('Austin3.csv', index=False)
# 
# df_Boston1.to_csv('Boston1.csv', index=False)
# df_Boston2.to_csv('Boston2.csv', index=False)
# df_Boston3.to_csv('Boston3.csv', index=False)
# df_Boston4.to_csv('Boston4.csv', index=False)
# 
# df_Dallas1.to_csv('Dallas1.csv', index=False)
# df_Dallas2.to_csv('Dallas2.csv', index=False)
# df_Dallas3.to_csv('Dallas3.csv', index=False)
# 
# df_Houston1.to_csv('Houston1.csv', index=False)
# df_Houston2.to_csv('Houston2.csv', index=False)
# df_Houston3.to_csv('Houston3.csv', index=False)
# 
# df_MD1.to_csv('MD1.csv', index=False)
# df_MD2.to_csv('MD2.csv', index=False)
# df_MD3.to_csv('MD3.csv', index=False)
# 
# df_Michigan1.to_csv('Michigan1.csv', index=False)
# df_Michigan2.to_csv('Michigan2.csv', index=False)
# df_Michigan3.to_csv('Michigan3.csv', index=False)
# df_Michigan4.to_csv('Michigan4.csv', index=False)
# 
# df_NJ1.to_csv('NJ1.csv', index=False)
# df_NJ2.to_csv('NJ2.csv', index=False)
# df_NJ3.to_csv('NJ3.csv', index=False)
# 
# df_PHL1.to_csv('PHL1.csv', index=False)
# df_PHL2.to_csv('PHL2.csv', index=False)
# df_PHL3.to_csv('PHL3.csv', index=False)
# df_PHL4.to_csv('PHL4.csv', index=False)
# 
# df_VA1.to_csv('VA1.csv', index=False)
# df_VA2.to_csv('VA2.csv', index=False)
# df_VA3.to_csv('VA3.csv', index=False)
# df_VA4.to_csv('VA4.csv', index=False)
# =============================================================================




