# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 21:15:32 2021

@author: Andrew
"""
import pandas as pd
import numpy as np

# =============================================================================
# example = {'python': '1',
#            'masters': '1',
#            'statistic': '1',
#            'SQL': '1',
#            'spark':'0',
#            'AWS':'0',
#            'Tableau':'0',
#            'Hadoop':'0',
#            'C_lang':'0',
#            'Java':'0',
#            'app':'0',
#            'debug':'0',
#            'HTML':'0',
#            'object':'1'}
# =============================================================================



example = {'python': 1,
           'masters': 1,
           'statistic': 1,
           'SQL': 1,
           'spark':0,
           'AWS':0,
           'Tableau':0,
           'Hadoop':0,
           'C_lang':0,
           'Java':0,
           'app':0,
           'debug':0,
           'HTML':0,
           'object':1}

data_in = example.values()

data_in = list(data_in)




#data_in = np.ravel(pd.DataFrame(data_in, index=['python', 'masters', 'statistic', 'SQL', 'spark', 'AWS', 'Tableau','Hadoop', 'C_lang', 'Java', 'app', 'debug', 'HTML', 'object']))

# =============================================================================
# data_in = np.ravel(pd.DataFrame.from_dict(example, orient='index'))
# data_in = list(data_in)
# data_in = [int(ele) for ele in data_in]
# =============================================================================

print(data_in)
