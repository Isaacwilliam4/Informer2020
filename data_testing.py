import pandas as pd 
import numpy as np 

lg_df = pd.read_csv('./data/tomato_lg_prepared.csv')
tom_df = pd.read_csv('./data/tomato.csv')
tom_df += 1
log_df = np.log(tom_df)

print('lg shape', lg_df.values.shape, 'lg max', lg_df.max().max(), 'lg min', lg_df.min().min())
print('tom shape', tom_df.values.shape,'tom max', tom_df.max().max(), 'tom min', tom_df.min().min())
print('log shape', log_df.values.shape,'log max', log_df.max().max(), 'log min', log_df.min().min())


