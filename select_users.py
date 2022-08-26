import pandas as pd
import ast
import sys

period = 'feb_may'
path = '/Storage/gkont/model_input/'


select_previous_users = True

if period == 'feb_apr':
    months_to_exclude = ['feb_mar']
elif period == 'feb_may':
    months_to_exclude = ['feb_mar','feb_apr']
elif period == 'feb_jun':
    months_to_exclude = ['feb_mar','feb_apr','feb_may']
else:
    print('Error: Select a valid month period')
    sys.exit()

users_filename = '/Storage/gkont/embeddings/{}/entity_{}_json/entity_names_user_0.json'.format(period,period)
'''

with open(users_filename) as f:
    names_set = ast.literal_eval(f.read())

print('names feb_mar : ',len(names_set))
'''
month_tsv = pd.read_csv(path + period + '/social_features_'+period+'.tsv',sep='\t',dtype={"user_id":"string"})

if select_previous_users == True:
    print('create tsv with only previous users')
    selected_users_output = path + period + '/previous_users_' + period + '.tsv'
else:
    print('create tsv with users that dont exist on previous months')
    selected_users_output = path + period + '/selected_users_' + period + '.tsv'

print('read {}.tsv (len : {})'.format(period,len(month_tsv)))

new_df = pd.DataFrame()
for month in months_to_exclude:
    users_filename = '/Storage/gkont/embeddings/{}/entity_{}_json/entity_names_user_0.json'.format(month,month)
    with open(users_filename) as f:
        names_set = set(ast.literal_eval(f.read()))
     
    #print('all feb_mar users are : ',len(names_set))
    #print('excluding all entries from month ',month)
    
    # select users that dont exist on previous month
    #new_df = pd.concat([new_df , month_tsv[~month_tsv['user_id'].isin(names_set)]])
    
    # select users that exist on previous month
    #new_df = month_tsv[month_tsv['user_id'].isin(names_set)]
    new_df = pd.concat([new_df , month_tsv[month_tsv['user_id'].isin(names_set)]])


print('new df (len : {})'.format(len(new_df)))
new_df.to_csv(selected_users_output,sep='\t')


