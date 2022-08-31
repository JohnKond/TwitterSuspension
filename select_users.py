import pandas as pd
import ast
import sys

period = 'feb_jun'
path = '/Storage/gkont/model_input/'


select_previous_users = True

if period == 'feb_apr':
    months_to_exclude = ['feb_mar']
elif period == 'feb_may':
    months_to_exclude = ['feb_mar']
elif period == 'feb_jun':
    months_to_exclude = ['feb_mar']
else:
    print('Error: Select a valid month period')
    sys.exit()

'''
users_filename = '/Storage/gkont/embeddings/{}/entity_{}_json/entity_names_user_0.json'.format(period,period)'

with open(users_filename) as f:
    names_set = ast.literal_eval(f.read())

print('names feb_mar : ',len(names_set))
'''
month_tsv = pd.read_csv(path + period + '/social_features_'+period+'.tsv',sep='\t',dtype={"user_id":"string"})

#print(month_tsv.value_counts())


if select_previous_users == True:
    print('create tsv with only previous users')
    selected_users_output = path + period + '/previous_users_' + period + '.tsv'
else:
    print('create tsv with users that dont exist on previous months')
    selected_users_output = path + period + '/selected_users_' + period + '.tsv'



# count users
# y = pd.read_csv(path+period+'/social_features_'+period+'.tsv',sep='\t',dtype={"user_id":"string"},usecols=['target'])
# y = selected_only_users['target']
# print(y.value_counts())


print('read {}.tsv (len : {})'.format(period,month_tsv.shape))

names_set = set()
new_df = pd.DataFrame()
for month in months_to_exclude:
    users_filename = '/Storage/gkont/embeddings/{}/entity_{}_json/entity_names_user_0.json'.format(month,month)
    with open(users_filename) as f:
        entities = set(ast.literal_eval(f.read()))
        for name in entities:
            names_set.add(name)
    print('excluding all entries from month ',month)
    
if select_previous_users == False:
    # select users that dont exist on previous month
    new_df = month_tsv[~month_tsv['user_id'].isin(names_set)]
else:
    # select users that exist on previous month
    new_df = month_tsv[month_tsv['user_id'].isin(names_set)]
        

print('{} selected users df (len : {})'.format(period, new_df.shape))
new_df.to_csv(selected_users_output,sep='\t')


