'''
Python script to select users for a given period. This script can be used to select :
    - in month users, users that dont exists on previous months.
    - previous users, users that exists on february-march graph.
'''

import pandas as pd
import ast
import sys
import argparse




parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--period', type=str, help='period to select users')
parser.add_argument('--select_previous_users', action='store_true', help='select users that exists on feb_mar graph')

parser.add_argument('integers', metavar='N', type=int, nargs='+',
                    help='an integer for the accumulator')
parser.add_argument('--sum', dest='accumulate', action='store_const',
                    const=sum, default=max,
                    help='sum the integers (default: find the max)')

args = parser.parse_args()



period = args.period
select_previous_users = args.select_previous_users

''' Specified path of model input folder '''
path = '/Storage/gkont/model_input/'

''' Specified path of entity names json file of users of the given period '''
users_filename = '/Storage/gkont/embeddings/{}/entity_{}_json/entity_names_user_0.json'.format(period, period)


if select_previous_users:
    if period == 'feb_apr':
        months_to_exclude = ['feb_mar']
    elif period == 'feb_may':
        months_to_exclude = ['feb_mar']
    elif period == 'feb_jun':
        months_to_exclude = ['feb_mar']
    else:
        print('Error: Select a valid month period')
        sys.exit()
else:
    if period == 'feb_apr':
        months_to_exclude = ['feb_mar']
    elif period == 'feb_may':
        months_to_exclude = ['feb_mar', 'feb_apr']
    elif period == 'feb_jun':
        months_to_exclude = ['feb_mar', 'feb_apr', 'feb_may']
    else:
        print('Error: Select a valid month period')
        sys.exit()



with open(users_filename) as f:
    names_set = ast.literal_eval(f.read())


month_tsv = pd.read_csv(path + period + '/social_features_'+period+'.tsv',sep='\t',dtype={"user_id":"string"})


if select_previous_users == True:
    print('create tsv with only previous users')
    selected_users_output = path + period + '/previous_users_' + period + '.tsv'
else:
    print('create tsv with users that dont exist on previous months')
    selected_users_output = path + period + '/selected_users_' + period + '.tsv'




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
    '''select users that dont exist on previous month'''
    new_df = month_tsv[~month_tsv['user_id'].isin(names_set)]
else:
    '''select users that exist on previous month'''
    new_df = month_tsv[month_tsv['user_id'].isin(names_set)]
        

print('{} selected users df (len : {})'.format(period, new_df.shape))
new_df.to_csv(selected_users_output, sep='\t')
