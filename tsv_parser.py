import json

import h5py
import pandas as pd

# change for each graph
period = 'feb_mar'

suspended_users = set()
compl_users = set()

compliance_file = '/home/gkont/compliance.txt'
entity_file = '/Storage/gkont/embeddings/{}/entity_{}_json/entity_names_user_0.json'.format(period, period)
embeddings_file = '/Storage/gkont/embeddings/{}/checkpoint_{}_json/embeddings_user_0.v500.h5'.format(period, period)
new_entity_file = '/Storage/gkont/embeddings/{}/entity_{}_json/new_users.txt'.format(period,period)
labels_file = '/Storage/gkont/embeddings/{}/entity_{}_json/labels.txt'.format(period,period)
output_file = '/Storage/gkont/embeddings/{}/social_features_{}.tsv'.format(period, period)


def read_compliance():
    comp_path = compliance_file
    compl_file = open(comp_path, 'r')
    lines = compl_file.readlines()

    for line in lines:
        if line == "\n":
            continue
        entry = json.loads(line)
        compl_users.add(entry['id'])
        if entry['reason'] == 'suspended':
            suspended_users.add(entry['id'])
    compl_file.close()
    print('finished reading compliance')
        

def selected_users_dict():
    uid = []
    target = []
    with open(entity_file, "rt") as tf:
        names = json.load(tf)
        for name in names:
            label = ""
            if name not in compl_users:
                label = "0"
            elif name in suspended_users:
                label = "1"
            else:
                continue

            uid.append(name)
            target.append(label)
    print('Got selected users')
    return uid, target
   # write_lists_in_files(uid,target)


def write_lists_in_files(uid,target):
    with open(new_entity_file,'w') as fp:
        for user in uid:
            fp.write("%s\n" % user)

    with open(labels_file,'w') as lp:
        for label in target:
            lp.write("%s\n" % label)

        print("Copied uids and labels in ",new_entity_file)


def create_tsv(uid, target):

    '''
    uid = []
    target = []
    with open(new_entity_file, 'r') as fp:
        for line in fp:
            x = line[:-1]
            uid.append(x)

    with open(labels_file,'r') as lp:
        for line in lp:
            x = line[:-1]
            target.append(x)

    '''
    with open(entity_file, "rt") as tf:
        names = json.load(tf)
    """Cast to dictionary making much faster search process"""
    names = {names[i]: i for i in range(len(names))}

    """Create list of vectors based on user ids from profile features, containing graph embeddings"""
    print('Started reading embeddings')
    f_out = open(output_file,'w+')
    columns = ["graph_dim_{}".format(i) for i in range(1, 151)]
    columns.append("user_id")
    columns.append("target")

    f_out.write('\t'.join(columns) + '\n')
    with h5py.File(embeddings_file, "r") as hf:
        for i in range(len(uid)):
            try:
                position = names[str(uid[i])]
                f_out.write('\t'.join([f'{item}' for item in list(hf["embeddings"][position, :])] + [uid[i], target[i]]) + '\n')
                
            except Exception as e:
                print(e)
                break
                """In case when user from profile features is not found in graph we just skip this user"""
                continue
    f_out.close()
    print("Store of embedding {} is done.".format(output_file))


read_compliance()
uid, target = selected_users_dict()
create_tsv(uid, target)


