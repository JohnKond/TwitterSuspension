import pickle
import json
import os, os.path

path = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + '/'


def save_params(Mtype, params):

    if not os.path.exists(path + 'output'):
        os.mkdir(path + 'output')

    if not os.path.exists(path + 'output/params'):
        os.mkdir(path + 'output/params')

    filename = path + 'output/params/' + Mtype + '_params.json'
    fp = open(filename, 'w+')
    json.dump(params, fp)
    fp.close()


def load_params(Mtype):
    filename = path + 'output/params/' + Mtype + '_params.json'
    return json.load(open(filename, 'r'))


def save_scores(Mtype, score, train_score, time):
    filename = path + 'output/' + 'stats.txt'
    fp = open(filename, 'a+')
    fp.write(Mtype + '\n' + '---------------' + '\n')
    fp.write('Training time : ' + str(time) + ' seconds\n')
    fp.write('Average validation score : ' + str(score) + '\n')
    fp.write('Average train score : ' + str(train_score) + '\n\n\n')
    
    
def save_model(model):
    filename = 'model.pkl'
    pickle.dump(model,open(filename,'wb'))


def load_model():
    loaded_model = pickle.load(open('model.pkl','rb'))
    return loaded_model


def save_scaler(scaler):
    filename = 'scaler.pkl'
    pickle.dump(scaler,open(filename,'wb'))


def load_scaler():
    loaded_scaler = pickle.load(open('scaler.pkl','rb'))
    return loaded_scaler



