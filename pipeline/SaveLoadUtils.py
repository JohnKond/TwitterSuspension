'''
-------------------------------
Author : Giannis Kontogiorgakis
Email : csd3964@csd.uoc.gr
-------------------------------
This file contains save and load functions for model and scaler.
'''

import pickle
import json
import os, os.path

path = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + '/'


def save_params(Mtype, params):
    """ Save model parameters in file """

    if not os.path.exists(path + 'output'):
        os.mkdir(path + 'output')

    if not os.path.exists(path + 'output/params'):
        os.mkdir(path + 'output/params')

    filename = path + 'output/params/' + Mtype + '_params.json'
    fp = open(filename, 'w+')
    json.dump(params, fp)
    fp.close()


def load_params(Mtype):
    """ Load model parameters from file """

    filename = path + 'output/params/' + Mtype + '_params.json'
    return json.load(open(filename, 'r'))


def save_scores(Mtype, score, train_score, time):
    """ Save model scores in file output/stats.txt """

    filename = path + 'output/' + 'stats.txt'
    fp = open(filename, 'a+')
    fp.write(Mtype + '\n' + '---------------' + '\n')
    fp.write('Training time : ' + str(time) + ' seconds\n')
    fp.write('Average validation score : ' + str(score) + '\n')
    fp.write('Average train score : ' + str(train_score) + '\n\n\n')
    
    
def save_model(model):
    """ Save model in pickle form """

    filename = 'model.pkl'
    pickle.dump(model,open(filename,'wb'))


def load_model():
    """ Load model """

    loaded_model = pickle.load(open('model.pkl','rb'))
    return loaded_model


def save_scaler(scaler):
    """ Save scaler in pickle form """

    filename = 'scaler.pkl'
    pickle.dump(scaler,open(filename,'wb'))


def load_scaler():
    """ Load scaler """
    loaded_scaler = pickle.load(open('scaler.pkl','rb'))
    return loaded_scaler



