import pickle


def save_model(type, model):
    filename = 'models/'+type + '_bot_detection.pkl'
    pickle.dump(model, open(filename, 'wb'))


def load_model(type):
    loaded_model = pickle.load(open(type + '_bot_detection.pkl'), 'rb')
    return loaded_model
