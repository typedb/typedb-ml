import pickle


def save_variable(variable, file_path):
    pickle.dump(variable, open(file_path, "wb"))


def load_variable(file_path):
    return pickle.load(open(file_path, "rb"))
