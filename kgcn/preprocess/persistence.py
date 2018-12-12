import os
import pickle

def save_variable(variable, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    pickle.dump(variable, open(file_path, "wb"))


def load_variable(file_path):
    return pickle.load(open(file_path, "rb"))


class FeedDictStorer:
    def __init__(self, storage_path, file_suffix='.p'):
        self._storage_path = storage_path
        self._file_suffix = file_suffix

    def _pack_feed_dict(self, feed_dict):
        print('Packing...')
        feed_dict_placeholder_names_as_keys = {}

        for placeholder, value in feed_dict.items():
            print('    ' + placeholder.name)
            feed_dict_placeholder_names_as_keys[placeholder.name] = value

        return feed_dict_placeholder_names_as_keys

    def _unpack_feed_dict(self, packed_feed_dict):

        print('Unpacking...')
        unpacked_feed_dict = {}
        for placeholder_name, value in packed_feed_dict.items():
            print('    ' + placeholder_name)
            unpacked_feed_dict[self._graph.get_tensor_by_name(placeholder_name)] = value

        return unpacked_feed_dict

    def _get_path(self, name):
        return self._storage_path + name + self._file_suffix

    def store_feed_dict(self, name, feed_dict):
        packed_feed_dict = self._pack_feed_dict(feed_dict)
        file_path = self._get_path(name)
        print(f'Storing in {file_path}')
        save_variable(packed_feed_dict, file_path)

    def retrieve_feed_dict(self, name):
        file_path = self._get_path(name)
        print(f'Loading from {file_path}')
        packed_feed_dict = load_variable(file_path)
        feed_dict = self._unpack_feed_dict(packed_feed_dict)
        return feed_dict
