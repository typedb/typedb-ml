
def preprocess_all(raw_arrays, preprocessors):
    """
    :param raw_arrays: numpy arrays to be transformed into a format such that can be fed into a TensorFlow op graph
    :return: tensor-ready arrays
    """

    preprocessed_raw_arrays = []
    for raw_array in raw_arrays:
        preprocessed_features = {}
        for key, features_array in raw_array.items():
            try:
                preprocessed_features[key] = preprocessors[key](features_array)
            except KeyError:
                print(f'Skipping key {key}')
        preprocessed_raw_arrays.append(preprocessed_features)

    return preprocessed_raw_arrays
