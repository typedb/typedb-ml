

class OneHotTypeLabelEncoder:
    def __init__(self, grakn_type_labels):
        """
        :param grakn_types: All Grakn type labels in the schema for entities, relationships (incl. implicit) and
        attributes, not including roles
        """
        self._grakn_type_labels = grakn_type_labels

    def __call__(self, type_labels_tsr):
        # def encode_type(self, type_labels_tsr):
        """
        Takes
        :param type_labels: A tensor of Grakn type labels
        :return: A tensor of Grakn type labels encoded into an array
        """

        # TODO One-hot encoding of either type labels or of a tensor of type ids (some class renaming required in the
        #  latter case
        # https://www.tensorflow.org/api_docs/python/tf/one_hot
        encoded_type_labels_tsr = None
        return encoded_type_labels_tsr


class ConnectionEncoder:

    def __call__(self, *args, **kwargs):
        pass


class AttributeValueEncoder:

    def __call__(self, *args, **kwargs):
        pass