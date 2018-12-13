import typing as typ
import kgcn.neighbourhood.data.sampling.random_sampling as random


class ConceptLabelExtractor:

    def __init__(self, query: str, attribute_vars_config: typ.Tuple[str, typ.MutableMapping[str, typ.List]],
                 sampling_method=random.random_sample):
        self._sampling_method = sampling_method
        self._attribute_vars_config = attribute_vars_config
        self._query = query

    def __call__(self, tx, sample_size):

        response = tx.query(self._query)
        sampled_responses = self._sampling_method(response, sample_size)
        owner_var = self._attribute_vars_config[0]

        concepts_with_labels = []
        for answer in sampled_responses:
            owner_concept = answer.get(owner_var)
            attribute_values = {}
            for attribute_var, category_options in self._attribute_vars_config[1].items():
                attribute_value = answer.get(attribute_var).value()

                if len(category_options) == 0:
                    attribute_value = [attribute_value]
                else:
                    # Then the output is categorical
                    attribute_one_hot = [0] * len(category_options)
                    attribute_one_hot[category_options.index(attribute_value)] = 1
                    attribute_value = attribute_one_hot
                attribute_values[attribute_var] = attribute_value
            concepts_with_labels.append((owner_concept, attribute_values))
        return concepts_with_labels
