
class ConceptInfo:
    def __init__(self, id, type_label, metatype_label, data_type=None, value=None):
        self.id = id
        self.type_label = type_label
        self.metatype_label = metatype_label

        # If the concept is an attribute
        self.data_type = data_type
        self.value = value


def build_concept_info(concept):

    id = concept.id
    type_label = concept.type().label()
    metatype_label = concept.base_type.lower()

    if metatype_label == 'ATTRIBUTE':
        data_type = concept.data_type()
        value = concept.value()

        return ConceptInfo(id, type_label, metatype_label, data_type, value)

    return ConceptInfo(id, type_label, metatype_label)
