import numpy as np


def encode_neighbour_role(neighbour_role, role_encoder, thing_encoder):
    return np.concatenate(
        (role_encoder(neighbour_role.role, neighbour_role.role_direction),
         thing_encoder(neighbour_role.neighbour_with_neighbourhood.concept),
         ),
        concat_axis)


class InstanceEncoder:
    """
    Encodes instances of schema types
    """

    def __init__(self, role_encoder, thing_encoder):
        self._neighbourhood_sizes = [3, 4]
        self._batch_size = 20
        self._role_encoder = role_encoder
        self._thing_encoder = thing_encoder

    def reset(self):
        """
        To be called when a new batch of samples is processed in order to remove any memory of the previous batch (if
        this is necessary)
        :return:
        """
        self._role_encoder.reset()
        self._thing_encoder.reset()

    def encode_concepts_with_neighbourhoods(self, concepts_with_neighbourhoods):

        # full_feats_shape = [self._batch_size] + self._neighbourhood_sizes + [
        #     self._role_encoder.output_size + self._thing_encoder.output_size]

        # n_depths = len(self._neighbourhood_sizes) + 1  # plus 1 to include the initial examples of interest as a depth
        full_depth_shape = [self._batch_size] + self._neighbourhood_sizes
        n_depths = len(full_depth_shape)  # plus 1 to include the initial examples of interest as a depth

        # for depth in range(1, n_depths + 1):
        #     depth is 1-indexed

        for depth in range(n_depths):
            shape_at_this_depth = full_depth_shape[:depth + 1]
            feats_at_this_depth = np.empty(shape_at_this_depth)

            for i in range(shape_at_this_depth[depth]):


            for j, concept_with_neighbourhood in enumerate(concepts_with_neighbourhoods):
                feats_at_this_depth[j]
                for neighbour_role in concept_with_neighbourhood.neighbourhood:
                    feats_at_this_depth[,:] = encode_neighbour_role(neighbour_role, self._role_encoder, self._thing_encoder)
