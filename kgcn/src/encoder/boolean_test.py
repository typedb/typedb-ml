import unittest

import numpy as np
import tensorflow as tf

import kgcn.src.encoder.boolean as bool_enc

tf.enable_eager_execution()


class TestOneHotBooleanEncoding(unittest.TestCase):
    def test_input_output(self):
        input = np.full((2, 3, 1), -1, dtype=np.int64)

        input[1, 1] = 0
        input[1, 0] = 1
        input[0, 2] = 1

        print(f'Input:\n{input}')
        expected_output = np.full((2, 3, 2), 0, dtype=np.int64)
        expected_output[1, 1, 0] = 1
        expected_output[1, 0, 1] = 1
        expected_output[0, 2, 1] = 1

        print(f'\nExpected output\n{expected_output}')

        output = bool_enc.one_hot_boolean_encode(tf.convert_to_tensor(input, dtype=tf.int64))

        print(f'output:\n{output.numpy()}')
        with self.subTest('Correctness'):
            np.testing.assert_array_equal(output.numpy(), expected_output)


if __name__ == "__main__":
    unittest.main()
