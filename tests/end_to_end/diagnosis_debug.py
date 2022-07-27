#
#  Copyright (C) 2022 Vaticle
#
#  Licensed to the Apache Software Foundation (ASF) under one
#  or more contributor license agreements.  See the NOTICE file
#  distributed with this work for additional information
#  regarding copyright ownership.  The ASF licenses this file
#  to you under the Apache License, Version 2.0 (the
#  "License"); you may not use this file except in compliance
#  with the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing,
#  software distributed under the License is distributed on an
#  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
#  KIND, either express or implied.  See the License for the
#  specific language governing permissions and limitations
#  under the License.
#
import os
import sys
import unittest

from examples.diagnosis.diagnosis import diagnosis_example


class TestDiagnosisExampleDebug(unittest.TestCase):
    """
    A copy of the end-to-end test for local debugging. Requires a TypeDB server to be started in the background
    manually. Run with:
    bazel test //tests/end_to_end:diagnosis --test_output=streamed --spawn_strategy=standalone --action_env=PATH --test_arg=--<path/to/your/typedb/directory>
    """

    def setUp(self):
        self._typedb_binary_location = sys.argv.pop()
        base_dir = os.getenv("TEST_SRCDIR") + "/" + os.getenv("TEST_WORKSPACE")
        self._data_file_location = base_dir + sys.argv.pop()
        self._schema_file_location = base_dir + sys.argv.pop()

    def test_learning_is_done(self):
        train_accuracy, test_accuracy = diagnosis_example(
            self._typedb_binary_location,
            200,
            schema_file_path=self._schema_file_location,
            seed_data_file_path=self._data_file_location
        )
        self.assertGreaterEqual(train_accuracy, 0.75)
        self.assertLessEqual(train_accuracy, 0.99)
        self.assertGreaterEqual(test_accuracy, 0.75)
        self.assertLessEqual(test_accuracy, 0.99)


if __name__ == "__main__":
    # This handles the fact that additional arguments that are supplied by our py_test definition
    # https://stackoverflow.com/a/38012249
    unittest.main(argv=['ignored-arg'])
