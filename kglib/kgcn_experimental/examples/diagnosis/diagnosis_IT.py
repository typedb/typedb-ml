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
import subprocess as sp
import unittest

from kglib.kgcn_experimental.examples.diagnosis.diagnosis import diagnosis_example
from kglib.utils.grakn.test.base import GraknServer

TEST_KEYSPACE = "diagnosis"
TEST_URI = "localhost:48555"


class TestDiagnosisExample(unittest.TestCase):
    def test_example_runs_without_exception(self):
        diagnosis_example(num_graphs=6,
                          num_processing_steps_tr=2,
                          num_processing_steps_ge=2,
                          num_training_iterations=20)


if __name__ == "__main__":

    with GraknServer() as gs:

        sp.check_call([
            'grakn', 'console', '-k', TEST_KEYSPACE, '-f',
            os.getenv("TEST_SRCDIR") + '/kglib/kglib/kgcn/test_data/schema.gql'
        ], cwd=gs.grakn_binary_location)

        unittest.main()
