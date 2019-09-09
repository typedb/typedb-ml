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

from grakn.client import GraknClient

from kglib.kgcn_experimental.examples.diagnosis.data import create_concept_graphs, CATEGORICAL_ATTRIBUTES, \
    write_predictions_to_grakn
from kglib.kgcn_experimental.examples.diagnosis.template import template
from kglib.utils.grakn.synthetic.examples.diagnosis.generate import generate_example_graphs


def diagnosis_example(num_graphs=60, keyspace="diagnosis", uri="localhost:48555"):

    tr_ge_split = int(num_graphs*0.5)

    generate_example_graphs(num_graphs, keyspace=keyspace, uri=uri)

    client = GraknClient(uri=uri)
    session = client.session(keyspace=keyspace)

    graphs = create_concept_graphs(list(range(num_graphs)), session)

    ge_graphs = template(graphs,
                         tr_ge_split,
                         session,
                         num_processing_steps_tr=10,
                         num_processing_steps_ge=10,
                         num_training_iterations=1000,
                         categorical_attributes=CATEGORICAL_ATTRIBUTES,
                         )

    with session.transaction().write() as tx:
        write_predictions_to_grakn(ge_graphs, tx)

    session.close()
    client.close()


if __name__ == "__main__":
    diagnosis_example()
