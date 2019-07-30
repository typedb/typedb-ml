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

import numpy as np
from grakn.client import GraknClient

from datasets.synthetic.generate.pmf import PMF


def get_example_queries(pmf, example_id):

    variable_values = pmf.select()

    queries = [(f'''insert 
                $p isa person, 
                has example-id 
                {example_id};''')]

    if variable_values['Meningitis']:
        queries.append(f'''match
                       $d isa meningitis;
                       $p isa person, has example-id 
                       {example_id};
                       insert
                       (patient: $p, diagnosed-disease: $d) 
                       isa diagnosis;''')

    if variable_values['Flu']:
        queries.append(f'''match
                       $p isa person, has example-id 
                       {example_id};
                       $d isa flu;
                       insert
                       (patient: $p, diagnosed-disease: $d) 
                       isa diagnosis;''')

    if variable_values['Light Sensitivity']:
        queries.append(f'''match
                       $p isa person, has example-id 
                       {example_id};
                       $s isa light-sensitivity;
                       insert
                       (presented-symptom: $s, symptomatic-patient: $p) isa 
                       symptom-presentation;''')

    if variable_values['Fever']:
        queries.append(f'''match
                       $p isa person, has example-id 
                       {example_id};
                       $s isa fever;
                       insert
                       (presented-symptom: $s, symptomatic-patient: $p) isa 
                       symptom-presentation;''')

    return queries


def main():
    client = GraknClient(uri="localhost:48555")
    session = client.session(keyspace="diagnosis")

    pmf_array = np.zeros([2, 2, 2, 2], dtype=np.float)
    pmf_array[0, 1, 1, 1] = 1.0

    pmf = PMF({
        'Flu':                  [False, True],
        'Meningitis':           [False, True],
        'Light Sensitivity':    [False, True],
        'Fever':                [False, True]
    }, pmf_array)
    print(pmf.to_dataframe())

    for example_id in range(0, 2):
        tx = session.transaction().write()
        for query in get_example_queries(pmf, example_id):
            tx.query(query)
        tx.commit()

    session.close()
    client.close()


if __name__ == '__main__':
    main()
