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

import subprocess as sp


def load_typeql_schema_file(database, typedb_binary_location, typeql_file_path):
    """Load a schema from a file"""
    _load_typeql_file(database, typeql_file_path, typedb_binary_location, "schema")  # TODO compress


def load_typeql_data_file(database, typedb_binary_location, typeql_file_path):
    """Load data from a file"""
    _load_typeql_file(database, typeql_file_path, typedb_binary_location, "data")


def _load_typeql_file(database, typeql_file_path, typedb_binary_location, schema_or_data):
    sp.check_call([
        './typedb',
        'console',
        f'--command=transaction {database} {schema_or_data} write',
        f'--command=source {typeql_file_path}',
        f'--command=commit'
    ], cwd=typedb_binary_location)
