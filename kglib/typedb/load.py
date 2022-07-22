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
from enum import Enum


class FileType(Enum):
    Schema = "schema"
    Data = "data"


def load_typeql_file(typedb_binary_location: str, database_name: str, typeql_file_path: str,
                     file_type: FileType):
    """
    Load a file into a TypeDB database

    Args:
        typedb_binary_location: the location of TypeDB
        database_name: The name of the TypeDB database to load into
        typeql_file_path: The path to the file to load
        file_type: The content of the file and therefore the transaction type to use, either schema or data
    Returns:
        None
    """
    sp.check_call([
        './typedb',
        'console',
        f'--command=transaction {database_name} {file_type.value} write',
        f'--command=source {typeql_file_path}',
        f'--command=commit'
    ], cwd=typedb_binary_location)
