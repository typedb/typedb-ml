#
#  Copyright (C) 2021 Vaticle
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

def get_call_args(mock):
    """
    Get the arguments used to call a mock for each call to the mock. Necessary since `.assert_has_calls` won't work
    for numpy arrays
    Args:
        mock: the mock

    Returns:
        A list of lists. The outer list is the calls made, the inner list is the arguments given for that call
    """
    flat_args = []
    args_list = mock.call_args_list
    for call in args_list:
        args, kwargs = call
        flat_args.append(args)
    return flat_args
