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


class MockConcept:
    def __init__(self, id, base_type):
        self.id = id
        assert base_type in {'ENTITY', 'RELATION', 'ATTRIBUTE'}
        self.base_type = base_type


class MockType(MockConcept):
    def __init__(self, id, base_type, label):
        super().__init__(id, base_type)
        self._label = label

    def label(self):
        return self._label


class MockThing(MockConcept):
    def __init__(self, id, base_type, type):
        super().__init__(id, base_type)
        assert(isinstance(type, MockType))
        assert(type.base_type == self.base_type)
        self._type = type

    def type(self):
        return self._type


class MockAttribute(MockThing):
    def __init__(self, id, base_type, type, data_type, value):
        super().__init__(id, base_type, type)
        self._value = value
        assert data_type in {'STRING', 'LONG', 'DOUBLE', 'DATE', 'BOOLEAN'}
        self._data_type = data_type

    def data_type(self):
        return self._data_type

    def value(self):
        return self.value()