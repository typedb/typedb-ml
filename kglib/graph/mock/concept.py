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
    def __init__(self, id):
        self.id = id


class MockType(MockConcept):
    def __init__(self, id, label, base_type):
        super().__init__(id)
        self._label = label
        assert base_type in {'ENTITY', 'RELATION', 'ATTRIBUTE'}
        self.base_type = base_type

    def label(self):
        return self._label


class DataType:
    def __init__(self, name):
        self.name = name


class MockAttributeType(MockType):
    def __init__(self, id, label, base_type, data_type):
        super().__init__(id, label, base_type)
        assert data_type in {'STRING', 'LONG', 'DOUBLE', 'DATE', 'BOOLEAN'}
        self._data_type = DataType(data_type)

    def data_type(self):
        return self._data_type


class MockThing(MockConcept):
    def __init__(self, id, type):
        super().__init__(id)
        assert(isinstance(type, MockType))
        self._type = type

    def type(self):
        return self._type

    @property
    def base_type(self):
        return self._type.base_type


class MockAttribute(MockThing):
    def __init__(self, id, value, type):
        super().__init__(id, type)
        self._value = value
        assert isinstance(type, MockAttributeType)

    def value(self):
        return self._value
