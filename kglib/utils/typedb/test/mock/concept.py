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

class MockConcept:
    def __init__(self, id):
        self.id = id


class MockType(MockConcept):
    def __init__(self, id, label, base_type):
        super().__init__(id)
        self._label = Label(label)
        assert base_type in {'ENTITY', 'RELATION', 'ATTRIBUTE'}
        self.base_type = base_type

    def get_label(self):
        return self._label


class Label:
    def __init__(self, name):
        self._name = name

    def name(self):
        return self._name


class ValueType:
    def __init__(self, name):
        self.name = name


class MockAttributeType(MockType):
    def __init__(self, id, label, base_type, value_type):
        super().__init__(id, label, base_type)
        assert value_type in {'STRING', 'LONG', 'DOUBLE', 'DATETIME', 'BOOLEAN'}
        self._value_type = ValueType(value_type)

    def get_value_type(self):
        return self._value_type


class MockThing(MockConcept):
    def __init__(self, id, type):
        super().__init__(id)
        assert(isinstance(type, MockType))
        self._type = type
        self._id = id

    def get_type(self):
        return self._type

    def get_iid(self):
        return self._id

    def is_entity(self):
        return 'ENTITY' in self._type.base_type

    def is_attribute(self):
        return 'ATTRIBUTE' in self._type.base_type

    def is_relation(self):
        return 'RELATION' in self._type.base_type

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
