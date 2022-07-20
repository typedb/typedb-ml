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

from typedb.api.concept.type.attribute_type import AttributeType


class Thing:

    VALUE_TYPES = (
        AttributeType.ValueType.OBJECT,
        AttributeType.ValueType.STRING,
        AttributeType.ValueType.LONG,
        AttributeType.ValueType.DOUBLE,
        AttributeType.ValueType.DATETIME,
        AttributeType.ValueType.BOOLEAN
    )

    def __init__(self, iid, type_label, base_type, value_type=None, value=None):
        self._hash = None
        self.iid = iid
        self.type_label = type_label
        self.base_type = base_type

        # If the thing is an attribute
        self.value_type = value_type
        self.value = value

        # TODO Make attribute a separate class
        if self.base_type == 'attribute':
            if self.value_type is None:
                raise ValueError('Attribute value_type must be provided')
            if self.value is None:
                raise ValueError('Attribute value must be provided')

    def __str__(self):
        string = f'<{self.type_label}, {self.iid}'
        if self.base_type == 'attribute':
            string += f': {self.value}'
        return string + '>'

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(other, self.__class__):
            if self.iid == other.iid:
                assert self.__dict__ == other.__dict__
                return True
            else:
                return False
        return NotImplemented

    def __ne__(self, other):
        """Overrides the default implementation (unnecessary in Python 3)"""
        x = self.__eq__(other)
        if x is not NotImplemented:
            return not x
        return NotImplemented

    def __hash__(self):
        """Overrides the default implementation"""
        if self._hash is None:
            self._hash = hash(self.iid)
        return self._hash


def build_thing(typedb_thing):

    iid = typedb_thing.get_iid()
    type_label = typedb_thing.get_type().get_label().name()
    if typedb_thing.is_entity():
        base_type = "entity"
    elif typedb_thing.is_relation():
        base_type = "relation"
    elif typedb_thing.is_attribute():
        base_type = "attribute"
    else:
        raise RuntimeError("Unexpected Concept")

    if base_type == 'attribute':
        value_type = typedb_thing.get_type().get_value_type()
        assert value_type in Thing.VALUE_TYPES
        value = typedb_thing.get_value()

        return Thing(iid, type_label, base_type, value_type, value)

    return Thing(iid, type_label, base_type)
