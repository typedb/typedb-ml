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

from __future__ import print_function

import os
import shutil
import subprocess
import subprocess as sp
import tarfile
import tempfile
import zipfile
from time import sleep


class ZipFile(zipfile.ZipFile):
    """ Custom ZipFile class handling file permissions. """
    def _extract_member(self, member, targetpath, pwd):
        if not isinstance(member, zipfile.ZipInfo):
            member = self.getinfo(member)

        targetpath = super()._extract_member(member, targetpath, pwd)

        attr = member.external_attr >> 16
        if attr != 0:
            os.chmod(targetpath, attr)
        return targetpath


class TypeDBServer(object):

    def __init__(self, distribution_location):
        self.__distribution_location = distribution_location
        self.__distribution_root_dir = None
        self.__unpacked_dir = None
        self.__pid = None

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self.stop()

    def start(self):
        if not self.__unpacked_dir:
            self._unpack()
        popen = subprocess.Popen(["typedb", "server"],
                                 cwd=os.path.join(self.__unpacked_dir, self.__distribution_root_dir))
        sleep(5)
        self.__pid = popen.pid

    def stop(self):
        sp.check_call([
            "kill", f"{self.__pid}"
        ], cwd=os.path.join(self.__unpacked_dir, self.__distribution_root_dir))
        shutil.rmtree(self.__unpacked_dir)
        self.__pid = None

    def _unpack(self):
        self.__unpacked_dir = tempfile.mkdtemp(prefix='typedb')
        with tarfile.open(self.__distribution_location) as tf:
            def is_within_directory(directory, target):
                abs_directory = os.path.abspath(directory)
                prefix = os.path.commonprefix([abs_directory, os.path.abspath(target)])
                return prefix == abs_directory
            
            def safe_extract(tar, path="."):
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Detected unsafe path traversal in .tar file")
                tar.extractall(path)

            safe_extract(tf, self.__unpacked_dir)
            self.__distribution_root_dir = os.path.commonpath(tf.getnames()[1:])

    @property
    def typedb_binary_location(self):
        return os.path.join(self.__unpacked_dir, self.__distribution_root_dir)
