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

"""
This code is a quick sanity check to check if the sampling strategy of algorithm 2 does do as it reads. I found that
line 3 of this algorithm isn't implemented in the author's code, and doesn't align with the description of the
approach either
"""

def sample(num_neighbours, u):
    # val = k * u
    return [u * num_neighbours] * num_neighbours


K = 2
B = [[]] * (K + 1)
B_start = [1, 5, 9]
B[K] = B_start

neighbour_sizes = [5, 2]

for k in range(K, 0, -1):
    B[k-1] = B[k]  # This is the debatable line, line 3 of algorithm 2. Comment out to see the difference
    for u in B[k]:
        B[k-1] = B[k-1] + sample(neighbour_sizes[k-1], u)

print(B)
print([len(b) for b in B])