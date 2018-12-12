
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