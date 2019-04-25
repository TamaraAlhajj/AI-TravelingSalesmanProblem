import numpy as np
import matplotlib.pyplot as plt
import time


## MACROS ##

CITIES = np.loadtxt(open("BigData.csv", "rb"), dtype=int, delimiter=" ")
TOTAL_CITIES = np.shape(CITIES)[0]
LONG_LAT = np.loadtxt(open("BigDataXY.csv", "rb"), delimiter=",", skiprows=1)
LONG_LAT = np.delete(LONG_LAT, 0, 1)


def totalDistance(tour):
    totalDistanceTraveled = 0

    for i in range(TOTAL_CITIES - 1):
        A, B = tour[i], tour[i+1]
        totalDistanceTraveled += CITIES[A, B]

    # close loop
    A, B = tour[-1], tour[0]
    totalDistanceTraveled += CITIES[A, B]

    return totalDistanceTraveled


def NearestNeighbour(u=None):
    """
    Running time: O(n^2)

    Initialize all vertices as unvisited.
    Select an arbitrary vertex, set it as the current vertex u. Mark u as visited.
    Find out the shortest edge connecting the current vertex u and an unvisited vertex v.
    Set v as the current vertex u. Mark v as visited.
    If all the vertices in the domain are visited, then terminate. Else, go to step 3. 
    """

    visited = np.zeros(TOTAL_CITIES)
    solution = np.zeros(TOTAL_CITIES)

    if u == None:
        u = np.random.randint(0, TOTAL_CITIES)
    visited[u] = 1
    tour = [u]

    while len(tour) < TOTAL_CITIES:
        minDistance = np.inf
        minVertex = None
        for v in range(0, TOTAL_CITIES):
            if visited[v] == 0:
                if CITIES[u, v] < minDistance:
                    minVertex = v

        u = minVertex
        visited[minVertex] = 1
        tour.append(u)

    return tour


print("\n\nGreedy Algorithm: Nearest Neighbour\n\n")

Solutions = []
Distances = []
Runtimes = []

bestSoln = NearestNeighbour()
bestDist = totalDistance(bestSoln)

for u in range(TOTAL_CITIES):
    start_time = time.time()
    solution = NearestNeighbour(u)
    runtime = time.time() - start_time  # seconds
    distance = totalDistance(solution)

    if distance < bestDist:
        bestSoln = solution
        bestDist = distance

    Solutions.append(solution)
    Runtimes.append(runtime)
    Distances.append(distance)

TotalMeanDist = np.mean(Distances)
TotalMeanRuntime = np.mean(Runtimes)

print("Total mean distance: ", TotalMeanDist)
print("Total mean Runtime: ", TotalMeanRuntime)

print("""
Best solution path found: {} 
Distance Traveled: {} km
""".format(["city{}".format(i) for i in bestSoln], bestDist))


plt.plot([LONG_LAT[bestSoln[i % 15]][0] for i in range(16)], [
         LONG_LAT[bestSoln[i % 15]][1] for i in range(16)], 'xb-')
plt.show()
