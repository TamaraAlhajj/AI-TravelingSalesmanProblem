import matplotlib.pyplot as plt
import numpy as np
import time

## MACROS ##

CITIES = np.loadtxt(open("BigData.csv", "rb"), dtype=int, delimiter=" ")
LONG_LAT = np.loadtxt(open("BigDataXY.csv", "rb"), delimiter=",", skiprows=1)
LONG_LAT = np.delete(LONG_LAT, 0, 1)
TOTAL_CITIES = np.shape(CITIES)[0]


## FUNCTIONALITY ##

def totalDistance(tour):
    totalDistanceTraveled = 0

    for i in range(TOTAL_CITIES - 1):
        A, B = tour[i], tour[i+1]
        totalDistanceTraveled += CITIES[A, B]

    """ 
    # close loop 
    A, B = tour[-1], tour[0]
    totalDistanceTraveled += CITIES[A, B]  
    """

    return totalDistanceTraveled


def SimulatedAnnealing(temperature, coolingRate):
    """ 
    pick an initial solution
    set an initial temperature
    choose the next neighbour of the current solution:
        if the neighbour is better, make neighbour the current solution
        if the neighbour is “worse”, probabilistically make neighbour the current solution
    go to 3.
    """

    currentSolution = np.random.permutation(TOTAL_CITIES)
    bestTour = currentSolution.copy()

    initialTourDistance = totalDistance(currentSolution)

    while temperature > 1:
        temperature *= 1 - coolingRate

        newTour = currentSolution.copy()

        # swap two CITIES in the newTour
        i, j = np.random.choice(TOTAL_CITIES, 2, replace=False)
        temp = newTour[i].copy()
        newTour[i] = newTour[j]
        newTour[j] = temp

        neighbour = [j, j-1, i, i-1]
        neighbour = [i, i-1, j, j-1]

        energyCurrent = totalDistance(currentSolution)
        energyChange = totalDistance(newTour)
        loss = energyChange - energyCurrent

        if(loss > 0):
            # energy gain, keep better solution
            bestTour = currentSolution.copy()
        else:
            # dist is longer, keep with certain probability
            # Gibbs distribution:
            # > probability that a system will be in a certain state
            # > as a function of that state's energy scaled by the current temperature

            if(np.exp(loss / temperature) > np.random.uniform()):
                currentSolution = newTour.copy()

    return (initialTourDistance, bestTour)

## RUN ##
print("\n\nSimulated Annealing Algorithm\n\n")

TEMPERATURE = 10000000
COOLING_RATE = 0.003
runs = 0

Solutions1 = []
Distances1 = []
Runtimes1 = []

bestSoln = SimulatedAnnealing(TEMPERATURE, COOLING_RATE)[1]
bestDist = totalDistance(bestSoln)

while runs < 3:
    start_time = time.time()
    solution = SimulatedAnnealing(TEMPERATURE, COOLING_RATE)[1]
    runtime = time.time() - start_time  # seconds
    distance = totalDistance(solution)

    if distance < bestDist:
        bestSoln = solution
        bestDist = distance

    Solutions1.append(solution)
    Runtimes1.append(runtime)
    Distances1.append(distance)
    runs += 1

TEMPERATURE = 10000000
COOLING_RATE = 0.005
runs = 0

Solutions2 = []
Distances2 = []
Runtimes2 = []

while runs < 3:
    start_time = time.time()
    solution = SimulatedAnnealing(TEMPERATURE, COOLING_RATE)[1]
    runtime = time.time() - start_time  # seconds
    distance = totalDistance(solution)

    if distance < bestDist:
        bestSoln = solution
        bestDist = distance

    Solutions2.append(solution)
    Runtimes2.append(runtime)
    Distances2.append(distance)
    runs += 1

TEMPERATURE = 10000000
COOLING_RATE = 0.008
runs = 0

Solutions3 = []
Distances3 = []
Runtimes3 = []

while runs < 3:
    start_time = time.time()
    solution = SimulatedAnnealing(TEMPERATURE, COOLING_RATE)[1]
    runtime = time.time() - start_time  # seconds
    distance = totalDistance(solution)

    if distance < bestDist:
        bestSoln = solution
        bestDist = distance

    Solutions3.append(solution)
    Runtimes3.append(runtime)
    Distances3.append(distance)
    runs += 1

TEMPERATURE = 10000000
COOLING_RATE = 0.01
runs = 0

Solutions4 = []
Distances4 = []
Runtimes4 = []

while runs < 3:
    start_time = time.time()
    solution = SimulatedAnnealing(TEMPERATURE, COOLING_RATE)[1]
    runtime = time.time() - start_time  # seconds
    distance = totalDistance(solution)

    if distance < bestDist:
        bestSoln = solution
        bestDist = distance

    Solutions4.append(solution)
    Runtimes4.append(runtime)
    Distances4.append(distance)
    runs += 1

MeanDistances = [np.mean(Distances1)]
MeanRuntimes = [np.mean(Runtimes1)]

MeanDistances.append(np.mean(Distances2))
MeanRuntimes.append(np.mean(Runtimes2))

MeanDistances.append(np.mean(Distances3))
MeanRuntimes.append(np.mean(Runtimes3))

MeanDistances.append(np.mean(Distances4))
MeanRuntimes.append(np.mean(Runtimes4))

TotalMeanDist = np.mean(MeanDistances)
TotalMeanRuntime = np.mean(MeanRuntimes)

print("Mean distances: ", MeanDistances)
print("Mean runtimes: ", MeanRuntimes)
print("Total mean distance: ", TotalMeanDist)
print("Total mean Runtime: ", TotalMeanRuntime)

print("""
Best solution path found: {} 
Distance Traveled: {} km
""".format(["city{}".format(i) for i in bestSoln], bestDist))


plt.plot([LONG_LAT[bestSoln[i % 15]][0] for i in range(16)], [
         LONG_LAT[bestSoln[i % 15]][1] for i in range(16)], 'xb-')
plt.show()
