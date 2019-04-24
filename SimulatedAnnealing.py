import copy
import matplotlib.pyplot as plt
import numpy as np
from math import exp

## MACROS ##

#CITIES = np.loadtxt(open("SmallData.csv", "rb"), dtype=int, delimiter=" ")
CITIES = np.loadtxt(open("BigData.csv", "rb"), dtype=int, delimiter=" ")
TOTAL_CITIES = np.shape(CITIES)[0]
TEMPERATURE = 10000
COOLING_RATE = 0.005

## FUNCATIONALITY ##

def totalDistance(tour):
    totalDistanceTraveled = 0

    for i in range(TOTAL_CITIES - 1):
        A, B = tour[i], tour[i+1]
        totalDistanceTraveled += CITIES[A, B]
    
    # close loop 
    A, B = tour[-1], tour[0]
    totalDistanceTraveled += CITIES[A, B] 

    return totalDistanceTraveled

def SimulatedAnnealing(temperature, coolingRate):

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

        neighbour = [j,j-1,i,i-1]
        neighbour = [i,i-1,j,j-1]

        oldDistances = totalDistance(currentSolution)
        newDistances = totalDistance(newTour)
        
        if(totalDistance(currentSolution) < totalDistance(bestTour)):
            # keep better solution
            bestTour = currentSolution.copy()
        else:
            # dist is longer, keep with certain probability
            if(np.exp((oldDistances - newDistances / temperature) > np.random.uniform())):
                currentSolution = newTour.copy()

    return (initialTourDistance, bestTour)

## RUN ##
init, final = SimulatedAnnealing(TEMPERATURE, COOLING_RATE)

print("Initial tour distance: ", init)
print("\nFinal solution: ", final)
print("Distance: ", totalDistance(final))
