"""
--------------------------------------------
Runtime Analysis for Dynamic solution to TSP
--------------------------------------------
Naive solution takes O(n!)
    > brute force search
Dynamic program takes O(n^2 * 2^n)
    > at most 2^n * n subproblems
    > each subproblem takes linear time to solve
    > improvement but still not polynomial
"""

import numpy as np
import itertools


## MACROS ##

#CITIES = np.loadtxt(open("SmallData.csv", "rb"), dtype=int, delimiter=" ")
CITIES = np.loadtxt(open("BigData.csv", "rb"), dtype=int, delimiter=" ")
TOTAL_CITIES = np.shape(CITIES)[0]
TOUR = np.random.permutation(TOTAL_CITIES)

def findsubsets(S,m):
    subset = set()
    for i in itertools.combinations(S, m):
        if(1 in i):
            subset.add(i)
    return subset

def totalDistance(tour):
    totalDistanceTraveled = 0

    for i in range(TOTAL_CITIES - 1):
        A, B = tour[i], tour[i+1]
        totalDistanceTraveled += CITIES[A, B]
    
    # close loop 
    A, B = tour[-1], tour[0]
    totalDistanceTraveled += CITIES[A, B] 

    return totalDistanceTraveled

def DynamicTSP():
    distances = 
    for i in range(2, TOTAL_CITIES):

        for subset in findsubsets(TOUR, i):
            for j in range(2, TOTAL_CITIES):
                
