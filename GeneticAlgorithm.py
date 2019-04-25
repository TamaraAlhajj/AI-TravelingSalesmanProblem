import numpy as np
import matplotlib.pyplot as plt
import time

"""
Genetic representation: list of tour order based on matrix of dist
Fitness function: evaluate total distance of tour
Mating: Order 1 Crossover
Parent 1: 8 4 7 3 6 2 5 1 9 0
                ---------
Parent 2: 0 1 2 3 4 5 6 7 8 9
            x x x   x x

Child 1:  0 4 7 3 6 2 5 1 8 9
          2 2 2 1 1 1 1 1 2 2

---------
Algorithm
---------
Init Population
Fitness Function
Selection
Crossover
Mutation

"""


## MACROS ##

CITIES = np.loadtxt(open("BigData.csv", "rb"), dtype=int, delimiter=" ")
TOTAL_CITIES = np.shape(CITIES)[0]
LONG_LAT = np.loadtxt(open("BigDataXY.csv", "rb"), delimiter=",", skiprows=1)
LONG_LAT = np.delete(LONG_LAT, 0, 1)
MUTATION_RATE = 0.6


def generateIndividual():
    return np.random.permutation(TOTAL_CITIES)


def generatePopulation(populationSize):

    population = []

    while len(population) < populationSize:
        newIndividual = generateIndividual()
        population.append(newIndividual)

    return population


def totalDistance(individual):
    totalDistanceTraveled = 0

    for i in range(len(individual) - 1):
        geneA, geneB = individual[i], individual[i+1]
        totalDistanceTraveled += CITIES[geneA, geneB]

    """
    # close loop
    A, B = tour[-1], tour[0]
    totalDistanceTraveled += CITIES[A, B]
    """

    return totalDistanceTraveled


def evaluateFitness(individual):
    # longer distance ==> less fit
    totalDistanceTraveled = totalDistance(individual)
    return 1 / totalDistanceTraveled


def selection(population):
    """ Sort by fittest and terminate the bottom half """

    updatedPopulation = list()
    for individual in population:
        updatedPopulation.append((individual, evaluateFitness(individual)))

    survivalOfTheFittest = sorted(
        updatedPopulation, key=lambda individual: individual[1], reverse=True)
    half = len(survivalOfTheFittest)//2
    del survivalOfTheFittest[half:]

    return [x[0] for x in survivalOfTheFittest]


def reproduce(individual1, individual2):
    """ Reproduction using Order 1 Crossover Operator """

    child = individual1.copy()

    fourth = TOTAL_CITIES//4
    crossingIndex = np.random.randint(0, TOTAL_CITIES-fourth)

    newChunk = individual1[crossingIndex:crossingIndex+fourth].copy()
    newGenes = list()
    copied = 0

    for gene in individual2:
        if (gene not in newChunk):
            newGenes.append(gene)

    for i in range(0, crossingIndex):
        if(copied >= len(newGenes)):
            break
        child[i] = newGenes[copied]
        copied += 1

    for i in range(crossingIndex + fourth, len(child)):
        if(copied >= len(newGenes)):
            break
        child[i] = newGenes[copied]
        copied += 1

    return child


def createChildren(individual1, individual2, totalOffspring):
    children = list()

    while len(children) < totalOffspring:
        children.append(reproduce(individual1, individual2))

    return children


def mutate(individual):
    a = np.random.randint(0, len(individual))
    b = np.random.randint(0, len(individual))

    while(a == b):
        b = np.random.randint(0, len(individual))

    temp = individual[a].copy()
    individual[a] = individual[b]
    individual[b] = temp


def GeneticAlgorithm(initialPopulationSize, terminationGeneration, mutationRate):

    gen = 0
    population = generatePopulation(initialPopulationSize)

    while (gen < terminationGeneration):

        # evaluate each individual and return the fittest
        survivors = selection(population)

        # pick the fittest two for most reproduction
        if(np.random.randint(0, 100) < 80):
            newGen = createChildren(survivors[0], survivors[1], 10)
            np.concatenate((survivors, newGen))

        if(np.random.randint(0, 100) < 20):
            newGen = createChildren(survivors[2], survivors[3], 5)
            np.concatenate((survivors, newGen))

        # mutate
        mutations = 0
        while mutations < len(survivors) // mutationRate*10:
            individual = survivors[np.random.randint(3, len(survivors))]
            mutate(individual)
            mutations += 1
        gen += 1

    bestFit = selection(population)[0]
    return (bestFit)

## RUN ##
print("\n\nGenetic Algorithm\n\n")

pop = 30
gen = 500
runs = 0

Solutions1 = []
Distances1 = []
Runtimes1 = []

bestSoln = GeneticAlgorithm(pop, gen, MUTATION_RATE)
bestDist = totalDistance(bestSoln)

while runs < 3:
    start_time = time.time()
    solution = GeneticAlgorithm(pop, gen, MUTATION_RATE)
    runtime = time.time() - start_time  # seconds
    distance = totalDistance(solution)

    if distance < bestDist:
        bestSoln = solution
        bestDist = distance

    Solutions1.append(solution)
    Runtimes1.append(runtime)
    Distances1.append(distance)
    runs += 1

pop = 30
gen = 1000
runs = 0

Solutions2 = []
Distances2 = []
Runtimes2 = []

while runs < 3:
    start_time = time.time()
    solution = GeneticAlgorithm(pop, gen, MUTATION_RATE)
    runtime = time.time() - start_time  # seconds
    distance = totalDistance(solution)

    if distance < bestDist:
        bestSoln = solution
        bestDist = distance

    Solutions2.append(solution)
    Runtimes2.append(runtime)
    Distances2.append(distance)
    runs += 1

pop = 100
gen = 500
runs = 0

Solutions3 = []
Distances3 = []
Runtimes3 = []

while runs < 3:
    start_time = time.time()
    solution = GeneticAlgorithm(pop, gen, MUTATION_RATE)
    runtime = time.time() - start_time  # seconds
    distance = totalDistance(solution)

    if distance < bestDist:
        bestSoln = solution
        bestDist = distance

    Solutions3.append(solution)
    Runtimes3.append(runtime)
    Distances3.append(distance)
    runs += 1

pop = 100
gen = 1000
runs = 0

Solutions4 = []
Distances4 = []
Runtimes4 = []

while runs < 3:
    start_time = time.time()
    solution = GeneticAlgorithm(pop, gen, MUTATION_RATE)
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


