import numpy as np


CITIES = np.loadtxt(open("SmallData.csv", "rb"), dtype=int, delimiter=" ")
#CITIES = np.loadtxt(open("BigData.csv", "rb"), dtype=int, delimiter=" ")

TOTAL_CITIES = np.shape(CITIES)[0]


def generateIndividual():
    return np.random.permutation(TOTAL_CITIES)


def generatePopulation (populationSize):

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
    
    # close loop
    geneA, geneB = individual[-1], individual[0]
    totalDistanceTraveled += CITIES[geneA, geneB] 

    return totalDistanceTraveled


def evaluateFitness (individual):
    # longer distance ==> less fit
    totalDistanceTraveled = totalDistance(individual)
    return 1 / totalDistanceTraveled


def selection(population):
    """ Sort by fittest and terminate the bottom half """

    updatedPopulation = list()
    for individual in population:
        updatedPopulation.append((individual, evaluateFitness(individual)))
    
    survivalOfTheFittest = sorted(updatedPopulation, key=lambda individual: individual[1], reverse=True)
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
    a = np.random.randint(0,len(individual))
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
        try:
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

        except IndexError:
            break
    
    bestFit = selection(population)[0]
    return (bestFit, gen)


solution, generations = GeneticAlgorithm(30, 1000, 0.6)
distance = totalDistance(solution)

print("""
Final solution path found: {} 
After {} generations.
Distance Traveled: {} km
""".format(solution, generations, distance))