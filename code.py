import pandas as pd
import numpy as np
import random
from deap import base, creator, tools
from sklearn.metrics import mean_squared_error
#----------------------------------------Data process----------------------------------------------------------
#import movie ratings
movie_cols = ['userId', 'movieId', 'rating', 'timestamp']
ratings = pd.read_csv('http://files.grouplens.org/datasets/movielens/ml-100k/u.data', 
                    sep = '\t',
                    names = movie_cols,
                    encoding = 'latin-1')

ratings = ratings.drop(columns = 'timestamp')

# Take every unique user id and map it to a contiguous user .
u_uniq = ratings.userId.unique()
user2idx = {o:i for i,o in enumerate(u_uniq)}
# Replace that userid with contiguous number.
ratings.userId = ratings.userId.apply(lambda x: user2idx[x]) 
ratings.rating = ratings.rating

#Do the same for movies
m_uniq = ratings.movieId.unique()
movie2idx = {o:i for i,o in enumerate(m_uniq)}
ratings.movieId = ratings.movieId.apply(lambda x: movie2idx[x])

#create table with rating count and average rating for each user
df1 = ratings.groupby('userId')['rating'].agg(['count','mean']).reset_index()
avRating = df1.values

x = ratings.values
usrRatings = np.zeros((943,1682)) #usrRatings[i,j] contains user's i rating on movie j
for i in range (943):
   usrRatings[i] = avRating[i][2]

for i in range(100000):
   usrRatings[x[i,0]][x[i,1]] = x[i,2]

forEval = usrRatings - avRating[0][2]

#---------------------Genetic Algorithm----------------------------------------------------------

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox = base.Toolbox()
# Attribute generator 
toolbox.register("attr_bool", random.randint, 0, 1)
# Structure initializers
toolbox.register("individual", tools.initRepeat, creator.Individual, 
    toolbox.attr_bool, 943)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evalOneMax(individual):
    indiv = np.asarray(individual)
    pred = np.zeros(1682)
    for i in range (0,1682):
        temp = forEval[:,i]
        pred[i] = avRating[0][2] + sum(indiv*temp)/sum(indiv)
    iratings = usrRatings[0,:]
    mse = mean_squared_error(iratings, pred)*(-1)
    return mse,

toolbox.register("evaluate", evalOneMax)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

#def main():
pop = toolbox.population(n=10)

# Evaluate the entire population
fitnesses = list(map(toolbox.evaluate, pop))
for ind, fit in zip(pop, fitnesses):
    ind.fitness.values = fit

CXPB, MUTPB = 0.5, 0.2

# Extracting all the fitnesses of 
fits = [ind.fitness.values[0] for ind in pop]

# Variable keeping track of the number of generations
g = 0
    
# Begin the evolution
while max(fits) < 100 and g < 1000:
    # A new generation
    g = g + 1
    print("-- Generation %i --" % g)

# Select the next generation individuals
offspring = toolbox.select(pop, len(pop))
# Clone the selected individuals
offspring = list(map(toolbox.clone, offspring))

# Apply crossover and mutation on the offspring
for child1, child2 in zip(offspring[::2], offspring[1::2]):
    if random.random() < CXPB:
        toolbox.mate(child1, child2)
        del child1.fitness.values
        del child2.fitness.values

for mutant in offspring:
    if random.random() < MUTPB:
        toolbox.mutate(mutant)
        del mutant.fitness.values

# Evaluate the individuals with an invalid fitness
invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
fitnesses = map(toolbox.evaluate, invalid_ind)
for ind, fit in zip(invalid_ind, fitnesses):
    ind.fitness.values = fit

pop[:] = offspring

