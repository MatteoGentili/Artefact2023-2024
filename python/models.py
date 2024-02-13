import pickle
from abc import abstractmethod
from gurobipy import *
import random
import matplotlib.pyplot as plt

import numpy as np
from sklearn.cluster import KMeans
from collections import Counter

class BaseModel(object):
    """
    Base class for models, to be used as coding pattern skeleton.
    Can be used for a model on a single cluster or on multiple clusters"""

    def __init__(self):
        """Initialization of your model and its hyper-parameters"""
        pass

    @abstractmethod
    def fit(self, X, Y):
        """Fit function to find the parameters according to (X, Y) data.
        (X, Y) formatting must be so that X[i] is preferred to Y[i] for all i.

        Parameters
        ----------
        X: np.ndarray
            (n_samples, n_features) features of elements preferred to Y elements
        Y: np.ndarray
            (n_samples, n_features) features of unchosen elements
        """
        # Customize what happens in the fit function
        return

    @abstractmethod
    def predict_utility(self, X):
        """Method to call the decision function of your model

        Parameters:
        -----------
        X: np.ndarray
            (n_samples, n_features) list of features of elements

        Returns
        -------
        np.ndarray:
            (n_samples, n_clusters) array of decision function value for each cluster.
        """
        # Customize what happens in the predict utility function
        return

    def predict_preference(self, X, Y):
        """Method to predict which pair is preferred between X[i] and Y[i] for all i.
        Returns a preference for each cluster.

        Parameters
        -----------
        X: np.ndarray
            (n_samples, n_features) list of features of elements to compare with Y elements of same index
        Y: np.ndarray
            (n_samples, n_features) list of features of elements to compare with X elements of same index

        Returns
        -------
        np.ndarray:
            (n_samples, n_clusters) array of preferences for each cluster. 1 if X is preferred to Y, 0 otherwise
        """
        X_u = self.predict_utility(X)
        Y_u = self.predict_utility(Y)

        return (X_u - Y_u > 0).astype(int)

    def predict_cluster(self, X, Y):
        """Predict which cluster prefers X over Y THE MOST, meaning that if several cluster prefer X over Y, it will
        be assigned to the cluster showing the highest utility difference). The reversal is True if none of the clusters
        prefer X over Y.
        Compared to predict_preference, it indicates a cluster index.

        Parameters
        -----------
        X: np.ndarray
            (n_samples, n_features) list of features of elements to compare with Y elements of same index
        Y: np.ndarray
            (n_samples, n_features) list of features of elements to compare with X elements of same index

        Returns
        -------
        np.ndarray:
            (n_samples, ) index of cluster with highest preference difference between X and Y.
        """
        X_u = self.predict_utility(X)
        Y_u = self.predict_utility(Y)

        return np.argmax(X_u - Y_u, axis=1)

    def save_model(self, path):
        """Save the model in a pickle file. Don't hesitate to change it in the child class if needed

        Parameters
        ----------
        path: str
            path indicating the file in which the model will be saved
        """
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load_model(clf, path):
        """Load a model saved in a pickle file. Don't hesitate to change it in the child class if needed

        Parameters
        ----------
        path: str
            path indicating the path to the file to load
        """
        with open(path, "rb") as f:
            model = pickle.load(f)
        return model


class RandomExampleModel(BaseModel):
    """Example of a model on two clusters, drawing random coefficients.
    You can use it to understand how to write your own model and the data format that we are waiting for.
    This model does not work well but you should have the same data formatting with TwoClustersMIP.
    """

    def __init__(self):
        self.seed = 444
        self.weights = self.instantiate()

    def instantiate(self):
        """No particular instantiation"""
        return []

    def fit(self, X, Y):
        """fit function, sets random weights for each cluster. Totally independant from X & Y.

        Parameters
        ----------
        X: np.ndarray
            (n_samples, n_features) features of elements preferred to Y elements
        Y: np.ndarray
            (n_samples, n_features) features of unchosen elements
        """
        np.random.seed(self.seed)
        num_features = X.shape[1]
        weights_1 = np.random.rand(num_features) # Weights cluster 1
        weights_2 = np.random.rand(num_features) # Weights cluster 2

        weights_1 = weights_1 / np.sum(weights_1)
        weights_2 = weights_2 / np.sum(weights_2)
        self.weights = [weights_1, weights_2]
        return self

    def predict_utility(self, X):
        """Simple utility function from random weights.

        Parameters:
        -----------
        X: np.ndarray
            (n_samples, n_features) list of features of elements

        Returns
        -------
        np.ndarray:
            (n_samples, n_clusters) array of decision function value for each cluster.
        """
        u_1 = np.dot(X, self.weights[0]) # Utility for cluster 1 = X^T.w_1
        u_2 = np.dot(X, self.weights[1]) # Utility for cluster 2 = X^T.w_2
        return np.stack([u_1, u_2], axis=1) # Stacking utilities over cluster on axis 1


class TwoClustersMIP(BaseModel):
    """Skeleton of MIP you have to write as the first exercise.
    You have to encapsulate your code within this class that will be called for evaluation.
    """

    def __init__(self, n_pieces, n_clusters):
        """Initialization of the MIP Variables

        Parameters
        ----------
        n_pieces: int
            Number of pieces for the utility function of each feature.
        n_clusters: int
            Number of clusters to implement in the MIP.
        """
        self.seed = 123
        self.L = n_pieces
        self.K = n_clusters
        self.epsilon = 0.0001
        self.model = self.instantiate()

    def instantiate(self): 
        
        model = Model("MIP_first_model")
        return model
        
    

    def fit(self, X, Y):
        """Estimation of the parameters - To be completed.

        Parameters
        ----------
        X: np.ndarray
            (n_samples, n_features) features of elements preferred to Y elements
        Y: np.ndarray
            (n_samples, n_features) features of unchosen elements
        """
        self.n = X.shape[1] # Number of features/criteria
        self.P = X.shape[0] # Number of elements
        max = np.ones(self.n) # Max value for each feature
        min = np.zeros(self.n) # Min value for each feature
        M = 2.5 # Big M value

        # Reference to the UTA University exercice 
        def BreakPoints(i, l):
            return min[i] + l * (max[i] - min[i]) / self.L
        def LastIndex(x, i):
            return np.floor(self.L * (x - min[i]) / (max[i] - min[i]))
        
        # Varariables of the model
        self.U = {
            (cluster, criteria, linearSegmentUTA): self.model.addVar(
                vtype=GRB.CONTINUOUS, lb=0, name="u_{}_{}_{}".format(cluster, criteria, linearSegmentUTA), ub=1)
                for cluster in range(self.K)
                for criteria in range(self.n)
                for  linearSegmentUTA in range(self.L+1) # Need to add +1 because of the last segment {Uk and UK+1}
        }

        uiKxiJ = {} # uik_xij[k, i, j] = U_k(i, X[j, i])
        for k in range(self.K):
            for i in range(self.n):
                for j in range(self.P):
                    l = LastIndex(X[j, i], i)
                    bp = BreakPoints(i, l)
                    bp1 = BreakPoints(i, l+1)
                    uiKxiJ[k, i, j] = self.U[(k, i, l)] + ((X[j, i] - bp) / (bp1 - bp)) * (self.U[(k, i, l+1)] - self.U[(k, i, l)])
        
        uiKyiJ = {}
        for k in range(self.K):
            for i in range(self.n):
                for j in range(self.P):
                    l = LastIndex(Y[j, i], i)
                    bp = BreakPoints(i, l)
                    bp1 = BreakPoints(i, l+1)
                    uiKyiJ[k, i, j] = self.U[(k, i, l)] + ((Y[j, i] - bp) / (bp1 - bp)) * (self.U[(k, i, l+1)] - self.U[(k, i, l)])

        uk_xj = {}
        for k in range(self.K):
            for j in range(self.P):
                uk_xj[k, j] = quicksum(uiKxiJ[k, i, j] for i in range(self.n))
        
        uk_yj = {}
        for k in range(self.K):
            for j in range(self.P):
                uk_yj[k, j] = quicksum(uiKyiJ[k, i, j] for i in range(self.n))

        # Overestimation and underestimation variables σy+, σy− and σx+, σx−
        self.sigmaxPLUS = {(jX): self.model.addVar(vtype=GRB.CONTINUOUS, lb=0, name="sigmaxp_{}".format(jX), ub=1) for jX in range(self.P)}
        self.sigmayPLUS = {(jY): self.model.addVar(vtype=GRB.CONTINUOUS, lb=0, name="sigmayp_{}".format(jY), ub=1) for jY in range(self.P)}
        self.sigmaxMINUS = {(jX): self.model.addVar(vtype=GRB.CONTINUOUS, lb=0, name="sigmaxm_{}".format(jX), ub=1) for jX in range(self.P)}
        self.sigmayMINUS = {(jY): self.model.addVar(vtype=GRB.CONTINUOUS, lb=0, name="sigmaym_{}".format(jY), ub=1) for jY in range(self.P)}

        # Delta(δ) allow us to which cluster the element belongs to
        self.delta = {(k, j): self.model.addVar(vtype=GRB.BINARY, name="delta{}_{}".format(k, j))for k in range(self.K)for j in range(self.P)}


        ##########################
        ###### Constraints #######
        ##########################

        ## Constraint 1:
        ### Monotonicity of the utility function:
        self.model.addConstrs((self.U[(k, i, l+1)] - self.U[(k, i, l)]>=self.epsilon for k in range(self.K) for i in range(self.n) for l in range(self.L)))

        ## Constraint 2:
        ### Normalisation of the utility function
        # ui(xi0) = 0
        self.model.addConstrs((self.U[(k, i, 0)] == 0 for k in range(self.K) for i in range(self.n)))
        # Σu_i(xi) = 1
        self.model.addConstrs((quicksum(self.U[(k, i, self.L)] for i in range(self.n)) == 1 for k in range(self.K)))

        ## Constraint 3:
        ### There is at least one cluster k ux > uy in this cluster
        # Σδ1(k, j) >= 1 
        for j in range(self.P):self.model.addConstr(quicksum(self.delta[(k, j)] for k in range(self.K)) >= 1)

        ## Constraint 4:
        ### Overestimation and underestimation variables
        # M(1 − δ) ≤ x − x0 < M.δ
        # x − x0 < M.δ ==> x − x0 <= M.δ - epsilon
        self.model.addConstrs((uk_xj[k, j] - self.sigmaxPLUS[j] + self.sigmaxMINUS[j] - (uk_yj[k, j] - self.sigmayPLUS[j] + self.sigmayMINUS[j] )<= M*self.delta[(k,j)] - self.epsilon for j in range(self.P) for k in range(self.K)))
        #  M(1 − δ) ≤ x − x0 ==> x − x0 >= -M(1 − δ) 
        self.model.addConstrs((uk_xj[k, j] - self.sigmaxPLUS[j] + self.sigmaxMINUS[j] - uk_yj[k, j] + self.sigmayPLUS[j] - self.sigmayMINUS[j] >= -M*(1-self.delta[(k,j)]) for j in range(self.P) for k in range(self.K)))
        
        ##########################
        ###### Objective #########
        ##########################
        self.model.setObjective(quicksum(self.sigmaxPLUS[j] + self.sigmaxMINUS[j] + self.sigmayPLUS[j] + self.sigmayMINUS[j] for j in range(self.P)), GRB.MINIMIZE)



        self.model.update()
        self.model.optimize()
        ########################################
        ########## Model evaluation ############
        ########################################
        if self.model.status == GRB.INFEASIBLE:
            raise Exception("Infeasible")
        elif self.model.status == GRB.UNBOUNDED:
            raise Exception("Unbounded")
        else:
            
            print("objective function value: ", self.model.objVal)
            self.U = {(k, i, l): self.U[k, i, l].x for k in range(self.K) for i in range(self.n) for l in range(self.L+1)}
            self.delta = {(k, j): self.delta[k, j].x for k in range(self.K) for j in range(self.P)}
            
            # Plot the utility functions
            for k in range(self.K):
                plt.figure(figsize=(5, 3))
                for i in range(self.n):
                    plt.plot([BreakPoints(i, l) for l in range(self.L+1)], [self.U[k, i, l] for l in range(self.L+1)], label=f'Cluster {k}, Feature {i}')
                plt.legend()
                plt.xlabel('Breakpoints')
                plt.ylabel('Utility Value')
                plt.title(f'Utility Functions for Cluster {k}')
                plt.show()

        return self


    def predict_utility(self, X):

        """Return Decision Function of the MIP for X. - To be completed.

        Parameters:
        -----------
        X: np.ndarray
            (n_samples, n_features) list of features of elements
        """
        max_i = np.ones(self.n)*1.01
        min_i = np.ones(self.n)*-0.01

         # Reference to the UTA University exercice 
        def LastIndex(x, i):
                return int(np.floor(self.L * (x - min_i[i]) / (max_i[i] - min_i[i])))

        def BreakPoints(i, l):
            return min_i[i] + l * (max_i[i] - min_i[i])/self.L

        utilities = np.zeros((X.shape[0], self.K))
        for k in range(self.K):
            for j in range(X.shape[0]):
                for i in range(self.n):
                    utilities[j, k] += self.U[k, i, LastIndex(X[j, i], i)] + ((X[j, i] - BreakPoints(i, LastIndex(X[j, i], i))) / (BreakPoints(i, LastIndex(X[j, i], i)+1) - BreakPoints(i, LastIndex(X[j, i], i)))) * (self.U[k, i, LastIndex(X[j, i], i)+1] - self.U[k, i, LastIndex(X[j, i], i)])
         
        return utilities

class Genetique(BaseModel):
    """Skeleton of MIP you have to write as the first exercise.
    You have to encapsulate your code within this class that will be called for evaluation.
    """

    def __init__(self, n_pieces, n_clusters, population_size=48, generations=500):
        """Initialization of the Heuristic Model.
        """
        self.seed = 123
        self.L = n_pieces
        self.K = n_clusters
        self.nb_criteres = 10
        self.epsilon = 0.001
        self.population_size = population_size
        self.generations = generations
        self.population = [] 
        self.model = self.instantiate()
        self.models = self.instantiate()
        self.best_individual = None

    def instantiate(self):
        """Instantiation of the MIP Variables"""
        return

    def fit(self, X, Y):
        """Estimation of the parameters - To be completed.

        Parameters
        ----------
        X: np.ndarray
            (n_samples, n_features) features of elements preferred to Y elements
        Y: np.ndarray
            (n_samples, n_features) features of unchosen elements
        """
        self.run_genetic_algorithm(X, Y)
        return

    def run_genetic_algorithm(self, X, Y):
        """Run the genetic algorithm to optimize the heuristic model."""
        self.initialize_population()

        for generation in range(self.generations):
        
            # Evaluate individuals in the population
            fitness_scores = self.evaluate_population(X, Y)
            print("score de la génération", generation, ":", fitness_scores)
            # Select individuals for reproduction
            selected_indices = self.selection(X,Y)

            # Create new population through crossover and mutation
            new_population = self.reproduce(selected_indices)

            # Replace old population with new population
            self.population = new_population

        # Final evaluation and update the model with the best individual
        best_individual_index = np.argmax(fitness_scores)
        best_individual_score = np.max(fitness_scores)
        self.best_individual = self.population[best_individual_index]

        print("best_individual_score:", best_individual_score)

    def initialize_population(self):
        """Initialize the population with random individuals."""
        for _ in range(self.population_size):
            individual = self.generate_random_individual()
            self.population.append(individual)

    def generate_random_individual(self):
        """Generate a random individual."""
        # Generate random values between 0 and 5 for each element in each criterion
        random_individual = [
            [random.uniform(0, 5) for _ in range(self.L)]
            for _ in range(self.nb_criteres)
        ]

        return random_individual

    def evaluate_population(self, X, Y):
        """Evaluate the fitness of each individual in the population."""
        fitness_scores = []
        for individual in self.population:
            fitness = self.evaluate_individual(individual, X, Y)
            fitness_scores.append(fitness)
        return np.array(fitness_scores)

    def evaluate_individual(self, individual, X, Y):
        
        """Evaluate the fitness of a single individual."""
        indivVals = [[np.sum([individual[l][i]*0.2 for i in range (k)]) for k in range(1,6)]for l in range(self.nb_criteres)]
        score = []
        for i in range(len(X)):
            scoreiX = []
            scoreiY = []
            for u in range(self.nb_criteres):
                indiceX = int(np.floor((X[i][u]-0.01) / 0.2))
                #print("indiceX", indiceX)
                scoreIndivICritUX = indivVals[u][indiceX] + individual[u][indiceX]*(X[i][u] - (np.floor(X[i][u]/0.2))*0.2)
                indiceY = int(np.floor((Y[i][u]-0.01)/0.2))
                #print("indiceY", indiceY)
                scoreIndivICritUY = indivVals[u][indiceY] + individual[u][indiceY]*(Y[i][u] - (np.floor(Y[i][u]/0.2))*0.2)
                scoreiX.append(scoreIndivICritUX)
                scoreiY.append(scoreIndivICritUY)
            scoreX = np.sum(scoreiX)
            scoreY = np.sum(scoreiY)
            score.append(scoreX - scoreY >= 0)

        scoreTotalIndiv = np.sum(score)/len(X)

        return scoreTotalIndiv

    def selection(self, X, Y):
            
        # Evaluate individuals in the population and get their fitness scores
        fitness_scores = [self.evaluate_individual(individual, X, Y) for individual in self.population]

        # Sort individuals based on fitness scores in descending order
        sorted_population = [ind for _, ind in sorted(zip(fitness_scores, self.population), reverse=True)]

        # Ensure the number of selected individuals is even
        num_selected = len(sorted_population) // 2 * 2

        # Select the top half of individuals
        selected_population = sorted_population[:num_selected]
        print(max(fitness_scores))
        return selected_population



    def reproduce(self, selected_population):
        
        new_population = selected_population

        # Perform crossover and mutation to create new individuals
        for _ in range(len(selected_population)//2):
            # Select two parents for crossover
            parent1, parent2 = self.select_parents(selected_population)
            selected_population.remove(parent1)
            selected_population.remove(parent2)

            # Apply crossover to create two offspring
            offspring1, offspring2 = self.crossover(parent1, parent2)

            # Apply mutation to the offspring
            offspring1 = self.mutate(offspring1)
            offspring2 = self.mutate(offspring2)

            # Add offspring to the new population
            new_population.extend([offspring1, offspring2])

        return new_population

    def select_parents(self, selected_population):
        """Select two parents from the selected population."""
        # Ensure the selected_population has at least two individuals
        if len(selected_population) < 2:
            raise ValueError("Insufficient individuals in the selected population for parent selection.")

        # Randomly select two distinct parents
        parent1, parent2 = random.sample(selected_population, 2)

        return parent1, parent2

    def crossover(self, parent1, parent2):
        # Define the crossover point 
        crossover_point = 3
        offspring1 = []
        offspring2 = []
        # Create offspring with elements from parents
        for k in range(self.nb_criteres):

            offspring1.append(parent1[k][:crossover_point] + parent2[k][crossover_point:])
            offspring2.append(parent2[k][:crossover_point] + parent1[k][crossover_point:])

        return offspring1, offspring2

    def mutate(self, individual):
        
        random_index = random.randint(0, self.nb_criteres - 1)
        random_l = random.randint(0, self.L - 1)
        random_value = random.uniform(0, 5)
        ThereIsMUTATION = (random.randint(0, 100) == 8)
        if ThereIsMUTATION:
            individual[random_index][random_l] = random_value

        mutated_individual = individual
        return mutated_individual
    

    def predict_utility(self, X,Y):
        """Return Decision Function of the MIP for X. - To be completed.

        Parameters:
        -----------
        X: np.ndarray
            (n_samples, n_features) list of features of elements

        Returns
        -------
        np.ndarray:
            (n_samples, n_clusters) array of decision function value for each cluster.
        """

        return

class HeuristicModel(BaseModel):
    """Skeleton of MIP you have to write as the first exercise.
    You have to encapsulate your code within this class that will be called for evaluation.
    """

    def __init__(self, n_pieces, n_clusters,nb_criteres,nb_iterations= 1):
        """Initialization of the MIP Variables

        Parameters
        ----------
        n_pieces: int
            Number of pieces for the utility function of each feature.
        n_clusters: int
            Number of clusters to implement in the MIP.
        """
        self.iterations=nb_iterations
        self.iteration = 0
        self.n = nb_criteres
        self.seed = 123
        self.L = n_pieces
        self.K = n_clusters
        self.epsilon = 0.001
        self.model = self.instantiate()
        self.U = [0 for _ in range(n_clusters)]
        self.len_clusters = [0 for _ in range(n_clusters)]
        self.sigmaxPLUS = [0 for _ in range(n_clusters)]
        self.sigmayPLUS = [0 for _ in range(n_clusters)]
        self.sigmaxMINUS = [0 for _ in range(n_clusters)]
        self.sigmayMINUS = [0 for _ in range(n_clusters)]
        self.kmean = []

    def count_occurrences(self,clustering):
        # Utiliser Counter pour compter les occurrences de chaque valeur
        occurrences = Counter(clustering)
        
        # Convertir le résultat en une liste de tuples (valeur, nombre d'occurrences)
        occurrences_list = list(occurrences.items())
        
        # Trier la liste par valeurs pour obtenir une représentation ordonnée
        sorted_occurrences = sorted(occurrences_list, key=lambda x: x[0])
        
        # Diviser la liste en deux listes séparées (valeurs, nombre d'occurrences)
        values, counts = zip(*sorted_occurrences)
        
        return counts
    
    def clustering_Kmeans(self,X,Y):
        #array = np.hstack((X, Y))
        array =X-Y
        
        kmeans = KMeans(n_clusters=self.K, random_state=42)
        clustering = kmeans.fit_predict(array)
        self.len_clusters = self.count_occurrences(clustering)
        self.kmean = clustering
        return  kmeans.fit_predict(array)

    def LastIndex(self,x, i):
        return int(np.floor(self.L * (x + 0.01) / (1.01 + 0.01)))
    
    # Reference to the UTA University exercice 
    def BreakPoints(self,i, l):
                return  -0.01 + l * (1.01 - 0.01) / self.L
    
    
    def instantiate(self): 
        
        model = Model("heuristique")

        return model
 
    def fit(self, X, Y):
        """Estimation of the parameters - To be completed.

        Parameters
        ----------
        X: np.ndarray
            (n_samples, n_features) features of elements preferred to Y elements
        Y: np.ndarray
            (n_samples, n_features) features of unchosen elements
        """
        for iter in range(self.iterations):
            if self.iteration == 0:
                clusters = self.clustering_Kmeans(X,Y) 
            else:
                uixiJ = {} # uik_xij[i, j] = U_k(i, X[j, i])
                for i in range(self.n):
                    j=0
                    for cluster,xj in zip(clusters,X):
                            l = self.LastIndex(xj[i], i)
                            bp = self.BreakPoints(i, l)
                            bp1 = self.BreakPoints(i, l+1)
                            uixiJ[i, j] = self.U[cluster][(i, l)] + ((xj[i] - bp) / (bp1 - bp)) * (self.U[cluster][(i, l+1)] - self.U[cluster][(i, l)])
                            j+=1
                
                uiyiJ = {}
                for i in range(self.n):
                    j=0
                    for cluster,yj in zip(clusters,Y):
                            l = self.LastIndex(yj[i], i)
                            bp = self.BreakPoints(i, l)
                            bp1 = self.BreakPoints(i, l+1)
                            uiyiJ[i, j] = self.U[cluster][(i, l)] + ((yj[i] - bp) / (bp1 - bp)) * (self.U[cluster][(i, l+1)] - self.U[cluster][(i, l)])
                            j+=1
                u_xj = {}
                u_yj = {}
                
                for j in range(len((X[0]))):
                    u_xj[j] = quicksum(uixiJ[i, j] for i in range(self.n))
                for j in range(len((X[0]))):
                    u_yj[j] = quicksum(uiyiJ[i, j] for i in range(self.n))

                Ux = [u_xj[j] for j in range(len(X[0]))]
                Uy = [u_yj[j] for j in range(len(X[0]))]
                clusters = self.clustering_Kmeans(Ux,Uy)
                self.iteration +=1

            # We place ourselves in the cluster k
            for k in range(self.K):
                self.U[k] = {
                    (criteria, linearSegmentUTA): self.model.addVar(
                        vtype=GRB.CONTINUOUS, lb=0, name="u_{}_{}".format(criteria, linearSegmentUTA), ub=1)
                        for criteria in range(self.n)
                        for  linearSegmentUTA in range(self.L+1) # Need to add +1 because of the last segment {Uk and UK+1}
                }

                # Overestimation and underestimation variables σy+, σy− and σx+, σx−
                self.sigmaxPLUS[k] = {(jX): self.model.addVar(vtype=GRB.CONTINUOUS, lb=0, name="sigmaxp_{}".format(jX), ub=1) for jX in range(self.len_clusters[k])}
                self.sigmayPLUS[k] = {(jY): self.model.addVar(vtype=GRB.CONTINUOUS, lb=0, name="sigmayp_{}".format(jY), ub=1) for jY in range(self.len_clusters[k])}
                self.sigmaxMINUS[k] = {(jX): self.model.addVar(vtype=GRB.CONTINUOUS, lb=0, name="sigmaxm_{}".format(jX), ub=1) for jX in range(self.len_clusters[k])}
                self.sigmayMINUS[k] = {(jY): self.model.addVar(vtype=GRB.CONTINUOUS, lb=0, name="sigmaym_{}".format(jY), ub=1) for jY in range(self.len_clusters[k])}


                                     
                uixiJ = {} # uik_xij[i, j] = U_k(i, X[j, i])
                for i in range(self.n):
                    j=0
                    for cluster,xj in zip(clusters,X):
                        if cluster == k:
                            l = self.LastIndex(xj[i], i)
                            bp = self.BreakPoints(i, l)
                            bp1 = self.BreakPoints(i, l+1)
                            uixiJ[i, j] = self.U[k][(i, l)] + ((xj[i] - bp) / (bp1 - bp)) * (self.U[k][(i, l+1)] - self.U[k][(i, l)])
                            j+=1
                
                uiyiJ = {}
                for i in range(self.n):
                    j=0
                    for cluster,yj in zip(clusters,Y):
                        if cluster == k:
                            l = self.LastIndex(yj[i], i)
                            bp = self.BreakPoints(i, l)
                            bp1 = self.BreakPoints(i, l+1)
                            uiyiJ[i, j] = self.U[k][(i, l)] + ((yj[i] - bp) / (bp1 - bp)) * (self.U[k][(i, l+1)] - self.U[k][(i, l)])
                            j+=1
                u_xj = {}
                u_yj = {}
                
                for j in range(self.len_clusters[k]):
                    u_xj[j] = quicksum(uixiJ[i, j] for i in range(self.n))
                for j in range(self.len_clusters[k]):
                    u_yj[j] = quicksum(uiyiJ[i, j] for i in range(self.n))

                #########################
                ###### Constraints #######
                ##########################      

                for j in range(self.len_clusters[k]):
                    self.model.addConstr((u_xj[j] - self.sigmaxPLUS[k][j] + self.sigmaxMINUS[k][j] - u_yj[j] + self.sigmayPLUS[k][j] - self.sigmayMINUS[k][j] >=self.epsilon))

                
                ### MONOTOMIE:
                self.model.addConstrs((self.U[k][(i, l+1)] - self.U[k][(i, l)]>=self.epsilon for i in range(self.n) for l in range(self.L)))

                ## Constraint 6:
                # Normalisation of the utility function
                # ui(xi0) = 0
                self.model.addConstrs((self.U[k][(i, 0)] == 0 for i in range(self.n)))
                # Σu_i(xi) = 1
                self.model.addConstr((quicksum(self.U[k][(i, self.L)] for i in range(self.n)) == 1 ))
                

                ##########################
                ###### Objective #########
                ##########################
                self.model.setObjective(quicksum(self.sigmaxPLUS[k][j] + self.sigmaxMINUS[k][j] + self.sigmayPLUS[k][j] + self.sigmayMINUS[k][j] for j in range(self.len_clusters[k])), GRB.MINIMIZE)
                
                self.model.update()
                self.model.optimize()
                if self.model.status == GRB.INFEASIBLE:
                    raise Exception("Infeasible")
                elif self.model.status == GRB.UNBOUNDED:
                    raise Exception("Unbounded")
                else:
                    
                    print(f"objective function value in cluster{k}: ", self.model.objVal)
                    self.U[k] = {(i, l): self.U[k][i, l].x for i in range(self.n) for l in range(self.L+1)}
                    self.sigmaxMINUS[k] ={(j): self.sigmaxMINUS[k][j].x for j in range(self.len_clusters[k])}
                    self.sigmayMINUS[k] ={(j): self.sigmayMINUS[k][j].x for j in range(self.len_clusters[k])}
                    self.sigmaxPLUS[k] ={(j): self.sigmaxPLUS[k][j].x for j in range(self.len_clusters[k])}
                    self.sigmayPLUS[k] ={(j): self.sigmayPLUS[k][j].x for j in range(self.len_clusters[k])}

                if isinstance(self.U[2], dict):
                    for k in range(self.K):
                        if isinstance(self.U[k], dict):
                            plt.figure(figsize=(8, 4))
                            for i in range(self.n):
                                plt.plot([self.BreakPoints(i, l) for l in range(self.L+1)], [self.U[k][(i, l)] for l in range(self.L+1)], label=f'Cluster {k}, Feature {i}')
                            plt.legend()
                            plt.xlabel('Breakpoints')
                            plt.ylabel('Utility Value')
                            plt.title(f'Utility Functions for Cluster {k}')
                            plt.show()
                                    
        return self


    def predict_utility(self, X):
            """Return Decision Function of the MIP for X. - To be completed.

            Parameters:
            -----------
            X: np.ndarray
                (n_samples, n_features) list of features of elements
            """
            max_i = np.ones(self.n)*1.01
            min_i = np.ones(self.n)*-0.01

            def LastIndex(x, i):
                return int(np.floor(self.L * (x - min_i[i]) / (max_i[i] - min_i[i])))

            def BreakPoints(i, l):
                return min_i[i] + l * (max_i[i] - min_i[i])/self.L
            
            utilities = np.zeros((X.shape[0], self.K))
            for k in range(self.K):
                for j in range(X.shape[0]):
                    for i in range(self.n):
                        utilities[j, k] += self.U[k][i, LastIndex(X[j, i], i)] + ((X[j, i] - BreakPoints(i, LastIndex(X[j, i], i))) / (BreakPoints(i, LastIndex(X[j, i], i)+1) - BreakPoints(i, LastIndex(X[j, i], i)))) * (self.U[k][i, LastIndex(X[j, i], i)+1] - self.U[k][i, LastIndex(X[j, i], i)])

            return utilities