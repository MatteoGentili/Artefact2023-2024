import pickle
from abc import abstractmethod
from gurobipy import *

import numpy as np


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
        self.epsilon = 0.01
        self.model = self.instantiate()

    def instantiate(self, n_pieces, n_clusters): 
        
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
        # To be completed
        self.n = X.shape[1] # Number of features/criteria
        self.P = X.shape[0] # Number of elements
        max = np.ones(self.n)
        min = np.zeros(self.n)


        def LastIndex(x, i):
            return np.floor(self.L * (x - min[i]) / (max[i] - min[i]))
        
        # Reference to the UTA University exercice 
        def BreakPoints(i, l):
                    return min[i] + l * (max[i] - min[i]) / self.L
        
        # Varariables of the model
        self.U = {
            (cluster, criteria, linearSegmentUTA): self.model.addVar(
                vtype=GRB.CONTINUOUS, lb=0, name="u_{}_{}_{}".format(cluster, criteria, linearSegmentUTA), ub=1)
                for cluster in range(self.K)
                for criteria in range(self.n)
                for  linearSegmentUTA in range(self.L+1) # Need to add +1 because of the last segment {Uk and UK+1}
        }

        # Overestimation and underestimation variables
        self.sigmaxPLUS = {(jX): self.model.addVar(vtype=GRB.CONTINUOUS, lb=0, name="sigmaxp_{}".format(jX), ub=1) for jX in range(self.P)}
        self.sigmayPLUS = {(jY): self.model.addVar(vtype=GRB.CONTINUOUS, lb=0, name="sigmayp_{}".format(jY), ub=1) for jY in range(self.P)}
        self.sigmaxMINUS = {(jX): self.model.addVar(vtype=GRB.CONTINUOUS, lb=0, name="sigmaxm_{}".format(jX), ub=1) for jX in range(self.P)}
        self.sigmayMINUS = {(jY): self.model.addVar(vtype=GRB.CONTINUOUS, lb=0, name="sigmaym_{}".format(jY), ub=1) for jY in range(self.P)}

        # Delta allow us to which cluster the element belongs to
        self.delta1 = {(k, j): self.model.addVar(vtype=GRB.BINARY, name="delta1_{}_{}".format(k, j))for k in range(self.K)for j in range(self.P)}


        # Constraints
        ## Constraint 1: Preferences with delta variable : 
        M = 2.5 # Big M
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

        ## Constraint 2: 
        uk_xj = {}
        for k in range(self.K):
            for j in range(self.P):
                uk_xj[k, j] = quicksum(uiKxiJ[k, i, j] for i in range(self.n))
        
        uk_yj = {}
        for k in range(self.K):
            for j in range(self.P):
                uk_yj[k, j] = quicksum(uiKyiJ[k, i, j] for i in range(self.n))

        ## Constraint 3:
        self.model.addConstrs(
            (uk_xj[k, j] - self.sigmaxPLUS[j] + self.sigmaxMINUS[j] - uk_yj[k, j] + self.sigmayPLUS[j] - self.sigmayMINUS[j] - self.epsilon <= M*self.delta1[(k,j)] - self.epsilon for j in range(self.P) for k in range(self.K))
        )

        ## Constraint 4:
        # uk(x) > uk(y) + ϵ ⇐⇒ x ≽k y ==> x is preferred to y in cluster k
        # => uk(x) - uk(y) + ϵ >= 0
        self.model.addConstrs((self.U[(k, i, l+1)] - self.U[(k, i, l)]>=self.epsilon for k in range(self.K) for i in range(self.n) for l in range(self.L)))

        ## Constraint 5:
        ### MONOTONICITY:
        self.model.addConstrs((self.U[(k, i, l+1)] - self.U[(k, i, l)]>=self.epsilon for k in range(self.K) for i in range(self.n) for l in range(self.L)))

        # Objective
        self.model.setObjective(quicksum(self.sigmaxPLUS[j] + self.sigmaxMINUS[j] + self.sigmayPLUS[j] + self.sigmayMINUS[j] for j in range(self.P)), GRB.MINIMIZE)

        return

    def predict_utility(self, X):
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
        # To be completed
        return



class HeuristicModel(BaseModel):
    """Skeleton of MIP you have to write as the first exercise.
    You have to encapsulate your code within this class that will be called for evaluation.
    """

    def __init__(self):
        """Initialization of the Heuristic Model.
        """
        self.seed = 123
        self.models = self.instantiate()

    def instantiate(self):
        """Instantiation of the MIP Variables"""
        # To be completed
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
        # To be completed
        return

    def predict_utility(self, X):
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
        # To be completed
        # Do not forget that this method is called in predict_preference (line 42) and therefor should return well-organized data for it to work.
        return
