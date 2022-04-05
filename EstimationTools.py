from matplotlib.pyplot import axis
import numpy as np
from numpy.linalg import inv
import statsmodels.api as sm
import statsmodels.formula.api as smf
import HWDataProcessing as dp
from scipy.optimize import fmin_bfgs

class Model:
    def __init__(self,type,data,J,y,x,z,p,brands,iv):
        self.type = type # type = logit, nested_logit, mixed_logit, blp
        self.data = data # data
        self.data["constant"] = 1
        self.yName = y # choice (binary)
        self.xName = x # product characterists
        self.zName = z # consumer attributes
        self.pName = p # price
        self.J     = J # number of products
        #self.marketName = market
        self.brands = brands
        self.iv = iv
        self.n = len(data)
    
    
    def fitLogit(self):
        """Estimate a logit model via GMM"""

        # Define  matrices
        #regressors = ["constant"] + self.xName + self.pName
        #instruments = ["constant"] + self.xName + self.iv
        regressors =  self.xName + [self.pName]
        instruments = self.xName + self.iv
        X = self.data[regressors].to_numpy()
        Z = self.data[instruments].to_numpy()
        Y = np.expand_dims(self.data[self.yName], axis=1)
        
        # sample, regressors and moment conditions
        n = X.shape[0]
        k = X.shape[1]
        l = Z.shape[1]
        j = self.J 

        # Initial parameter and Weight matrix
        theta0 = np.zeros([k,1])
        W = np.identity(l)

        # Minimization
        # first step
        args = (X, Y, Z, W, j)
        step1opt = fmin_bfgs(LogitGmmObjective, theta0, args=args)

        # second step
        out = LogitGmmObjective(step1opt, X,Y,Z,W,j,out=True)
        Omega = np.cov(out[1].T)
        W2 = inv(Omega)
        args = (X, Y, Z, W2, j)
        step2opt = fmin_bfgs(LogitGmmObjective, step1opt, args=args)

        # Estimate VCV of parameter estimates
        out = LogitGmmObjective(step2opt, X,Y,Z,W2,j,out=True)
        G = LogitGmmG(step2opt,X,Y,Z,j)
        S = np.cov(out[1].T)
        vcv = inv(G @ inv(S) @ G.T)/n

        return(step2opt,vcv)

    def getElasticityLogit(self,theta):
        # Define  matrices
        price =  self.pName
        regressors =  self.xName + [self.pName]
        X = self.data[regressors].to_numpy()
        Y = np.expand_dims(self.data[self.yName], axis=1)
        p = np.expand_dims(self.data[price], axis = 1)
        J = self.J 
        n,k = X.shape

        alpha = theta[-1]
        Ypred = LogitGetFittedProb(theta,X,Y,J)
        ownPE = (alpha*p*(1-Ypred)).reshape((J,n//J),order='F').mean(axis=1)
        crossPE = (-alpha*p*Ypred).reshape((J,n//J),order='F').mean(axis=1)

        elasticities = np.kron(np.ones((J,1)),crossPE.T)

        for j in range(J):
            elasticities[j,j] = ownPE[j]
        
        return elasticities

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def LogitGmmObjective(theta,X,Y,Z,W,J,out=False):
    """Compute GMM obj. function, J, given data (X,Y,Z) and parameters theta.
    Logit model with individual level data."""

    n,k = X.shape
    U = np.expand_dims(X @ theta,axis=1).reshape((J,n//J),order='F')
    Ypred = softmax(U).reshape((n,1),order='F')
    M = Z*(Y-Ypred)        # moments
    G = M.mean(axis=0).T   # avg. moments 
    J = n * (G.T @ W @ G)  # GMM objective
    if not out:
        return J
    else:
        return J, M

def LogitGmmG(theta,X,Y,Z,J):
    """Compute GMM G matrix, the derivative of the moments function,
    given data (X,Y,Z) and parameters theta.
    Logit model with individual level data."""

    n,k = X.shape
    n,l = Z.shape
    U = np.expand_dims(X @ theta,axis=1).reshape((J,n//J),order='F')
    Ypred = softmax(U).reshape((n,1),order='F')

    Xi = X.reshape((J,int(n/J*k)),order='F') # each column is consumerXcharacteristic
    Xj = np.kron(Xi,np.ones((J,1))) # product 1 J times, then product 2 J times, and so on
    Xq = np.kron(np.ones((J,1)),Xi) # products 1 to J, repeat J times
    Wq = np.kron(np.ones((J,k)), softmax(U)) # weights
    Rq = (Xj-Xq)*Wq # Result times weight. Now need to sum every J entries
    Ri = Rq.reshape(-1,J,Rq.shape[-1]).sum(1) # weighted avg. of Xj-Xq
    R = Ri.reshape((n,k),order='F')
    G = -1/n * (Z*Ypred).T @ R 
    
    return G

def LogitGetFittedProb(theta,X,Y,J):
    n,k = X.shape
    U = np.expand_dims(X @ theta,axis=1).reshape((J,n//J),order='F')
    Ypred = softmax(U).reshape((n,1),order='F')
    return Ypred