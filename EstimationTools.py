from matplotlib.pyplot import axis
import numpy as np
from numpy.linalg import inv
import statsmodels.api as sm
import statsmodels.formula.api as smf
import HWDataProcessing as dp
from scipy.optimize import fmin_bfgs

class ModelSpecification():
    pass

class Model:
    def __init__(self,spec):

        # Unpacking the model spepcification
        self.type   = spec.type # type = logit, nested_logit, mixed_logit, blp
        self.data   = spec.data # data
        self.i      = spec.i # consumer identifier
        self.J      = spec.J # number of products
        self.data["constant"] = 1
        self.yName   = spec.y # choice (binary)
        self.y2Name  = spec.y2 # second choice (binary)
        self.xName  = spec.x # product characterists
        self.pName  = spec.p # price
        self.iv     = spec.iv # iv for price
        self.zName  = spec.zeta # consumer attributes
        self.brands = spec.brands        
        self.n = len(data)
        self.nu = spec.nu # unobserved consumer attributes
        self.unobsHet = sum(self.nu) > 0 # whether to include unobserved heterogeneity
        self.ns = spec.ns # number of draws for MC integration
        self.secondChoice = spec.secondChoice # whether to use second choice moments
        self.seed = spec.seed
        self.XZetaInter = spec.XZetaInter
        self.X1X2Inter =  spec.X1X2Inter

    
    
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
        """Compute elasticity in the Logit model. This command has to be runned 
        after fitting the model."""

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
    
    def fitMixedLogit(self):
        """Estimate a mixed logit model via GMM"""

        # Specification
        unobsHet = self.unobsHet # whether to model unboserved heterogeneity
        nuPosition = (np.array(self.nu) == 1)
        secondChoice = self.secondChoice # whether to use second-choice moments
        xzPairs = self.XZetaInter  # interactions for first-choice moments
        x1x2Charac = self.X1X2Inter  # interactions for second-choice moments

        # share of consumers that chose product j 
        jChosenCount = self.data[[self.yName,"product"]]\
                        .groupby(by="product").sum().to_numpy()
        jChosenShare = jChosenCount/sum(jChosenCount)


        # Define  matrices X, Z , Zeta, Y
        regressors =  self.xName + [self.pName]
        instruments = self.xName + self.iv
        consumerAttr = self.zName

        X = self.data[regressors].to_numpy()    # product characteristics
        Z = self.data[instruments].to_numpy()   # instruments
        Zeta = self.data[consumerAttr].to_numpy() # observed consumer attributes
        Y = np.expand_dims(self.data[self.yName], axis=1) # first choice


        # restrict data to chosen products as first choice
        dataChosen = self.data[self.data[self.yName] == 1]

        # restrict data to chosen products and info about second choice
        dataSecondChoice = self.data[(~ self.data[self.y2Name].isnull()) &
                           ((self.data[self.yName]==1) |
                           (self.data[self.y2Name]==1))]
        

        # MC sample

        
        # sizes of sample, regressors and moment conditions
        n, k = X.shape # n = observations; k = number of product characteristics
        l = Z.shape[1] # number of instruments for demand equation
        j = self.J  # number of products
        ro = Zeta.shape[1] # number of observed consumer attributes
        ru = sum(self.nu)   # number of unobserved consumer attributes
        nXZpairs = len(xzPairs) # number of first-choice moments
        nX1X2pairs = len(x1x2Charac) # number of first-choice moments
        nParameters = k + k*ro + ru
        nMoment1 = l
        nMoment2 = nXZpairs
        nMoment3 = nX1X2pairs
        nMoments = nMoment1 + nMoment2 + nMoment3 # total number of moments

        # If we have unobserved heterogeneity
        if unobsHet:
            # TBD: draw from a given dist
            pass
        
        # If we use second choice data
        if secondChoice:
            # TBD: get Y2
            pass    
        
        # add X*Zeta to datasets
        XZetaNames = ["XZeta" + str(s)  for s in range(nXZpairs)]
        for i in range(nXZpairs):
            varName = XZetaNames[i]
            Xvar = regressors[xzPairs[i][0]]
            Zetavar = regressors[xzPairs[i][1]]
            dataChosen[varName] = dataChosen[Xvar]*dataChosen[Zetavar]

        # add X1*X2 to datasets
        X1X2Names = ["X1X2_" + str(s)  for s in range(nX1X2pairs)]
        # split dataset into first and second-choice product
        firstProductData  = dataSecondChoice[dataSecondChoice[self.yName]==1]
        secondProductData = dataSecondChoice[dataSecondChoice[self.y2Name]==1]
        # add interaction terms 
        dataInteractionsX1X2 = firstProductData.copy()
        for i in range(nX1X2pairs):
            varName = X1X2Names[i]
            Xvar = regressors[x1x2Charac[i]]
            dataInteractionsX1X2[varName] = firstProductData[Xvar]*\
                                            secondProductData[Xvar]


        # Compute sample version of each set of moments

        # Moment 1: exclusion restriction Cov(Z*Y)
        # number of  moments = l (number of instruments)
        sampleG1 = (Z*Y).mean(axis=0).T

        # Moment 2: first choice moments: Cov(X,zeta)
        # number of momentes = len(interactions between X and Zeta)
        meanXZetaByProduct = dataChosen[XZetaNames+["product"]]\
                              .groupby(by="product").mean().to_numpy()
        sampleG2 = (jChosenShare*meanXZetaByProduct).mean(axis=0).T

        # Moment 3: first= and second-choice moments: Cov(X^1,X^2)
        # number of moments = len(characteristics to compare)
        meanX1X2ByProduct = dataInteractionsX1X2[X1X2Names+["product"]]\
                              .groupby(by="product").mean().to_numpy()
        sampleG3 = (jChosenShare*meanX1X2ByProduct).mean(axis=0).T

        
        # Initial parameter and Weight matrix
        theta0 = np.zeros([nParameters,1])
        W = np.identity(nMoments)

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
    """Return the fitted value of Y in the logit model."""
    
    n,k = X.shape
    U = np.expand_dims(X @ theta,axis=1).reshape((J,n//J),order='F')
    Ypred = softmax(U).reshape((n,1),order='F')
    return Ypred

def MixedLogitGmmObj(theta,X,Y,Z,Zeta):
    """Compute GMM obj. function, J, given data (X,Y,Z) and parameters theta.
    Mixed Logit model with individual level data."""

    n,k = X.shape

    # first set of moments, G1: predecited choice probability
    U = np.expand_dims(X @ theta,axis=1).reshape((J,n//J),order='F')
    Ypred = softmax(U).reshape((n,1),order='F')
    M = Z*(Y-Ypred)        # moments
    G3 = M.mean(axis=0).T   # avg. moments
    
def MixedLogitMoment1(theta,X,Z,Zeta,nu,J,k,ro,nuPosition,jChosenShare):
    
    # shape
    n,k = X.shape
    # unpack theta 
    betaBar, betaO, betaU = unpackParameters(theta,k,ro,nuPosition)

    # random coefficients
    beta = np.ones((n,k))*betaBar.T + Zeta @ (betaO.T) +\
           nu*(np.ones((n,k))*betaU.T)
    
    # utility. each row is a product, each column is a consumer
    U = np.expand_dims(X*beta,axis=1).sum(axis=1).reshape((J,n//J),order='F')
    Ypred = (softmax(U)*jChosenShare).reshape((n,1),order='F')
    M1 = Z*Ypred        # moments
    G1 = M1.mean(axis=0).T   # avg. moments

    return G1, M1

def MixedLogitMoment2(X,Zeta,nu,):
    


def unpackParameters(theta,k,ro,nuPosition):
    betaBar = theta[:k,:].reshape((k,1))
    betaO = theta[k:(k+k*ro),:].reshape((k,ro),order='F')
    betaUp = theta[(k+k*ro):,:]
    betaU = np.zeros((k,1))
    betaU[nuPosition,:] = betaUp
    
    return betaBar, betaO, betaU


     