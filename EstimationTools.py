from asyncio.windows_events import NULL
from audioop import mul
from matplotlib.pyplot import axis
import numpy as np
import pandas as pd
from numpy.linalg import inv
import statsmodels.api as sm
import statsmodels.formula.api as smf
import HWDataProcessing as dp
from scipy.optimize import fmin_bfgs

class ModelSpecification():
    pass

class Model:
    def __init__(self,spec):
        """Class constructor. Creates a new model.
        Currently implemented: logit and mixed logit."""

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
        self.n = len(self.data)
        self.nu = spec.nu # unobserved consumer attributes
        self.unobsHet = sum(self.nu) > 0 # whether to include unobserved heterogeneity
        self.ns = spec.ns # number of draws for MC integration
        self.secondChoice = spec.secondChoice # whether to use second choice moments
        self.seed = spec.seed  # seed for random number generation
        self.XZetaInter = spec.XZetaInter  # first-choice interactions
        self.X1X2Inter =  spec.X1X2Inter   # first- and second-choice interaction

        # simulate data for MC integration
        if self.type != 'logit':
            k = len(self.xName) + 1
            self.unobsNames = ["nu" + str(s)  for s in range(k)]
            self.MCSample = dp.genMCSample(self.data,self.ns,self.i,self.unobsNames,
                                self.seed,self.J)
        else:
            self.unobsNames = []
            self.MCSample = []

    def fit(self):
        """Estimate the model"""

        modelType = self.type
        match modelType:
            case 'logit':
                self.estimates, self.se = self.fitLogit()
            case 'mixed_logit':
                self.estimates, self.se = self.fitMixedLogit()
        
    def reportEstimates(self):
        """Report parameter estimates."""

        modelType = self.type
        match modelType:
            case 'logit':
                estimates = pd.DataFrame()
                estimates["var. name"] = self.xName + [self.pName]
                estimates["coefficient"] = self.estimates
                estimates["s.e."] = self.se
            case 'mixed_logit':
                
                regressors =  self.xName + [self.pName]
                consumerAttr = self.zName
                unobsConsAttr = self.unobsNames
                nuPosition = self.nu
                
            
                k = len(regressors)
                ro = len(consumerAttr)
                ru = sum(self.nu)
                nParameters = k + k*ro + ru

                XZNames = []
                for i in range(ro):
                    for j in range(k):
                        thisVar = regressors[j] + '_' + consumerAttr[i]
                        XZNames = XZNames + [thisVar]
                
                unobsNames = []
                for i in range(k):
                    if nuPosition[i] == 1:
                        unobsNames = unobsNames + [regressors[i]]

                paramName = ["betaBar"]*k + ["betaO"]*(k*ro) + ["betaU"]*ru
                varName = regressors + XZNames + unobsNames

                estimates = pd.DataFrame()
                estimates["coeficient"] = paramName
                estimates["var. name"] = varName
                estimates["coefficient"] = self.estimates
                estimates["s.e."] = self.se

        return estimates
                

    
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
        unobsConsAttr = self.unobsNames

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

        # Sample for MC integration
        MCSample = self.MCSample
        
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
         
        
        # add X*Zeta to datasets
        XZetaNames = ["XZeta_" + str(s)  for s in range(nXZpairs)]
        for i in range(nXZpairs):
            varName = XZetaNames[i]
            Xvar = regressors[xzPairs[i][0]]
            Zetavar = regressors[xzPairs[i][1]]
            dataChosen[varName] = dataChosen[Xvar]*dataChosen[Zetavar]
            MCSample[varName] = MCSample[Xvar]*MCSample[Zetavar]

        # If we use second choice data, add X1X2 to the sample
        if secondChoice:             
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
                MCSample[varName] = MCSample[Xvar]

        
        Xr = MCSample[regressors].to_numpy()    # product characteristics
        Zr = MCSample[instruments].to_numpy()   # instruments
        Zetar = MCSample[consumerAttr].to_numpy() # observed consumer attributes
        Nur = MCSample[unobsConsAttr].to_numpy() # unobserved consumer attributes
        XZetar = MCSample[XZetaNames].to_numpy() # interactions X*Zeta
        X12r = MCSample[X1X2Names].to_numpy()    # characteristics to interact 
                                                 # between first and second choice


        # Compute sample version of each set of moments

        # Moment 1: exclusion restriction Cov(Z*Y)
        # number of  moments = l (number of instruments)
        sampleG1 = (Z*Y).mean(axis=0)
        sampleG1 = np.expand_dims(sampleG1,axis=1)

        # Moment 2: first choice moments: Cov(X,zeta)
        # number of momentes = len(interactions between X and Zeta)
        meanXZetaByProduct = dataChosen[XZetaNames+["product"]]\
                              .groupby(by="product").mean().to_numpy()
        sampleG2 = (jChosenShare*meanXZetaByProduct).mean(axis=0)
        sampleG2 = np.expand_dims(sampleG2,axis=1)

        # Moment 3: first= and second-choice moments: Cov(X^1,X^2)
        # number of moments = len(characteristics to compare)
        if secondChoice:             
            meanX1X2ByProduct = dataInteractionsX1X2[X1X2Names+["product"]]\
                                .groupby(by="product").mean().to_numpy()
            sampleG3 = (jChosenShare*meanX1X2ByProduct).mean(axis=0).T
            sampleG3 = np.expand_dims(sampleG3,axis=1)
            sampleG = np.vstack((sampleG1,sampleG2,sampleG3))
        else:
            sampleG3 = []
            sampleG = np.vstack((sampleG1,sampleG2))

        
        # Initial parameter and Weight matrix
        theta0 = np.zeros([nParameters,1])
        W = np.identity(nMoments)

        # initialize iterations
        global iteration, lastvalue, functionCount
        iteration = 0
        lastValue = 0
        functionCount = 0

        # Minimization
        # first step
        out = False
        args = (Xr, Zr,Zetar,XZetar,X12r,Nur,j,k,ro,nuPosition,jChosenShare,
                sampleG,W,out)
        step1opt = fmin_bfgs(MixedLogitGmmObj, theta0, args=args,
                              callback=iter_print)

        return(step1opt,step1opt*0.5)

        # # second step
        # out = LogitGmmObjective(step1opt, X,Y,Z,W,j,out=True)
        # Omega = np.cov(out[1].T)
        # W2 = inv(Omega)
        # args = (X, Y, Z, W2, j)
        # step2opt = fmin_bfgs(LogitGmmObjective, step1opt, args=args)

        # # Estimate VCV of parameter estimates
        # out = LogitGmmObjective(step2opt, X,Y,Z,W2,j,out=True)
        # G = LogitGmmG(step2opt,X,Y,Z,j)
        # S = np.cov(out[1].T)
        # vcv = inv(G @ inv(S) @ G.T)/n

        # return(step2opt,vcv)


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
    
def MixedLogitGmmObj(theta,X,Z,Zeta,XZeta,X12,nu,
                      J,k,ro,nuPosition,jChosenShare,
                      sampleG, W, out):
    """Compute GMM obj. function, given data (X,Y,Z,Zeta,Nu) and parameters theta.
    Mixed Logit model with individual level data."""
    
    global lastValue, functionCount # used in message printing

    # shape
    n,k = X.shape
    ni = n//J # number of consumers
    nXZeta = XZeta.shape[1]
    
    # unpack theta 
    betaBar, betaO, betaU = unpackParameters(theta,k,ro,nuPosition)

    # random coefficients
    beta = np.ones((n,k))*betaBar.T + Zeta @ (betaO.T) +\
           nu*(np.ones((n,k))*betaU.T)
    
    # utility. each row is a product, each column is a consumer
    U = (X*beta).sum(axis=1).reshape((J,ni),order='F')
    Ypred = softmax(U).reshape((n,1),order='F')

    # moments set 1
    M1 = Z*Ypred  # moments
    # mean over j and N
    G1 = M1.mean(axis=0)
    G1 = np.expand_dims(G1,axis=1) # make it a vector

    # moments set 2
    M2 = XZeta*Ypred    # moments
    # weighted mean over j
    G2 =  (M2.reshape((J,ni,nXZeta),order='F').mean(axis=1)/
            Ypred.reshape((J,ni,1),order='F').mean(axis=1)*
            jChosenShare).mean(axis=0)    # avg. moments
    G2 = np.expand_dims(G2,axis=1) # make it a vector
    
    # moments 3
    k2 = X12.shape[1]
    X2Prob2 = secondChoiceXP(X12,U,n,J,k2)
    M3 = (X12*X2Prob2)*Ypred    # moments
    # weighted mean over j
    G3 =  (M3.reshape((J,ni,k2),order='F').mean(axis=1)/
            np.expand_dims(Ypred.reshape((J,ni),order='F').mean(axis=1),axis=1)*
            jChosenShare).mean(axis=0)   # avg. moments
    G3 = np.expand_dims(G3,axis=1) # make it a vector

    # stack moments
    G = np.vstack((G1,G2,G3))

    # compute difference between sample and model moments
    Gd = sampleG - G

    # Compute objective (J function)
    Obj = ni * (Gd.T @ W @ Gd)
    

    lastValue = Obj
    functionCount += 1

    if not out:
        return Obj
    else:
        return Obj, M1 # TBD 

def secondChoiceXP(X,U,n,J,k):

    ni = n//J # number of consumers

    # reshape: product x consumer
    X = X.reshape((J,ni,k),order='F')

    # expand vertically by J
    U = np.kron(np.ones((J,1)),U)
    X = np.kron(np.ones((J,1,1)),X)

    # remove jth row (1st choice)
    index =  np.identity(J)
    index = (index == 0).reshape((J**2,),order='F')
    U = U[index,:]
    X  =X[index,:,:]

    # softmax, multiply by X and sum
    Prob = softmax(U)
    Prob = np.kron(np.ones((1,1,k)),np.expand_dims(Prob,axis=2)) 
    XProb = (X*Prob).reshape((J-1,J,ni,k),order='F').sum(axis=0)\
                    .reshape((n,k),order='F')

    return XProb    


def MixedLogitMoment2(X,Zeta,nu,J,k,ro,nuPosition,jChosenShare):
    
    # shape
    n,k = X.shape

    # unpack theta 
    betaBar, betaO, betaU = unpackParameters(theta,k,ro,nuPosition)

    # random coefficients
    beta = np.ones((n,k))*betaBar.T + Zeta @ (betaO.T) +\
           nu*(np.ones((n,k))*betaU.T)

    

def unpackParameters(theta,k,ro,nuPosition):
    betaBar = theta[:k].reshape((k,1))
    betaO = theta[k:(k+k*ro)].reshape((k,ro),order='F')
    betaUp = theta[(k+k*ro):]
    betaU = np.zeros((k,1))
    betaU[nuPosition,:] = np.expand_dims(betaUp,axis=1)
    
    return betaBar, betaO, betaU

def iter_print(params):
    global iteration, lastValue, functionCount
    iteration += 1
    print('Func value: {0:}, Iteration: {1:}, \
           Function Count: {2:}'.format(lastValue, 
                                         iteration, functionCount))
     