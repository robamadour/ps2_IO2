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
        self.zName  = spec.zeta # consumer attributes
        self.brands = spec.brands        
        self.n = len(self.data)
        self.nu = spec.nu # unobserved consumer attributes
        self.unobsHet = sum(self.nu) > 0 # whether to include unobserved heterogeneity
        self.ns = spec.ns # number of draws for MC integration
        self.nr = spec.nr # number of resamplings needed to compute variance
        self.secondChoice = spec.secondChoice # whether to use second choice moments
        self.seed = spec.seed  # seed for random number generation
        self.XZetaInter = spec.XZetaInter  # first-choice interactions
        self.X1X2Inter =  spec.X1X2Inter   # first- and second-choice interaction

        # simulate data for MC integration
        if self.type != 'logit':
            k = len(self.xName) + 1
            self.unobsNames = ["nu" + str(s)  for s in range(k)]
            self.MCSample = dp.genMCSample(self.data,self.ns*self.nr,self.i,
                    self.unobsNames, self.seed,self.J)
            if len(self.XZetaInter) == 0:
                self.useM2 = False
            else:
                self.useM2 = True
        else:
            self.unobsNames = []
            self.MCSample = []
            self.useM2  = False

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
                if len(consumerAttr):
                    for i in range(ro):
                        for j in range(k):
                            thisVar = regressors[j] + '_' + consumerAttr[i]
                            XZNames = XZNames + [thisVar]
                
                unobsNames = []
                if len(nuPosition) > 0:
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
        X = self.data[regressors].to_numpy()
        Y = np.expand_dims(self.data[self.yName], axis=1)
        
        # sample, regressors and moment conditions
        n = X.shape[0]
        k = X.shape[1]
        j = self.J 

        # Initial parameter and Weight matrix
        theta0 = np.zeros([k,1])
        W = np.identity(k)

        # Minimization
        # first step
        args = (X, Y, W, j)
        step1opt = fmin_bfgs(LogitGmmObjective, theta0, args=args)

        # second step
        out = LogitGmmObjective(step1opt, X,Y,W,j,out=True)
        Omega = np.cov(out[1].T)
        W2 = inv(Omega)
        args = (X, Y, W2, j)
        step2opt = fmin_bfgs(LogitGmmObjective, step1opt, args=args)

        # Estimate VCV of parameter estimates
        out = LogitGmmObjective(step2opt, X,Y,W2,j,out=True)
        G = LogitGmmG(step2opt,X,Y,j)
        S = np.cov(out[1].T)
        vcv = inv(G @ inv(S) @ G.T)/n
        se = np.sqrt(np.diag(vcv))

        return(step2opt,se)

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
        data = self.data.copy()
        unobsHet = self.unobsHet # whether to model unboserved heterogeneity
        nuPosition = (np.array(self.nu) == 1)
        useM2 = self.useM2
        secondChoice = self.secondChoice # whether to use second-choice moments
        xzPairs = self.XZetaInter  # interactions for first-choice moments
        x1x2Charac = self.X1X2Inter  # interactions for second-choice moments

        # share of consumers that chose product j 
        jChosenCount = self.data[[self.yName,"product"]]\
                        .groupby(by="product").sum().to_numpy()
        jChosenShare = jChosenCount/sum(jChosenCount)
        
        # Define  matrices X, Z , Zeta, Y
        regressors =  self.xName + [self.pName]
        consumerAttr = self.zName
        unobsConsAttr = self.unobsNames

        X = self.data[regressors].to_numpy()    # product characteristics
        Zeta = self.data[consumerAttr].to_numpy() # observed consumer attributes
        Y = np.expand_dims(self.data[self.yName], axis=1) # first choice


        # restrict data to chosen products as first choice
        dataChosen = self.data[self.data[self.yName] == 1]

        # restrict data to chosen products and info about second choice
        dataSecondChoice = self.data[(~ self.data[self.y2Name].isnull()) &
                           ((self.data[self.yName]==1) |
                           (self.data[self.y2Name]==1))]
        
        # consumers for whom there is second choice data
        SecondData = self.data[(~ self.data[self.y2Name].isnull())]
        consumerSecondData = SecondData[self.i]
        Y1SecondData = np.expand_dims(SecondData[self.yName].to_numpy(),axis=1)
        Y2SecondData = np.expand_dims(SecondData[self.y2Name].to_numpy(),axis=1)
        nConsSecondData = len(consumerSecondData)//self.J  
        indexConsumersSecondChoice = ~ dataChosen[self.y2Name].isnull()

        # Sample for MC integration
        MCSample = self.MCSample
        
        # sizes of sample, regressors and moment conditions
        n, k = X.shape # n = observations; k = number of product characteristics
        j = self.J  # number of products
        nConsumers = n//j # number of consumers
        ro = Zeta.shape[1] # number of observed consumer attributes
        ru = sum(self.nu)   # number of unobserved consumer attributes
        nXZpairs = len(xzPairs) # number of first-choice moments
        nX1X2pairs = len(x1x2Charac) # number of first-choice moments
        nParameters = k + k*ro + ru
        nMoment1 = k
        nMoment2 = nXZpairs
        nMoment3 = nX1X2pairs
        nMoments = nMoment1 + nMoment2 + nMoment3 # total number of moments
        ns = self.ns
        nr = self.nr 
        
        # add X*Zeta to datasets
        XZetaNames = ["XZeta_" + str(s)  for s in range(nXZpairs)]
        for i in range(nXZpairs):
            varName = XZetaNames[i]
            Xvar = regressors[xzPairs[i][0]]
            Zetavar = regressors[xzPairs[i][1]]
            data[varName] = data[Xvar]*data[Zetavar]
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
                dataInteractionsX1X2[varName] = firstProductData[Xvar].to_numpy()*\
                                                secondProductData[Xvar].to_numpy()
                MCSample[varName] = MCSample[Xvar]
                dX1X2 = np.expand_dims(dataInteractionsX1X2[varName].to_numpy(),
                                        axis=1)
                SecondData[varName] = np.kron(dX1X2,np.ones((j,1)))
            X12rAll = MCSample[X1X2Names].to_numpy()    # characteristics to interact 
                                                 # between first and second choice
            X1X2sample = SecondData[X1X2Names].to_numpy() 
        else:
            X12rAll = []
            X1X2sample = []

        # Simulated variables
        XrAll = MCSample[regressors].to_numpy()    # product characteristics
        ZetarAll = MCSample[consumerAttr].to_numpy() # observed consumer attributes
        NurAll = MCSample[unobsConsAttr].to_numpy() # unobserved consumer attributes
        XZetarAll = MCSample[XZetaNames].to_numpy() # interactions X*Zeta
        
        # Just the first sample (we will use the rest to compute the variance)
        Xr     = XrAll[:ns*j,:]
        Zetar  = ZetarAll[:ns*j,:]
        Nur    = NurAll[:ns*j,:]
        XZetar = XZetarAll[:ns*j,:]
        if len(X12rAll)>0:
            X12r   = X12rAll[:ns*j,:]
        else:
            X12r = []

        # Compute sample version of each set of moments

        # Moment 1: exclusion restriction Cov(X*Y)
        # number of  moments = l (number of covariates)
        sampleM1 = ((X*Y).reshape((j,nConsumers,k),order='F')).\
                    mean(axis=0)
        sampleG1 = np.expand_dims(sampleM1.mean(axis=0),axis=1)

        # Moment 2: first choice moments: Cov(X,zeta)
        # number of momentes = len(interactions between X and Zeta)
        if useM2:
            XZeta  = data[XZetaNames].to_numpy()
            sampleM2 = (XZeta*Y).reshape((j,nConsumers,nXZpairs),order='F').\
                        sum(axis=0)
            sampleG2 = np.expand_dims(sampleM2.mean(axis=0),axis=1)

        else:
            sampleM2 = []
            sampleG2 = []

        # Moment 3: first= and second-choice moments: Cov(X^1,X^2)
        # number of moments = len(characteristics to compare)
        if secondChoice:

            sampleM3 = (X1X2sample*Y1SecondData*Y2SecondData).\
                        reshape((j,nConsSecondData,nX1X2pairs),order='F').\
                        sum(axis=0)
            sampleG3 = np.expand_dims(sampleM3.mean(axis=0),axis=1)

        else:
            sampleM3 = []
            sampleG3 = []
        
        # stack moments
        if (secondChoice and useM2):
            sampleG = np.vstack((sampleG1,sampleG2,sampleG3))
        elif secondChoice:
            sampleG = np.vstack((sampleG1,sampleG3))
        elif useM2:
            sampleG = np.vstack((sampleG1,sampleG2))
        else:
            sampleG = sampleG1

        # Compute sample component of variance-covariance matrix
        sampleVar = computeSampleVariance(sampleM1,sampleM2,sampleM3,
                            useM2,secondChoice, indexConsumersSecondChoice)
        
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
        args = (Xr, Zetar,XZetar,X12r,Nur,j,k,ro,nuPosition,jChosenShare,
                sampleG,useM2,secondChoice,W,out)
        step1opt = fmin_bfgs(MixedLogitGmmObj, theta0, args=args,
                              callback=iter_print)

        # Compute variance-covariance matrix
        # simulation variance
        simulationVar = computeSimulationVariance(step1opt,j,k,ro,nuPosition,jChosenShare,nMoments,ns,nr,
                            XrAll,ZetarAll,NurAll,XZetarAll,X12rAll,
                            sampleG,useM2,secondChoice,W)

        # compute total variance
        moment_variance = sampleVar + simulationVar

        # set W = inv(variance)
        W2 = inv(moment_variance)

        # second step
        args = (Xr, Zetar,XZetar,X12r,Nur,j,k,ro,nuPosition,jChosenShare,
                sampleG,useM2,secondChoice,W2,out)
        step2opt = fmin_bfgs(MixedLogitGmmObj, step1opt, args=args,
                              callback=iter_print)
        
        
        # compute s.e.
        # Get numerical derivative
        d = 1e-6 
        D = computeDerivativeMomentMixedLogit(d,step2opt,Xr,Zetar,XZetar,X12r,Nur,
                      j,k,ro,nuPosition,jChosenShare,
                      sampleG, useM2,secondChoice,W2)
        
        simulationVar = computeSimulationVariance(step2opt,j,k,ro,nuPosition,
                            jChosenShare,nMoments,ns,nr,
                            XrAll,ZetarAll,NurAll,XZetarAll,X12rAll,
                            sampleG,useM2,secondChoice,W2)
        moment_variance = sampleVar + simulationVar
        W3 = inv(moment_variance)
        # get variance covariance matrix
        VarCov = inv(D.T @ W3 @ D)/nConsumers
        # se
        se = np.sqrt(np.diag(VarCov))

        return(step2opt,se)



def softmax(x):
    """Compute softmax values for each sets of scores in x."""

    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def LogitGmmObjective(theta,X,Y,W,J,out=False):
    """Compute GMM obj. function, J, given data (X,Y) and parameters theta.
    Logit model with individual level data."""

    n,k = X.shape
    U = np.expand_dims(X @ theta,axis=1).reshape((J,n//J),order='F')
    Ypred = softmax(U).reshape((n,1),order='F')
    M = X*(Y-Ypred)        # moments
    G = M.mean(axis=0).T   # avg. moments 
    J = n * (G.T @ W @ G)  # GMM objective
    if not out:
        return J
    else:
        return J, M

def LogitGmmG(theta,X,Y,J):
    """Compute GMM G matrix, the derivative of the moments function,
    given data (X,Y,Z) and parameters theta.
    Logit model with individual level data."""

    n,k = X.shape
    U = np.expand_dims(X @ theta,axis=1).reshape((J,n//J),order='F')
    Ypred = softmax(U).reshape((n,1),order='F')

    Xi = X.reshape((J,int(n/J*k)),order='F') # each column is consumerXcharacteristic
    Xj = np.kron(Xi,np.ones((J,1))) # product 1 J times, then product 2 J times, and so on
    Xq = np.kron(np.ones((J,1)),Xi) # products 1 to J, repeat J times
    Wq = np.kron(np.ones((J,k)), softmax(U)) # weights
    Rq = (Xj-Xq)*Wq # Result times weight. Now need to sum every J entries
    Ri = Rq.reshape(-1,J,Rq.shape[-1]).sum(1) # weighted avg. of Xj-Xq
    R = Ri.reshape((n,k),order='F')
    G = -1/n * (X*Ypred).T @ R 
    
    return G

def LogitGetFittedProb(theta,X,Y,J):
    """Return the fitted value of Y in the logit model."""
    
    n,k = X.shape
    U = np.expand_dims(X @ theta,axis=1).reshape((J,n//J),order='F')
    Ypred = softmax(U).reshape((n,1),order='F')
    return Ypred
    
def MixedLogitGmmObj(theta,X,Zeta,XZeta,X12,nu,
                      J,k,ro,nuPosition,jChosenShare,
                      sampleG, useM2,secondChoice,W, out):
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
    M1 = X*Ypred  # moments
    # mean over j and N
    G1 = (M1.reshape((J,ni,k),order = 'F').mean(axis=1)).mean(axis=0)
    G1 = np.expand_dims(G1,axis=1) # make it a vector

    # moments set 2
    if useM2:
        M2 = XZeta*Ypred    # moments
        # weighted mean over j
        G2 =  (M2.reshape((J,ni,nXZeta),order='F').mean(axis=1)/
                Ypred.reshape((J,ni,1),order='F').mean(axis=1)*
                jChosenShare).sum(axis=0)    # avg. moments
        G2 = np.expand_dims(G2,axis=1) # make it a vector
    else:
        G2 = []
    
    # moments 3
    if secondChoice:
        k2 = X12.shape[1]
        X2Prob2 = secondChoiceXP(X12,U,n,J,k2)
        M3 = (X12*X2Prob2)*Ypred    # moments
        # weighted mean over j
        G3 =  (M3.reshape((J,ni,k2),order='F').mean(axis=1)/
                np.expand_dims(Ypred.reshape((J,ni),order='F').mean(axis=1),axis=1)*
                jChosenShare).sum(axis=0)   # avg. moments
        G3 = np.expand_dims(G3,axis=1) # make it a vector
    else:
        G3 = []

    # stack moments
    if useM2 and secondChoice:
        G = np.vstack((G1,G2,G3))
    elif useM2:
        G = np.vstack((G1,G2))
    elif secondChoice:
        G = np.vstack((G1,G3))
    else:
        G = G1

    # compute difference between sample and model moments
    Gd = sampleG - G

    if np.isnan(Gd).any():
        pass

    # Compute objective (J function)
    Obj = ni * (Gd.T @ W @ Gd)
    

    lastValue = Obj
    functionCount += 1

    if not out:
        return Obj
    else:
        return Obj, Gd 

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

def computeSampleVariance(M1,M2,M3,useM2,secondChoice, index):
    """Computes the variance covariance matrix of the moments vector"""

    if (secondChoice and useM2):
        # in this case use two samples
        n,k1 = M1.shape
        n,k2 = M2.shape
        # restrict to sample of M3
        M1r = M1[index,:]
        M2r = M2[index,:]
        # compute variance-covariance using full sample 
        M  = np.hstack((M1,M2))
        Var = np.cov(M.T)
        # compute variance-covariance using restricted sample
        Mr = np.hstack((M1r,M2r,M3))
        Var_r = np.cov(Mr.T)
        # replace the block with the full sample estimate
        Var_r[:(k1+k2),:(k1+k2)] = Var
        # return Var_r as Var in the end
        Var = Var_r

    elif secondChoice:
        # in this case use two samples
        n,k1 = M1.shape
        # restrict to sample of M3
        M1r = M1[index,:]
        # compute variance-covariance using full sample 
        M  = M1
        Var = np.cov(M.T)
        # compute variance-covariance using restricted sample
        Mr = np.hstack((M1r,M3))
        Var_r = np.cov(Mr.T)
        # replace the block with the full sample estimate
        Var_r[:(k1),:(k1)] = Var
        # return Var_r as Var in the end
        Var = Var_r
    elif useM2:
        M = np.hstack((M1,M2))
        Var = np.cov(M.T)
    else:
        Var = np.cov(M1.T)

    return Var

def computeDerivativeMomentMixedLogit(d,theta,X,Zeta,XZeta,X12,nu,
                      J,k,ro,nuPosition,jChosenShare,
                      sampleG, useM2,secondChoice,W):
    """Compute the expected moment derivative in the mixed logit model.
    we do so by numerical approximation (finite differences). d is the deviation
    in all directions of parameter theta"""
        
    k = len(theta) # number of parameters

    # get central value
    obj, gCentral = MixedLogitGmmObj(theta,X,Zeta,XZeta,X12,nu,
                      J,k,ro,nuPosition,jChosenShare,
                      sampleG, useM2,secondChoice,W, True)
    m,l = gCentral.shape # numer of moments

    # allocate memory for derivative D
    D = np.zeros([m,k])

    for i in range(k):

        # get values in theta_i + d
        theta_i = theta
        theta_i[i] = theta[i] + d
        obj,g_i =  MixedLogitGmmObj(theta_i,X,Zeta,XZeta,X12,nu,
                      J,k,ro,nuPosition,jChosenShare,
                      sampleG, useM2,secondChoice,W, True)
        D[:,i] = ((g_i-gCentral)/d).T
    
    return D

def computeSimulationVariance(theta,j,k,ro,nuPosition,jChosenShare,nMoments,ns,nr,
                            XrAll,ZetarAll,NurAll,XZetarAll,X12rAll,
                            sampleG,useM2,secondChoice,W):
    """Estimate the variance-covariance of moments due to simulation"""
    
    momentSimulations = np.zeros([nr,nMoments])
    for ri in range(nr):
        Xri     = XrAll[(ri*ns*j):((ri+1)*ns*j),:]
        Zetari  = ZetarAll[(ri*ns*j):((ri+1)*ns*j),:]
        Nuri    = NurAll[(ri*ns*j):((ri+1)*ns*j),:]
        XZetari = XZetarAll[(ri*ns*j):((ri+1)*ns*j),:]
        if len(X12rAll)>0:
            X12ri   = X12rAll[(ri*ns*j):((ri+1)*ns*j),:]
        else:
            X12ri   = []

        gg,momentsri = MixedLogitGmmObj(theta,Xri,Zetari,
                        XZetari,X12ri,Nuri,j,k,ro,nuPosition,jChosenShare,
                        sampleG, useM2,secondChoice,W, True)
        momentSimulations[ri,:] = momentsri.T
    simulationVar = np.cov(momentSimulations.T)

    return simulationVar
