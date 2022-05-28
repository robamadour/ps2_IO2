from asyncio.windows_events import NULL
from audioop import mul
from matplotlib.pyplot import axis
import numpy as np
import pandas as pd
from numpy.linalg import inv
from psutil import ZombieProcess
import statsmodels.api as sm
import statsmodels.formula.api as smf
import HWDataProcessing as dp
from scipy.optimize import fmin_bfgs
import warnings


class ModelSpecification():
    
    def copy(self):
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
        # RC interactions
        self.XZetaRCPairs = spec.XZetaRC 
        self.XZetaRC = pairsToIndex(spec.XZetaRC,len(self.xName)+1,len(self.zName))  
        self.XZetaInter = spec.XZetaInter  # first-choice interactions
        self.X1X2Inter =  spec.X1X2Inter   # first- and second-choice interaction
        self.M2M3short =  spec.M2M3short   # whether moments M2 and M3 are computed
                                           # using short formula or not

        
        # demean varaibles (important for moments involving correlations)
        self.demean()

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

    def demean(self):
        
        xNameNoConst = self.xName.copy()
        if 'constant' in xNameNoConst:
            xNameNoConst.remove('constant')
        old_data = self.data[xNameNoConst+self.zName].copy()
        means = np.expand_dims(old_data.to_numpy().mean(axis=0),axis=0)
        sds =  np.expand_dims(old_data.to_numpy().std(axis=0),axis=0)       
        a =  (self.data[xNameNoConst+self.zName].to_numpy() - means)/sds
        self.data[xNameNoConst+self.zName] = a
        return self
    
    def fit(self):
        """Estimate the model"""

        modelType = self.type
        match modelType:
            case 'logit':
                self.estimatesS1, self.seS1,self.estimates, self.se = self.fitLogit()
            case 'mixed_logit':
                self.estimatesS1, self.seS1,self.estimates, self.se = self.fitMixedLogit()
        
    def reportEstimates(self,step = 'step2'):
        """Report parameter estimates."""

        modelType = self.type

        if step == 'step1':
            estimate = self.estimatesS1
            se = self.seS1
        else:
            estimate = self.estimates
            se = self.se

        match modelType:
            case 'logit':
                estimates = pd.DataFrame()
                estimates["var. name"] = self.xName + [self.pName]
                estimates["coefficient"] = estimate
                estimates["s.e."] = se
                estimates["t-stat"] = estimates["coefficient"]/estimates["s.e."]
                estimates["sig"] = np.where(
                        np.abs(estimates['t-stat']) >= 2.576, "***", np.where(
                        np.abs(estimates['t-stat']) >= 1.96, "**", np.where(
                        np.abs(estimates['t-stat']) >= 1.645, "*", ""    
                        ))) 


            case 'mixed_logit':
                
                regressors =  self.xName + [self.pName]
                consumerAttr = self.zName
                unobsConsAttr = self.unobsNames
                nuPosition = self.nu
                
            
                k = len(regressors)
                ro = len(consumerAttr)
                ru = int(sum(self.nu))
                nRC = len(self.XZetaRC)
                nParameters = k + nRC + ru

                XZNames = []
                pairs = self.XZetaRCPairs
                if len(consumerAttr)>0:
                    for i in range(nRC):                        
                        thisVar = regressors[pairs[i][0]] + '_' + consumerAttr[pairs[i][1]]
                        XZNames = XZNames + [thisVar]
                
                unobsNames = []
                if len(nuPosition) > 0:
                    for i in range(k):
                        if nuPosition[i] == 1:
                            unobsNames = unobsNames + [regressors[i]]

                paramName = ["betaBar"]*k + ["betaO"]*(nRC) + ["betaU"]*ru
                varName = regressors + XZNames + unobsNames

                estimates = pd.DataFrame()
                estimates["coeficient"] = paramName
                estimates["var. name"] = varName
                estimates["coefficient"] = estimate
                estimates["s.e."] = se
                estimates["t-stat"] = estimates["coefficient"]/estimates["s.e."]
                estimates["sig"] = np.where(
                        np.abs(estimates['t-stat']) >= 2.576, "***", np.where(
                        np.abs(estimates['t-stat']) >= 1.96, "**", np.where(
                        np.abs(estimates['t-stat']) >= 1.645, "*", ""    
                        ))) 

        return estimates
    
    def estimateElasticities(self):
        modelType = self.type
        match modelType:
            case 'logit':
                self.elasticities = self.getElasticityLogit()
            case 'mixed_logit':
                self.elasticities, self.L1, self.L2, self.L3,\
                    self.profit_before, self.profit_merger,\
                    self.delta_welfare, self.corr = \
                    self.getElasticitiesMixedLogit(1e-2)
        

    def reportElasticities(self):
        """Report estimated elasticities."""

        elasticities = self.elasticities

        table = pd.DataFrame({'product quantity' : 1+np.arange(self.J)})
        for j in range(self.J):
            table["p_"+str(1+j)] = elasticities[:,j]
        
        table["L_singleProd"] = self.L1
        table["L_multiProd"] = self.L2
        table["L_merger"] = self.L3
        table["Profit_before"] = self.profit_before
        table["Profit_merger"] = self.profit_merger
        table["Change_welfare"] = self.delta_welfare
        
        print("-----------------------------------------------------")
        print("Correlation between brand fixed effects and price sensitivity:")
        print(self.corr)
        return table 
                

    
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
        # compute se step1
        G = LogitGmmG(step1opt,X,Y,j)
        out = LogitGmmObjective(step1opt, X,Y,W,j,out=True)
        S = np.cov(out[1].T)
        vcv = inv(G @ inv(S) @ G.T)/n
        seStep1 = np.sqrt(np.diag(vcv))

        # second step
        Omega = np.cov(out[1].T)
        W2 = inv(Omega)
        args = (X, Y, W2, j)
        step2opt = fmin_bfgs(LogitGmmObjective, step1opt, args=args)

        # Estimate VCV of parameter estimates
        out = LogitGmmObjective(step2opt, X,Y,W2,j,out=True)
        G = LogitGmmG(step2opt,X,Y,j)
        S = np.cov(out[1].T)
        vcv = inv(G @ inv(S) @ G.T)/n
        seStep2 = np.sqrt(np.diag(vcv))

        return(step1opt,seStep1,step2opt,seStep2)

    def getElasticityLogit(self):
        """Compute elasticity in the Logit model. This command has to be runned 
        after fitting the model."""

        # Define  matrices
        theta = self.estimates
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

    def getComputationMatrices(self):

        nuPosition = (np.array(self.nu) == 1)
        XZetaRC = self.XZetaRC
        xzPairs = self.XZetaInter  # interactions for first-choice moments
        data = self.data.copy()

        # share of consumers that chose product j 
        jChosenCount = self.data[[self.yName,"productId"]]\
                        .groupby(by="productId").sum().to_numpy()
        
        # Define  matrices X, Z , Zeta, Y
        regressors =  self.xName + [self.pName]
        consumerAttr = self.zName
        unobsConsAttr = self.unobsNames

        X = self.data[regressors].to_numpy()    # product characteristics
        Zeta = self.data[consumerAttr].to_numpy() # observed consumer attributes
        J = self.J  # number of products
        nXZpairs = len(xzPairs)
        XZetaNames = ["XZeta_" + str(s)  for s in range(nXZpairs)]
        for i in range(nXZpairs):
            varName = XZetaNames[i]
            Xvar = regressors[xzPairs[i][0]]
            Zetavar = consumerAttr[xzPairs[i][1]]
            data[varName] = data[Xvar]*data[Zetavar]
        XZeta  = data[XZetaNames].to_numpy()

        ro = Zeta.shape[1]
        
        MCSample = self.MCSample

        NurAll = MCSample[unobsConsAttr].to_numpy()
        assert MCSample.shape[0]>=X.shape[0]
        nu    = NurAll[:X.shape[0],:]

        price_index = X.shape[1]-1

        return X, Zeta, XZeta, nu, J, ro, nuPosition,XZetaRC, price_index

    def getElasticitiesMixedLogit(self,d):
        
        theta = self.estimates
        X, Zeta, XZeta, nu, J, ro, nuPosition,XZetaRC, price_index = self.getComputationMatrices()
        Y = np.expand_dims(self.data[self.yName], axis=1)

        # shape
        n,k = X.shape
        ni = n//J # number of consumers
        nXZeta = XZeta.shape[1]
        
        # unpack theta 
        betaBar, betaO, betaU = unpackParameters(theta,k,ro,nuPosition,XZetaRC)

        # random coefficients
        beta = np.ones((n,k))*betaBar.T + Zeta @ (betaO.T) +\
            nu*(np.ones((n,k))*betaU.T)
        
        # Get prob. of buying product j
        U = (X*beta).sum(axis=1).reshape((J,ni),order='F')
        Ypred = softmax(U).reshape((n,1),order='F')

        # get price
        p = np.expand_dims(X[:,price_index],axis=1)

        # avg price and probability
        avg_p  = p.reshape((J,ni),order='F').mean(axis=1)
        avg_y = Ypred.reshape((J,ni),order='F').mean(axis=1)

        # get price coefficient
        p_coeff = np.expand_dims(beta[:,-1],axis=1)

        # elasticity matrix
        elasticities_1 = np.zeros([J,J])
        DyDp = np.zeros([J,J])

        for j in range(J):
            
            # product index
            index_j = np.zeros((J,1))
            index_j[j,0] = 1
            index_j = np.kron(np.ones((ni,1)),index_j)

            # price j
            pj = avg_p[j]
            
            Yj = np.kron(Ypred.reshape((J,ni),order='F')[j,:],np.ones((1,J))).T

            # compute elasticity wrt price pj

            # derivative
            dydp = ((index_j - Yj)*p_coeff*Ypred).\
                        reshape((J,ni),order='F').mean(axis=1)            

            DyDp[:,j] = dydp

            # average across consumers
            e1 = dydp*pj/avg_y
            elasticities_1[:,j] = e1 

        # Correlation: brand fixed efects and price sensitivity
        betaCorr = beta[:,[0,1,2,price_index]]
        Corr = np.corrcoef(betaCorr.T)
        
        # compute Lerner indexes

        # Ownership matrices
        Omega_1 = np.identity(J)
        Omega_2 = np.zeros((J,J))
        Omega_2[:3,:3] = 1
        Omega_2[3:6,3:6] = 1
        Omega_2[6:8,6:8] = 1
        Omega_2[8:10,8:10] = 1

        avg_p = np.expand_dims(avg_p,1)
        avg_y = np.expand_dims(avg_y,1)

        # Lerner indexes
        L1 = -(inv(Omega_1*DyDp) @ avg_y)/avg_p
        L2 = -(inv(Omega_2*DyDp) @ avg_y)/avg_p

        # Get marginal costs
        mc = avg_p + inv(Omega_2*DyDp) @ avg_y

        # Solve for prices after merger of Crest and Colgate
        
        # new ownership matrix
        Omega_3 = np.zeros((J,J))
        Omega_3[:3,:3] = 1
        Omega_3[3:8,3:8] = 1
        Omega_3[8:10,8:10] = 1

        # initial price
        p0 = mc - inv(Omega_3*DyDp) @ avg_y
        args = (Omega_3,DyDp,beta,X,mc,price_index,n,J,ni)
        # numerical solver
        step1opt = fmin_bfgs(FuncPMerger, p0, args=args)
        p_post_merger = step1opt.reshape((J,1))

        # get new Lerner index
        L3 = (p_post_merger - mc)/p_post_merger

        # compute profits
        
        # at equilibrium
        Pi_0 = (avg_p-mc)*avg_y*ni
        # after merger
        y_post_merger = getNewShare(p_post_merger,beta,X,price_index,n,J,ni) 
        Pi_merger = (p_post_merger-mc)*y_post_merger.reshape((J,1))*ni


        # Remove Crest

        # predict first chice without Crest     

        # first choice if Crest is not available
        UnC = U.copy()
        UnC[6:8,:] = -np.inf
        choice = oneMax(UnC)
        choice = choice.reshape((n,1),order='F')

        # identify consumers who chose Crest in the first place
        crest_index = np.zeros((J,1))
        crest_index[6:8] = 1
        crest_index = np.kron(np.ones((ni,1)),crest_index)
        chose_crest = (Y==crest_index)
        chose_crest = chose_crest.reshape((J,ni)).sum(axis=0)

        # price coefficient to normalize utlity in dollar terms
        p_coeff_i = p_coeff.reshape((J,ni)).mean(axis=0)

        # computation of welfare change for consumers who chose Crest first and 
        # had to change
        U2 = U.reshape((n,1),order='F')
        u_actual_choice = (U2*Y).reshape((J,ni),order='F').sum(axis=0)
        u_no_crest = (U2*choice).reshape((J,ni),order='F').sum(axis=0)
        d_welfare = ((u_no_crest-u_actual_choice)/p_coeff_i*chose_crest).sum()


        return elasticities_1, L1, L2, L3, Pi_0, Pi_merger,d_welfare,Corr      

    
    def getElasticitiesMixedLogit2(self,d):
        
        theta = self.estimates
        X, Zeta, XZeta, nu, J, ro, nuPosition,XZetaRC, price_index = self.getComputationMatrices()

        # shape
        n,k = X.shape
        ni = n//J # number of consumers
        nXZeta = XZeta.shape[1]
        
        # unpack theta 
        betaBar, betaO, betaU = unpackParameters(theta,k,ro,nuPosition,XZetaRC)

        # random coefficients
        beta = np.ones((n,k))*betaBar.T + Zeta @ (betaO.T) +\
            nu*(np.ones((n,k))*betaU.T)
        
        # Get prob. of buying product j
        U = (X*beta).sum(axis=1).reshape((J,ni),order='F')
        Ypred = softmax(U).reshape((n,1),order='F')

        # get price
        p = np.expand_dims(X[:,price_index],axis=1)

        # elasticity matrix
        elasticities_1 = np.zeros([J,J])
        elasticities_2 = np.zeros([J,J])

        for j in range(J):
            # change in price
            dp = np.zeros([J,1])
            dp[j,:] = d*p.mean(axis=0).mean()
            pj = p + np.kron(np.ones((ni,1)),dp)
            Xj = np.copy(X)
            Xj[:,price_index] = pj.reshape((n,))
            Uj = (Xj*beta).sum(axis=1).reshape((J,ni),order='F')
            Yj = softmax(Uj).reshape((n,1),order='F')

            # compute elasticity wrt price pj
            ej = ((Yj-Ypred)/d)*p/Ypred

            # average across consumers
            e1 = ej.reshape((J,ni),order='F').mean(axis=1)
            elasticities_1[:,j] = e1 

            # method 2: first average
            avgDY = (Yj-Ypred).reshape((J,ni),order='F').mean(axis=1)
            if np.any(avgDY==0):
                pass
            avgY = (Ypred).reshape((J,ni),order='F').mean(axis=1)
            avgP = (p).reshape((J,ni),order='F').mean(axis=1)
            e2 = (avgDY/avgY)/(d/avgP)
            elasticities_2[:,j] = e2

        return elasticities_2
    
    def fitMixedLogit(self):
        """Estimate a mixed logit model via GMM"""

        # Specification
        data = self.data.copy()
        nuPosition = (np.array(self.nu) == 1)
        XZetaRC = self.XZetaRC
        useM2 = self.useM2
        secondChoice = self.secondChoice # whether to use second-choice moments
        xzPairs = self.XZetaInter  # interactions for first-choice moments
        x1x2Charac = self.X1X2Inter  # interactions for second-choice moments
        M2M3short = self.M2M3short

        # share of consumers that chose product j 
        jChosenCount = self.data[[self.yName,"productId"]]\
                        .groupby(by="productId").sum().to_numpy()
        jChosenShare = jChosenCount/sum(jChosenCount)
        
        # Define  matrices X, Z , Zeta, Y
        regressors =  self.xName + [self.pName]
        consumerAttr = self.zName
        unobsConsAttr = self.unobsNames

        X = self.data[regressors].to_numpy()    # product characteristics
        Zeta = self.data[consumerAttr].to_numpy() # observed consumer attributes
        Y = np.expand_dims(self.data[self.yName], axis=1) # first choice


        # restrict data to chosen products as first choice
        dataChosen = self.data[self.data[self.yName] == 1].copy()

        # restrict data to chosen products and info about second choice
        dataSecondChoice = self.data[(~ self.data[self.y2Name].isnull()) &
                           ((self.data[self.yName]==1) |
                           (self.data[self.y2Name]==1))].copy()
        
        # consumers for whom there is second choice data
        SecondData = self.data[(~ self.data[self.y2Name].isnull())].copy()
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
        ru = int(sum(self.nu))   # number of unobserved consumer attributes
        nRC = len(XZetaRC)  # number of RC with observed attributes
        nXZpairs = len(xzPairs) # number of first-choice moments
        nX1X2pairs = len(x1x2Charac) # number of first-choice moments
        nParameters = k + nRC + ru
        nMoment1 = k
        nMoment2 = nXZpairs
        nMoment3 = nX1X2pairs
        nMoments = nMoment1 + nMoment2 + nMoment3 # total number of moments

        #assert nMoments>=nParameters, "Model is underindentified"
        if nMoments>=nParameters:
            warnings.warn("Model is underindentified")

        ns = self.ns
        nr = self.nr 
        
        # add X*Zeta to datasets
        XZetaNames = ["XZeta_" + str(s)  for s in range(nXZpairs)]
        for i in range(nXZpairs):
            varName = XZetaNames[i]
            Xvar = regressors[xzPairs[i][0]]
            Zetavar = consumerAttr[xzPairs[i][1]]
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

                X1 = firstProductData[Xvar].to_numpy()
                #X1 = (X1 - np.expand_dims(X1.mean(axis=0),0))/\
                #        np.expand_dims(X1.std(axis=0),0)
                X2 = secondProductData[Xvar].to_numpy()
                #X2 = (X2 - np.expand_dims(X2.mean(axis=0),0))/\
                #        np.expand_dims(X2.std(axis=0),0)

                dataInteractionsX1X2[varName] = X1*X2
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
            
            #sampleM3 = X1X2sample[Y1SecondData.reshape((nConsSecondData*j,))==1,:]
            sampleM3 = (X1X2sample*Y1SecondData).\
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
        theta0 = np.zeros([nParameters,1])+0.1
        W = np.identity(nMoments)

        # initialize iterations
        global iteration, lastvalue, functionCount, lastBetaO, lastBetaBar
        iteration = 0
        lastValue = 0
        functionCount = 0

        # Minimization
        # first step
        out = False
        args = (Xr, Zetar,XZetar,X12r,Nur,j,k,ro,nuPosition,XZetaRC,jChosenShare,
                M2M3short,sampleG,useM2,secondChoice,W,out)
        step1opt = fmin_bfgs(MixedLogitGmmObj, theta0, args=args,
                              callback=iter_print)

        # Compute variance-covariance matrix
        # simulation variance
        simulationVar = computeSimulationVariance(step1opt,j,k,ro,nuPosition,XZetaRC,
                            jChosenShare,M2M3short,nMoments,ns,nr,
                            XrAll,ZetarAll,NurAll,XZetarAll,X12rAll,
                            sampleG,useM2,secondChoice,W)

        # compute s.e. of first step
        d = 1e-6 
        D = MixedLogitGmmDerivative(step1opt,Xr,Zetar,XZetar,X12r,Nur,
                      j,k,ro,nuPosition,XZetaRC,jChosenShare,M2M3short,
                      sampleG, useM2,secondChoice)
        VarCov = np.linalg.lstsq(D.T @ W @ D,np.eye(nParameters))[0]/nConsumers
        # se
        seStep1 = np.sqrt(np.diag(VarCov))
        
        # compute total variance
        moment_variance = sampleVar + simulationVar

        # set W = inv(variance)
        P = np.diag(np.diag(moment_variance))
        y = np.linalg.lstsq(moment_variance@inv(P),np.eye(nMoments))[0]
        W2 = np.linalg.lstsq(P,y)[0]        

        # second step
        args = (Xr, Zetar,XZetar,X12r,Nur,j,k,ro,nuPosition,XZetaRC,jChosenShare,
                M2M3short,
                sampleG,useM2,secondChoice,W2,out)
        step2opt = fmin_bfgs(MixedLogitGmmObj, step1opt, args=args,
                              callback=iter_print)
                
        # compute s.e.
        # Get numerical derivative
        d = 1e-6
        D_num = computeDerivativeMomentMixedLogit(d,step2opt,Xr,Zetar,XZetar,X12r,Nur,
                      j,k,ro,nuPosition,XZetaRC,jChosenShare,M2M3short,
                      sampleG, useM2,secondChoice,W2)

        D = MixedLogitGmmDerivative(step2opt,Xr,Zetar,XZetar,X12r,Nur,
                      j,k,ro,nuPosition,XZetaRC,jChosenShare,M2M3short,
                      sampleG, useM2,secondChoice)

        #D = D_num
        
        simulationVar = computeSimulationVariance(step2opt,j,k,ro,nuPosition,
                            XZetaRC,
                            jChosenShare,M2M3short,nMoments,ns,nr,
                            XrAll,ZetarAll,NurAll,XZetarAll,X12rAll,
                            sampleG,useM2,secondChoice,W2)
        moment_variance = sampleVar + simulationVar
        W3 = np.linalg.lstsq(moment_variance,np.eye(nMoments))[0]
        
        # get variance covariance matrix
        # precondition method
        A = D.T @ W3 @ D
        P = np.diag(np.diag(A))
        y = np.linalg.lstsq(A@inv(P),np.eye(nParameters))[0]
        VarCov = np.linalg.lstsq(P,y)[0]/nConsumers
        # se
        seStep2 = np.sqrt(np.abs(np.diag(VarCov)))

        return(step1opt,seStep1,step2opt,seStep2)

    def print_results(self,file):
        estimates = self.reportEstimates('step2')
        estimates_s1 = self.reportEstimates('step1')
        print(estimates)

        self.estimateElasticities()
        elasticities = self.reportElasticities()
        print(elasticities)

        # save to excel file
        sheet_estimates = 'estimates'
        sheet_estimates_s1 = 'estimates_step1'
        sheet_elasticities = 'elasticities'

        estimates.to_excel(file,sheet_name=sheet_estimates)
        with pd.ExcelWriter(file,mode='a',if_sheet_exists='replace') as writer:  
            estimates.to_excel(writer,sheet_name=sheet_estimates)
            estimates_s1.to_excel(writer,sheet_name=sheet_estimates_s1)
            elasticities.to_excel(writer,sheet_name=sheet_elasticities)



def softmax(x):
    """Compute softmax values for each sets of scores in x."""

    e_x = np.exp(x - 0*np.expand_dims(x[0,:],axis=0)) +1e-30
    r = e_x / e_x.sum(axis=0)
    #assert ~np.isnan(r).any()
    return r

def keepMax(x):
    n,k = x.shape
    b = np.zeros_like(x)
    b[x.argmax(0),np.arange(k)] = 1
    return b 

def oneMax(x):
    n,k = x.shape
    b = np.zeros_like(x)
    b[x.argmax(0),np.arange(k)] = 1
    return b

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
                      J,k,ro,nuPosition,XZetaRC,jChosenShare,M2M3short,
                      sampleG, useM2,secondChoice,W, out):
    """Compute GMM obj. function, given data (X,Y,Z,Zeta,Nu) and parameters theta.
    Mixed Logit model with individual level data."""
    
    global lastValue, functionCount, lastBetaBar, lastBetaO # used in message printing

    # shape
    n,k = X.shape
    ni = n//J # number of consumers
    nXZeta = XZeta.shape[1]
    
    # unpack theta 
    betaBar, betaO, betaU = unpackParameters(theta,k,ro,nuPosition,XZetaRC)

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
        if M2M3short:
            G2 =  (M2.reshape((J,ni,nXZeta),order='F').mean(axis=1)).sum(axis=0)
        else:
            G2 =  (M2.reshape((J,ni,nXZeta),order='F').mean(axis=1)/
                    Ypred.reshape((J,ni,1),order='F').mean(axis=1)*
                    jChosenShare).sum(axis=0)
        G2 = np.expand_dims(G2,axis=1) # make it a vector
    else:
        G2 = []
    
    # moments 3
    if secondChoice:
        k2 = X12.shape[1]
        X2Prob2 = secondChoiceXP(X12,U,n,J,k2)
        M3 = (X12*X2Prob2)*Ypred    # moments
        # weighted mean over j
        if M2M3short:
            G3 =  (M3.reshape((J,ni,k2),order='F').mean(axis=1)).sum(axis=0)
        else:
            G3 =  (M3.reshape((J,ni,k2),order='F').mean(axis=1)/
                np.expand_dims(Ypred.reshape((J,ni),order='F').mean(axis=1),axis=1)*
                jChosenShare).sum(axis=0)
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

    #if functionCount//5 == 0:
     #   print(str((Gd/np.expand_dims(np.sqrt(np.diag(W)),axis=1)).reshape((1,Gd.shape[0]))))
        #print(str(betaBar))
        #print(str(betaO))

    

    assert ~np.isnan(Gd).any(), "Moments contain NaN"

    # Compute objective (J function)
    Obj = ni * (Gd.T @ W @ Gd)
    
    assert Obj>=0, "Negative objective function"
    #if np.isnan(Gd).any():
    #    Obj = lastValue*2
    #    warnings.warn("Warning: found NaN in moments")

    lastValue = Obj
    #lastBetaBar = betaBar
    #lastBetaO = betaO
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
    U = U.reshape((J-1,J*ni),order='F')
    #Prob = keepMax(softmax(U)) # choose product with max utility
    Prob = oneMax(U) # choose product with max utility
    Prob = Prob.reshape(((J-1)*J,ni),order='F')
    
    Prob = np.kron(np.ones((1,1,k)),np.expand_dims(Prob,axis=2)) 
    XProb = (X*Prob).reshape((J-1,J,ni,k),order='F').sum(axis=0)\
                    .reshape((n,k),order='F')

    return XProb    
    

def unpackParameters(theta,k,ro,nuPosition,XZetaRC):

    nRC = len(XZetaRC)
    betaBar = theta[:k].reshape((k,1))

    betaO = np.zeros((k*ro,)) 
    betaO[XZetaRC] = theta[k:(k+nRC)]
    betaO = betaO.reshape((k,ro),order='F')

    betaUp = theta[(k+nRC):]
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
                      J,k,ro,nuPosition,XZetaRC,jChosenShare,M2M3short,
                      sampleG, useM2,secondChoice,W):
    """Compute the expected moment derivative in the mixed logit model.
    we do so by numerical approximation (finite differences). d is the deviation
    in all directions of parameter theta"""
        
    k = len(theta) # number of parameters

    # get central value
    obj, gCentral = MixedLogitGmmObj(theta,X,Zeta,XZeta,X12,nu,
                      J,k,ro,nuPosition,XZetaRC,jChosenShare,M2M3short,
                      sampleG, useM2,secondChoice,W, True)
    m,l = gCentral.shape # numer of moments

    # allocate memory for derivative D
    D = np.zeros([m,k])

    for i in range(k):

        # get values in theta_i + d
        theta_i = theta.copy()
        theta_i[i] = theta[i] + d
        obj,g_i =  MixedLogitGmmObj(theta_i,X,Zeta,XZeta,X12,nu,
                      J,k,ro,nuPosition,XZetaRC,jChosenShare,M2M3short,
                      sampleG, useM2,secondChoice,W, True)
        D[:,i] = ((g_i-gCentral)/d).T
    
    return D

def MixedLogitGmmDerivative(theta,X,Zeta,XZeta,X12,nu,
                      J,k,ro,nuPosition,XZetaRC,jChosenShare,M2M3short,
                      sampleG, useM2,secondChoice):
    """Compute GMM obj. function, given data (X,Y,Z,Zeta,Nu) and parameters theta.
    Mixed Logit model with individual level data."""
    
    global lastValue, functionCount, lastBetaBar, lastBetaO # used in message printing

    n,k = X.shape
    
    betaBar, betaO, betaU = unpackParameters(theta,k,ro,nuPosition,XZetaRC)

    # random coefficients
    thetaC = np.ones((n,k))*betaBar.T + Zeta @ (betaO.T) +\
           nu*(np.ones((n,k))*betaU.T)

    U = np.expand_dims((X * thetaC).sum(axis=1),axis=1).\
                    reshape((J,n//J),order='F')
    Ypred = softmax(U).reshape((n,1),order='F')
    Xi = X.reshape((J,int(n/J*k)),order='F') # each column is consumerXcharacteristic
    Xj = np.kron(Xi,np.ones((J,1))) # product 1 J times, then product 2 J times, and so on
    Xq = np.kron(np.ones((J,1)),Xi) # products 1 to J, repeat J times
    Wq = np.kron(np.ones((J,k)), softmax(U)) # weights
    Rq = (Xj-Xq)*Wq # Result times weight. Now need to sum every J entries
    Ri = Rq.reshape(-1,J,Rq.shape[-1]).sum(1) # weighted avg. of Xj-Xq
    R = Ri.reshape((n,k),order='F')

    # shape
    ni = n//J # number of consumers
    nXZeta = XZeta.shape[1]
    nParameters = len(theta)

    RCDerivative = np.zeros((n,nParameters))

    for i in range(nParameters):
        thetaD = np.zeros((len(theta),))
        thetaD[i] = 1
        betaBar, betaO, betaU = unpackParameters(thetaD,k,ro,nuPosition,XZetaRC)
        Dbeta = np.ones((n,k))*betaBar.T + Zeta @ (betaO.T) +\
           nu*(np.ones((n,k))*betaU.T)
        RCDerivative[:,i] = (Dbeta * R).sum(axis=1)

    
    # Derivative of G1
    D1 = -1/n * (X*Ypred).T @ RCDerivative 

    # Derivative of G2
    if useM2:
        M2 = -XZeta*Ypred
        D2 = -J/n * M2.T @ RCDerivative
    else:
        D2 = []

    # Derivative of G3
    if secondChoice:
        k2 = X12.shape[1]
        X2Prob2 = secondChoiceXP(X12,U,n,J,k2)
        M3 = (X12*X2Prob2)*Ypred

        D3 = -J/n * M3.T @ RCDerivative
    else:
        D3 = []

    # stack derivatives
    if useM2 and secondChoice:
        D = np.vstack((D1,D2,D3))
    elif useM2:
        D = np.vstack((D1,D2))
    elif secondChoice:
        D = np.vstack((D1,D3))
    else:
        D = D1

    assert ~np.any(D==0), "Derivative = 0"

    return D

def computeSimulationVariance(theta,j,k,ro,nuPosition,XZetaRC,jChosenShare,M2M3short,
                            nMoments,ns,nr,
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
                        XZetari,X12ri,Nuri,j,k,ro,nuPosition,XZetaRC,jChosenShare,
                        M2M3short,sampleG, useM2,secondChoice,W, True)
        momentSimulations[ri,:] = momentsri.T
    simulationVar = np.cov(momentSimulations.T)

    return simulationVar


def pairsToIndex(pairs,k,r):

    nPairs = len(pairs)

    index = np.zeros((nPairs,),dtype=np.uint64)
    for i in range(nPairs):
        entry = pairs[i][1]*k + pairs[i][0]
        assert entry <k*r, "Index is not valid"
        index[i] = int(entry)

    return index  
    

def FuncPMerger(p,Omega,DyDp,beta,X,mc,price_index,n,J,ni):
        X2 = X.copy()
        p = np.expand_dims(p,axis=1)
        X2[:,price_index] = np.kron(np.ones((ni,1)),p).reshape((n,))

        U = (X*beta).sum(axis=1).reshape((J,ni),order='F')
        Ypred = softmax(U).reshape((n,1),order='F')
        avg_y = Ypred.reshape((J,ni),order='F').mean(axis=1)

        r = np.linalg.norm(p - mc + inv(Omega*DyDp) @ avg_y)

        return r

def getNewShare(p,beta,X,price_index,n,J,ni):
        X2 = X.copy()
        #p = np.expand_dims(p,axis=1)
        X2[:,price_index] = np.kron(np.ones((ni,1)),p).reshape((n,))

        U = (X*beta).sum(axis=1).reshape((J,ni),order='F')
        Ypred = softmax(U).reshape((n,1),order='F')
        avg_y = Ypred.reshape((J,ni),order='F').mean(axis=1)

        return avg_y