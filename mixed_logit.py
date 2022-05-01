import numpy as np
import pandas as pd
import HWDataProcessing as dp
import EstimationTools as et
from EstimationTools import ModelSpecification
from EstimationTools import Model

# Load data
dataFile = "data/Shining32.csv"
data = pd.read_csv(dataFile)

# Process data
data = dp.CleanDataSet(data)

# Mixed logit estimation ######################################################
# Model specification
modelSpec = ModelSpecification()
modelSpec.type = 'mixed_logit' # two options: logit, mixed_logit
modelSpec.data = data
modelSpec.J = 10     # number of products
modelSpec.i = "buyerid" # consumer identifier
modelSpec.y = "Chosen" # choice variable
modelSpec.y2 = "Chosen" # choice variable
modelSpec.x = ["mint","white","fluoride","kids"]  # product characteristics
modelSpec.p = "priceperpack"  # product price
modelSpec.zeta = ["inc"]              # obs. consumer attributes

# Second moment interactions: choose which product characteristics (X) and 
# consumer attributes to interact (zeta) to form first-choice moments
# It must be defined as a list of index pairs [X,zeta]
modelSpec.XZetaInter = [[4,0]] # X=4(=len(x)) -> price; zeta=0 -> income 

# Third moment interactions: choose which product characteristics of first- and
# second-choice to interact to form second-choice momentes
# It must be defined as a list of indexes
modelSpec.X1X2Inter = [4] #X=4 -> interact price

# unobs. consumer attributes. It is a kx1 vector, where k = len([X,p]), or 0s 
# and 1s. A 1 in entry k indicates that product characteristic k is interacted with
# an unobserved consumer attribute.
modelSpec.nu = np.array([1,1,1,1,0])
modelSpec.ns = 1000   # number of draws for Monte-Carlo integration
modelSpec.nr = 50    # number of resamplings needed to compute variance
modelSpec.seed = 1984 # seed for random number generation

modelSpec.secondChoice = True # Whether second choice moments are used in estimation
modelSpec.brands = "brandid"      # brand name variable

# Model instance creation
mixedLogitMod = Model(modelSpec)


# Estimation and results
mixedLogitMod.fit()
res = mixedLogitMod.reportEstimates()
print(res)
