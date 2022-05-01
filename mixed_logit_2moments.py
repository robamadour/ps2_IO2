import numpy as np
import pandas as pd
import HWDataProcessing as dp
import EstimationTools as et
from EstimationTools import ModelSpecification
from EstimationTools import Model

np.set_printoptions(precision=3)

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
modelSpec.y2 = "SecondChoice" # choice variable
modelSpec.x = ["mint","white","fluoride","kids"]  # product characteristics
modelSpec.p = "priceperoz"  # product price
# obs. consumer attributes
modelSpec.zeta = ["inc","ed_MoreCollege","purchase_InStore"]  
                   
# Second moment interactions: choose which product characteristics (X) and 
# consumer attributes (zeta) to interact  to form first-choice moments
# It must be defined as a list of index pairs [X,zeta]
# Example: X=4(=len(x)) -> price; zeta=0 -> income
modelSpec.XZetaInter = [[0,0],[1,0],[2,0],[3,0],[4,0],
                        [0,1],[1,1],[2,1],[3,1],[4,1],
                        [0,2],[1,2],[2,2],[3,2],[4,2]]

# Third moment interactions: choose which product characteristics of first- and
# second-choice to interact to form second-choice momentes
# It must be defined as a list of indexes
# ExampleX=4 -> interact price
modelSpec.X1X2Inter = [] 

# unobs. consumer attributes. It is a kx1 vector, where k = len([X,p]), of 0s 
# and 1s. A 1 in entry k indicates that product characteristic k is interacted with
# an unobserved consumer attribute.
modelSpec.nu = np.array([0,0,0,0,1])
modelSpec.ns = 1000   # number of draws for Monte-Carlo integration
modelSpec.nr = 50    # number of resamplings needed to compute variance
modelSpec.seed = 1984 # seed for random number generation

modelSpec.secondChoice = False # Whether second choice moments are used in estimation
modelSpec.brands = "brandid"      # brand name variable

# Model instance creation
mixedLogitMod = Model(modelSpec)


# Estimation and results
mixedLogitMod.fit()
res = mixedLogitMod.reportEstimates()
print(res)

mixedLogitMod.estimateElasticities()
elasticities = mixedLogitMod.reportElasticities()
print(elasticities)