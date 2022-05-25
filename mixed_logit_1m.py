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
modelSpec.y2 = "SecondChoice" # choice variable
# product characteristics
modelSpec.x = ["brand_Colgate","brand_Crest","brand_Sensodyne",
                "mint","white","fluoride","kids",
                "sizeNorm","discount","familypack"]  
modelSpec.p = "priceperoz"  # product price
# obs. consumer attributes
modelSpec.zeta = [] 

# what consumer and product characteristics to interact to form random coefficients
# list of pairs [X,zeta]
modelSpec.XZetaRC = []
                   
# Second moment interactions: choose which product characteristics (X) and 
# consumer attributes (zeta) to interact  to form first-choice moments
# It must be defined as a list of index pairs [X,zeta]
# Example: X=4(=len(x)) -> price; zeta=0 -> income
# get as cartersian product
nX = len(modelSpec.x) +1
nZeta = len(modelSpec.zeta)
x = np.arange(nX)
zeta = np.arange(nZeta)
if nZeta>0:
    modelSpec.XZetaInter = np.transpose([np.tile(x, len(zeta)),
                                np.repeat(zeta, len(x))]).tolist()
else:
    modelSpec.XZetaInter = []

# Third moment interactions: choose which product characteristics of first- and
# second-choice to interact to form second-choice momentes
# It must be defined as a list of indexes
# ExampleX=4 -> interact price
modelSpec.X1X2Inter = [] 

# unobs. consumer attributes. It is a kx1 vector, where k = len([X,p]), or 0s 
# and 1s. A 1 in entry k indicates that product characteristic k is interacted with
# an unobserved consumer attribute.
#nu = np.zeros((len(modelSpec.x)+1,))
#nu[-1] = 1
nu = np.ones((len(modelSpec.x)+1,))
modelSpec.nu = nu
modelSpec.ns = 1000   # number of draws for Monte-Carlo integration
modelSpec.nr = 50    # number of resamplings needed to compute variance
modelSpec.seed = 1984 # seed for random number generation

modelSpec.secondChoice = False # Whether second choice moments are used in estimation
modelSpec.brands = "brandid"      # brand name variable
modelSpec.M2M3short = True  # whether moments M2 and M3 are computed
                            # using short formula or not

# Model instance creation
mixedLogitMod = Model(modelSpec)


# Estimation and results
mixedLogitMod.fit()
estimates = mixedLogitMod.reportEstimates('step2')
estimates_s1 = mixedLogitMod.reportEstimates('step1')
print(estimates)

mixedLogitMod.estimateElasticities()
elasticities = mixedLogitMod.reportElasticities()
print(elasticities)

# save to excel file
file = 'outputs/m_short_peroz_rcPrice_unobs.xlsx'
sheet_estimates = 'estimates'
sheet_estimates_s1 = 'estimates_step1'
sheet_elasticities = 'elasticities'

estimates.to_excel(file,sheet_name=sheet_estimates)
with pd.ExcelWriter(file,mode='a',if_sheet_exists='replace') as writer:  
    estimates.to_excel(writer,sheet_name=sheet_estimates)
    estimates_s1.to_excel(writer,sheet_name=sheet_estimates_s1)
    elasticities.to_excel(writer,sheet_name=sheet_elasticities)