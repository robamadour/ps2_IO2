import numpy as np
import pandas as pd
import HWDataProcessing as dp
import EstimationTools as et
from EstimationTools import ModelSpecification
from EstimationTools import Model

# Load data
dataFile = "data/Shining32.csv"
data = pd.read_csv(dataFile)

# Choose model
model = 3

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
modelSpec.x = ["brand_Aquafresh","brand_Colgate","brand_Sensodyne",
                "mint","white","fluoride","kids",
                "sizeNorm","discount","familypack"]  
modelSpec.p = "priceperoz"  # product price
# obs. consumer attributes
modelSpec.zeta = []  
# what consumer and product characteristics to interact to form random coefficients
# list of pairs [X,zeta]
indexP = len(modelSpec.x)
modelSpec.XZetaRC = []
# Second moment interactions                   
modelSpec.XZetaInter = []
# Third moment interactions
modelSpec.X1X2Inter = []
# unobserved characteristics 
nu = np.ones((len(modelSpec.x)+1,))
#nu[indexP] = 0
modelSpec.nu = nu
modelSpec.ns = 1000   # number of draws for Monte-Carlo integration
modelSpec.nr = 10    # number of resamplings needed to compute variance
modelSpec.seed = 1984 # seed for random number generation

modelSpec.secondChoice = False # Whether second choice moments are used in estimation
modelSpec.brands = "brandid"      # brand name variable

modelSpec.M2M3short = True  # whether moments M2 and M3 are computed
                            # using short formula or not

X1X2Inter = [3,8,7,indexP,1,5,9,4] 
zeta = ["inc","ed_HighSchool","purchase_InStore","age","gen_Female"] 
XZetaRC = [
     [indexP,0], # price and income
     [3,2],[5,2],[6,2],[7,2],[8,2], # in store and several
     [4,1], [5,1], [7,1], [9,1] # moreCollege and several characteristics 
    ]

XZetaInter = [
    [indexP,0], # price and income
    [0,1], [0,2],  # Aquafresh vs ed_MoreCollege, purchase_InStore
    [2,1], [2,2],  # Sensodyne and...
    [4,0], [8,0], # white, discount and income
    [4,1], [5,1], [7,1], [9,1], # moreCollege and several characteristics
    [3,2],[5,2],[6,2],[7,2],[8,2] # in store and several
]

match model:
    case 1:
        # simple logit
        modelSpec.nu = np.array([])

    case 2:
        # mixed logit, no observed characteristics
        pass # nothing to change here

    case 3:
        # mixed logit, no obs. charactericstics + 2nd choice moments
        modelSpec.X1X2Inter = X1X2Inter
        modelSpec.secondChoice = True
    
    case 4:
        # mixed logit + obs. charactericstics
        modelSpec.zeta = zeta
        modelSpec.XZetaRC = XZetaRC
    
    case 5:
        # mixed logit + obs ch. + 1st choice moments
        modelSpec.zeta = zeta
        modelSpec.XZetaRC = XZetaRC
        modelSpec.XZetaInter = XZetaInter

    case 6:
        # mixed logit + obs. ch + 2nd choice moments
        modelSpec.zeta = zeta
        modelSpec.XZetaRC = XZetaRC
        modelSpec.X1X2Inter = X1X2Inter
        modelSpec.secondChoice = True

    case 7:
         # mixed logit + obs. ch + 1st and 2nd choice moments
        modelSpec.zeta = zeta
        modelSpec.XZetaRC = XZetaRC
        modelSpec.X1X2Inter = X1X2Inter
        modelSpec.XZetaInter = XZetaInter
        modelSpec.secondChoice = True

    case 8:
         # mixed logit + 1st choice moments
        modelSpec.zeta = zeta
        modelSpec.XZetaInter = XZetaInter

    case 9:
         # mixed logit +  1st and 2nd choice moments
        modelSpec.zeta = zeta
        modelSpec.X1X2Inter = X1X2Inter
        modelSpec.XZetaInter = XZetaInter
        modelSpec.secondChoice = True



# Model instance creation
myModel = Model(modelSpec)

# Estimation and results
myModel.fit()

# print results
filename = 'outputs/mod_' + str(model) + '.xlsx'
myModel.print_results(filename)