import numpy as np
import pandas as pd
import HWDataProcessing as dp
import EstimationTools as et
from EstimationTools import Model

# Load data
dataFile = "data/Shining32.csv"
data = pd.read_csv(dataFile)

# Process data
data = dp.CleanDataSet(data)

# Define markets by location? Not implemented, just one market
#data = dp.addMarketVar(data,True)

# Simple logit estimation
# Model specification
J = 10     # number of products
yName = "Chosen"
xName = ["mint","white","fluoride","kids"]  # product characteristics
zName = []              # consumer attributes
pName = "priceperpack"  # product price
brands = "brandid"      # brand identifier
iv = ["discount"]

logitMod = Model(type='logit',data=data,J=J,y=yName,x=xName,z=zName,
                p=pName,brands = brands,iv=iv)

estimates,vcv = logitMod.fitLogit()
estimates,vcv

elasticities= logitMod.getElasticityLogit(estimates)
elasticities
#print(data.head(20))
#data.to_csv("treated.csv")