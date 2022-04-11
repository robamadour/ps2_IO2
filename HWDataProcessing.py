import pandas as pd
import numpy as np

def CleanDataSet(data):
    """Cleans the shining32.csv data."""

    # Convert yes/no to 1/0
    dictYes  = {"Yes" : 1, "No" : 0}
    data.replace({"mint": dictYes,"white": dictYes,"fluoride": dictYes, 
                  "kids": dictYes},inplace=True)

    # Normalization
    data["sizeNorm"] = np.divide(data["size"]-np.min(data["size"]),
                                    np.max(data["size"])-np.min(data["size"]))
    data["inc"] = data["inc"]/1000
    
    # Define product id
    uniqueProducts  = data.groupby(['brandid','mint','white','fluoride','kids'])\
                          .size().reset_index().rename(columns={0:'count'})
    
    data["product"] = 0
    for i in range(len(uniqueProducts)):
        index = ((data.brandid == uniqueProducts.iloc[i,0])      & 
                (data.mint == uniqueProducts.iloc[i,1])          &
                (data.white == uniqueProducts.iloc[i,2])         &
                (data.fluoride == uniqueProducts.iloc[i,3])      &
                (data.kids == uniqueProducts.iloc[i,4]))
        data.iloc[index,-1] = i + 1

    # Add dummies
    data = data.join(pd.get_dummies(data.education,prefix="ed"))
    data = data.join(pd.get_dummies(data.location,prefix="loc"))
    data = data.join(pd.get_dummies(data.gender,prefix="gen"))
    data = data.join(pd.get_dummies(data.purchase,prefix="purchase"))
    data = data.join(pd.get_dummies(data.brandid,prefix="brand"))

    # Rename some columns
    data = data.rename({'ed_High School':'ed_HighSchool',
        'ed_More than College':'ed_MoreCollege',
        'purchase_In Store':'purchase_InStore'}, axis='columns')

    # Normalization
    data["sizeNorm"] = np.divide(data["size"]-np.min(data["size"]),
                                    np.max(data["size"])-np.min(data["size"]))
    data["inc"] = data["inc"]/1000

    return(data)

def genMCSample(data,ns,consumerId,unobsNames,seed,J):
    """Generates a new dataset with a similar structure to data, but with "ns"
    consumers randomly sampled from the data. Also adds draws from a normal 
    distribution to model unobservable consumer attributes."""
    
    # set seed
    np.random.seed(seed)
    
    nUnobs = len(unobsNames) # number of unobservable attributes to generate
    nConsumers = len(data[consumerId].unique()) # number of consumers in the data

    # random sample from all consumers
    consumers = 1 + np.random.randint(nConsumers, size=ns)

    # find consumers in the original data and add their attributes to new sample
    dataIndex = np.zeros(J*ns)
    for i in range(ns):
        dataIndex[(J*i):(J*(i+1))] = data[data[consumerId]==consumers[i]].index
    MCSample = data.iloc[dataIndex].copy()

    # generate unobservable attributes from a normal distribution
    unobs = np.random.normal(loc=0.0, scale=1.0, size=(ns,nUnobs))
    unobs = np.kron(unobs,np.ones((J,1)))

    # add "unobservables" to the sample 
    MCSample[unobsNames] = unobs

    return(MCSample)

def shareChoseJ(data,choice):
    
    shares = data[[choice,"product"]].groupby(by="product").mean()
    return(shares)








def addMarketVar(data,byBorough):
    if not byBorough:
        data["market"] = 1
    else:
        dictBorough = {"Brooklyn": 1, "Manhattan": 2, "Queens": 3, "Other":4}
        data["market"] = data["location"].map(dictBorough)
    return(data)


def getMarketLevelData(data,xNames,priceName):

    vars  = xNames + ["product", priceName]
    dataMkt  = data[vars].groupby(["product"]).agg("mean")
    sales  = data[["Chosen","product"]].groupby(["product"]).agg("sum")
    dataMkt = dataMkt.join(sales)
    dataMkt["delta"] = dataMkt["Chosen"]/np.sum(dataMkt.Chosen)
    return(dataMkt)