import numpy as np

def Normalization(nparray):
    item_max = np.max(nparray)
    item_min = np.min(nparray)

    for i in range(len(nparray)):
        nparray[i] = (nparray[i]-item_min)/(item_max-item_min)

    return nparray



