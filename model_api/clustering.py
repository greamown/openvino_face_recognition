import numpy as np
from database import OBJ_FEATURE

def euclidean_distance(vectors):
    (featsA, featsB) = vectors
    sumSquared = np.sum(np.square(featsA - featsB), axis=0, keepdims=True)
    epsilon = np.finfo(float).eps 
    max_value = np.maximum(sumSquared, epsilon)
    result = np.squeeze(np.sqrt(max_value))
    return result

def calcu_distance(featsA):
    distance, name = np.Inf, ""
    for ind, featsB in enumerate(OBJ_FEATURE['feature']):
        new_dis = euclidean_distance([featsA, featsB])
        distance = min(distance, new_dis)
        if new_dis == distance:
            name = OBJ_FEATURE['name'][ind]
    return name