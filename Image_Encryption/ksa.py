import math
import numpy as np


def ksa(key):
    sc = [i for i in range(0,256)]
    sc.insert(0, 0)
    key = np.insert(key, 0, 0)
    j = 0
    for i in range(0,256):
        index=math.floor(i%(len(key)-1))
        j = math.floor((j+sc[i+1]+key[index+1]) % 256)
        temp = sc[i+1]
        sc[i+1] = sc[j+1]
        sc[j+1] = temp
    del sc[0]
    return sc