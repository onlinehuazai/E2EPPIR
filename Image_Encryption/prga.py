import numpy as np


def prga(sc, data):
    data = data[0]
    sc.insert(0, 0)
    data = np.insert(data, 0, 0)
    i = 0
    j = 0
    r = [0]
    for x in range(0, len(data) - 1):
        i = (i+1) % 256
        j = (j+sc[i+1]) % 256
        temp = sc[i+1]
        sc[i+1] = sc[j+1]
        sc[j+1] = temp
        r.append(sc[(sc[i+1]+sc[j+1]) % 256+1])
    del r[0]
    del sc[0]
    return r
