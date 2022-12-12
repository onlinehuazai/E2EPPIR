import copy


def yates_shuffle(plain,key):
    p = copy.copy(plain)
    n = len(p)
    p.insert(0,0)
    bit_len = len(bin(int(str(n),10))) - 1
    key = '0' + key
    key_count = 1
    for i in range(n,1,-1):
        num = int('0b' + key[key_count:key_count+bit_len], 2) + 1
        index = num % i + 1
        temp = p[i]
        p[i] = p[index]
        p[index] = temp
        key_count = key_count + 1
    del p[0]
    return p
