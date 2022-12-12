import numpy as np
import cv2
import scipy.io as scio
from yates_shuffle import yates_shuffle
import os
from ksa import ksa
from prga import prga


def encryption_operation(plainimage, dataset, key_permutation, key_selection, key_repalcement_r, key_repalcement_g, key_repalcement_b, srcFile, block_size=16, rate=0.5):
    row, col, channel = plainimage.shape
    R = plainimage[:, :, 0]
    G = plainimage[:, :, 1]
    B = plainimage[:, :, 2]
    for i in range(0, int(16*np.ceil(col/16)-col)):
        R = np.c_[R, R[:, -1]]
        G = np.c_[G, G[:, -1]]
        B = np.c_[B, B[:, -1]]
        
    for i in range(0, int(16*np.ceil(row/16)-row)):
        R = np.r_[R, [R[-1, :]]]
        G = np.r_[G, [G[-1, :]]]
        B = np.r_[B, [B[-1, :]]]
        
    # generate block permutation sequence
    block_number=int((row*col)/(block_size*block_size))
    block_index = [i for i in range(block_number)]
    p_block = yates_shuffle(block_index, key_permutation)
    
    # random select value replace block index, %
    val_rep_idx = yates_shuffle(block_index, key_selection)
    repBlockNume = int(rate * block_number)

    # generate value replace sequence
    colorNumRange = [i for i in range(0, 256)]
    r_val_rep = yates_shuffle(colorNumRange, key_repalcement_r)
    g_val_rep = yates_shuffle(colorNumRange, key_repalcement_g)
    b_val_rep = yates_shuffle(colorNumRange, key_repalcement_b)


    # encryption 1: block permuta1
    block_r = np.zeros([block_size, block_size, block_number])
    block_g = np.zeros([block_size, block_size, block_number])
    block_b = np.zeros([block_size, block_size, block_number])
    count = 0
    for n in range(0, row, block_size):
        for m in range(0, col, block_size):
            block_r[:, :, p_block[count]] = R[n:n+block_size, m:m+block_size]
            block_g[:, :, p_block[count]] = G[n:n+block_size, m:m+block_size]
            block_b[:, :, p_block[count]] = B[n:n+block_size, m:m+block_size]
            count = count + 1
            
    # encryption 2: value replacement
    for num in range(repBlockNume):
        tempr = block_r[:, :, val_rep_idx[num]]
        tempg = block_g[:, :, val_rep_idx[num]]
        tempb = block_b[:, :, val_rep_idx[num]]
        for n in range(block_size):
            for m in range(block_size):
                tempr[n, m] = r_val_rep[int(tempr[n, m])]
                tempg[n, m] = g_val_rep[int(tempg[n, m])]
                tempb[n, m] = b_val_rep[int(tempb[n, m])]
        block_r[:, :, val_rep_idx[num]] = tempr
        block_g[:, :, val_rep_idx[num]] = tempg
        block_b[:, :, val_rep_idx[num]] = tempb
        
    # restore
    count = 0
    for n in range(0, row, block_size):
        for m in range(0, col, block_size):
            R[n:n+block_size, m:m+block_size] = block_r[:, :, count]
            G[n:n+block_size, m:m+block_size] = block_g[:, :, count]
            B[n:n+block_size, m:m+block_size] = block_b[:, :, count]
            count = count + 1
    
    # save cipher-image
    merged = cv2.merge([B.astype(np.uint8), G.astype(np.uint8), R.astype(np.uint8)])
    path = '../data/cipherimages_'+ dataset + '_' + str(block_size)+'_'+str(repBlockNume)
    if not os.path.exists(path):
        os.makedirs(path)
    
    cv2.imwrite(path+'/{}'.format(srcFile.split("/")[-1].split("\\")[-1]), merged, [int(cv2.IMWRITE_JPEG_QUALITY), 90])


def generate_bitstream(key_path, row, col, idx):
    key = scio.loadmat(key_path)  # Y component encryption key
    key = key['key'][0][idx]

    data_len = np.ones([1, row * col])
    s = ksa(key)
    r = prga(s, data_len)
    encryption_key = ''
    for i in range(0, len(r)):
        temp1 = str(r[i])
        temp2 = bin(int(temp1, 10))
        temp2 = temp2[2:]
        for j in range(0, 8 - len(temp2)):
            temp2 = '0' + temp2
        encryption_key = encryption_key + temp2
    return encryption_key


def generate_encryption_key(key, row, col):
    encryption_key_permutation = generate_bitstream(key, row, col, 1)
    encryption_key_selection = generate_bitstream(key, row, col, 2)
    encryption_key_repalcement_r = generate_bitstream(key, row, col, 3)
    encryption_key_repalcement_g = generate_bitstream(key, row, col, 4)
    encryption_key_repalcement_b = generate_bitstream(key, row, col, 5)

    return encryption_key_permutation, encryption_key_selection, encryption_key_repalcement_r, encryption_key_repalcement_g, encryption_key_repalcement_b
