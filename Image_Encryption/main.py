## image encyption - content owner
import glob
import cv2
from tqdm import tqdm
from encryption_operation import generate_encryption_key, encryption_operation


if __name__ == '__main__':
    # read plain-images
    dataset = 'ukbench'
    srcFiles = glob.glob('../data/'+dataset+'/*.jpg')
    key_path = 'key.mat'
    if dataset == 'Corel10K':
        encryption_key_permutation, encryption_key_selection, encryption_key_repalcement_r, encryption_key_repalcement_g, encryption_key_repalcement_b = generate_encryption_key(key_path, 192, 128)
    else:
        encryption_key_permutation, encryption_key_selection, encryption_key_repalcement_r, encryption_key_repalcement_g, encryption_key_repalcement_b = generate_encryption_key(key_path, 320, 240)

    length_srcFiles = len(srcFiles)
    for imagename in tqdm(srcFiles):
        img = cv2.imread(imagename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encryption_operation(img, dataset, encryption_key_permutation, encryption_key_selection, encryption_key_repalcement_r, encryption_key_repalcement_g, encryption_key_repalcement_b, imagename, block_size=16, rate=0.95)
