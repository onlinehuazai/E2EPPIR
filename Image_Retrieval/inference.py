import torch
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np


def test(net, test_loader):
    net.eval()
    feature_bank = []
    labels = []
    with torch.no_grad():
        for data_1, label in tqdm(test_loader):
            feature = net(data_1.cuda(non_blocking=True))
            label = label.cuda()
            feature = F.normalize(feature, dim=1)
            feature_bank.append(feature)
            labels.append(label)
        feature_bank = torch.cat(feature_bank, dim=0).contiguous()
        feature_labels = torch.cat(labels, dim=0).contiguous()
        average_precision_li = []
        for idx in range(feature_bank.size(0)):
            query = feature_bank[idx].expand(feature_bank.shape)
            label = feature_labels[idx]
            sim = F.cosine_similarity(feature_bank, query)
            _, indices = torch.topk(sim, 10)
            match_list = feature_labels[indices] == label
            pos_num = 0
            total_num = 0
            precision_li = []
            for item in match_list[1:]:
                if item == 1:
                    pos_num += 1
                    total_num += 1
                    precision_li.append(pos_num / float(total_num))
                else:
                    total_num += 1

            if precision_li == []:
                average_precision_li.append(0)
            else:
                average_precision = np.mean(precision_li)
                average_precision_li.append(average_precision)
        mAP = np.mean(average_precision_li)
        print('test mAP:', mAP)
        return mAP

