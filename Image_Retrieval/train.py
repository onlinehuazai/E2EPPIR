import argparse
import torch
from tqdm import tqdm


# train for one epoch
def train(net, data_loader, train_optimizer, epoch, scheduler, epochs):
    net.train()
    scheduler.step()

    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    for im_1, label in train_bar:
        im_1, label = im_1.cuda(non_blocking=True), label.cuda(non_blocking=True)
        loss = net(im_1, label)

        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_num += data_loader.batch_size
        total_loss += loss.item() * data_loader.batch_size
        train_bar.set_description('Train Epoch: [{}/{}], lr: {:.6f}, Loss: {:.4f}'.format(epoch, epochs,
                                                                                          train_optimizer.param_groups[
                                                                                              0]['lr'],
                                                                                          total_loss / total_num))

    return total_loss / total_num

