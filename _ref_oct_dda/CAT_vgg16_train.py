import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import argparse
import statistics
from torch.autograd import Function
import math
import torch.nn.functional as F
from util_cat import unsorted_segment_sum_device

batch_size = 8
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def adaptation_factor(x):
    den=1.0+math.exp(-10*x)
    lamb=2.0/den-1.0
    return min(lamb,1.0)


class Discriminator(nn.Module):
    def __init__(self, input_features):
        super(Discriminator, self).__init__()
        self.input_features = input_features
        hidden_size = 200

        # Define hidden linear layers
        self.fc1 = nn.Linear(input_features,hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size,1)
        # Activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = torch.sigmoid(self.out(x))
        return x


def test(encoder, classifier, dataloader_test, dataset_size_test):
    since = time.time()
    acc_test = 0
    for i, data in enumerate(dataloader_test):
        encoder.eval()
        classifier.eval()
        inputs, labels = data

        with torch.no_grad():
            if torch.cuda.is_available():
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            features = encoder(inputs)
            outputs = classifier(features.view(features.size(0), -1))
            _, preds = torch.max(outputs.data, 1)
            acc_test += torch.sum(preds == labels.data).item()

        del inputs, labels, features, preds
        torch.cuda.empty_cache()

    elapsed_time = time.time() - since
    print("Test completed in {:.2f}s".format(elapsed_time))

    avg_acc = float(acc_test) / dataset_size_test
    print("test acc={:.4f}".format(avg_acc))
    print()
    torch.cuda.empty_cache()
    return avg_acc


def cat_train(encoder, classifier, netD, dataloader_train_s, dataloader_train_t, epochs):
    since = time.time()
    """Train classifier for source domain."""
    lr= 0.001
    LAMBDA = 0.01  # distance margin between each cluster,  LAMBDA =30 in original paper
    class_nums = 3
    lamb2_rampup = 5  # Control the ratio of classification loss and sntg_loss
    lamb2_rampup_win = 15  # Control the ratio of classification loss and sntg_loss

    # setup criterion and optimizer
    optimizer = optim.SGD(list(encoder.parameters()) + list(classifier.parameters()),
                          lr=lr, momentum=0.9)
    optimizerD = optim.SGD(netD.parameters(),lr=0.001, momentum=0.9)

    criterion = nn.CrossEntropyLoss()

    encoder.train()
    classifier.train()
    netD.train()

    gd=0
    for epoch in range(epochs):

        gd+=1
        lamb2 = math.exp(-(1 - min((gd - lamb2_rampup) * 1.0 / lamb2_rampup_win, 1.)) * 10) if gd >= lamb2_rampup else 0.
        lamb = adaptation_factor(gd * 1.0 / epochs)

        len_dataloader = min(len(dataloader_train_s), len(dataloader_train_t))
        iter_source = iter(dataloader_train_s)
        iter_target = iter(dataloader_train_t)
        num_iter = len_dataloader

        err_cls, err_sntg, err_sntg_s, err_sntg_t, err_D, acc_train = 0, 0, 0, 0, 0, 0

        for i in range(num_iter):

            data_source, label_source = iter_source.next()
            data_target, _ = iter_target.next()

            with torch.no_grad():
                if torch.cuda.is_available():
                    data_source, label_source = Variable(data_source.cuda()), Variable(label_source.cuda())
                    data_target = Variable(data_target.cuda())
                else:
                    data_source, label_source = Variable(data_source), Variable(label_source)
                    data_target = Variable(data_target)

            optimizer.zero_grad()
            optimizerD.zero_grad()

            # Classification Loss
            features = encoder(data_source)
            features = features.view(features.size(0), -1)
            label_source_pred = classifier(features)
            _, preds = torch.max(label_source_pred.data, 1)
            loss_cls = criterion(label_source_pred, label_source)

            # SNTG Loss
            feature_t = encoder(data_target)
            feature_t = feature_t.view(feature_t.size(0), -1)
            target_feature = classifier(feature_t)

            with torch.no_grad():
                feature_t_2 = encoder(data_target)
                feature_t_2 = feature_t_2.view(feature_t_2.size(0), -1)
                target_feature_2 = classifier(feature_t_2)
                target_feature_2 = target_feature_2.detach()

            source_feature = label_source_pred
            target_pred = target_feature_2

            # Covert target_pred to onehot
            index = torch.max(target_pred, 1)[1].unsqueeze_(-1)
            target_pred_onehot = torch.zeros(batch_size, class_nums).cuda().scatter_(1, index, 1)

            # use D to calculate the logits, input:(?,3), output:(?,1)
            source_logits = netD(source_feature)  # (32,1)
            target_logits = netD(target_feature)  # (32,1)

            y = torch.zeros(batch_size, class_nums).cuda().scatter_(1, label_source.unsqueeze(-1), 1)
            graph_source = torch.sum(y[:, None, :] * y[None, :, :], dim=2)
            distance_source = torch.mean((source_feature[:, None, :] - source_feature[None, :, :]) ** 2,
                                         dim=2)

            source_sntg_loss = torch.mean(
                graph_source * distance_source + (1 - graph_source) * F.relu(LAMBDA - distance_source))

            source_result = torch.max(y,1)[1]
            target_result = torch.max(target_pred,1)[1]

            current_source_count = unsorted_segment_sum_device(torch.ones_like(source_result, dtype=float).cuda(),
                                                               source_result, class_nums, device)  # (7,)
            current_target_count = unsorted_segment_sum_device(torch.ones_like(target_result, dtype=float).cuda(),
                                                               target_result, class_nums, device)  # (7,)

            current_positive_source_count = torch.max(current_source_count,
                                                      torch.ones_like(current_source_count).to(device)).float()  # (7,)
            current_positive_target_count = torch.max(current_target_count,
                                                      torch.ones_like(current_target_count).to(device)).float()  # (7,)

            current_source_centroid = torch.div(
                unsorted_segment_sum_device(data=source_feature, segment_ids=source_result, num_segments=class_nums,
                                            device=device),
                current_positive_source_count[:, None])  # (7,7)
            current_target_centroid = torch.div(
                unsorted_segment_sum_device(data=target_feature, segment_ids=target_result, num_segments=class_nums,
                                            device=device),
                current_positive_target_count[:, None])  # (7,7)

            fm_mask = torch.gt(current_source_count * current_target_count, 0).float()  # (7,)
            fm_mask /= torch.mean(fm_mask + 1e-8)

            graph_target = torch.sum(target_pred_onehot[:, None, :] * target_pred_onehot[None, :, :], dim=2)
            distance_target = torch.mean((target_feature[:, None, :] - target_feature[None, :, :]) ** 2, dim=2)

            target_sntg_loss = torch.mean(
                graph_target * distance_target + (1 - graph_target) * F.relu(LAMBDA - distance_target))

            sntg_loss = torch.mean(torch.mean((current_source_centroid - current_target_centroid) ** 2,
                                              1) * fm_mask) + target_sntg_loss + source_sntg_loss

            D_real_loss = torch.mean(
                nn.MultiLabelSoftMarginLoss()(target_logits, torch.ones_like(target_logits).to(device)))
            D_fake_loss = torch.mean(
                nn.MultiLabelSoftMarginLoss()(source_logits, torch.zeros_like(source_logits).to(device)))

            D_loss = D_real_loss + D_fake_loss
            G_loss = -D_loss
            # Scale adversarial loss by 0.1 following RevGrad
            D_loss = 0.1 * D_loss
            G_loss = 0.1 * G_loss

            loss = loss_cls + lamb2 * sntg_loss

            loss.backward(retain_graph=True)
            D_loss.backward()

            optimizer.step()
            optimizerD.step()

            acc_train += torch.sum(preds == label_source.data).item()
            err_cls += loss_cls.data.item()
            err_sntg += sntg_loss.data.item()
            err_sntg_s += source_sntg_loss.data.item()
            err_sntg_t += target_sntg_loss.data.item()
            err_D += D_loss.data.item()

        print('Epoch: ', epoch, '| classify_loss: %.5f' % (err_cls/num_iter),
              '| sntg_loss: %.5f' % (err_sntg/num_iter), '| source sntg_loss: %.5f' % (err_sntg_s/num_iter),
              '| target sntg_loss: %.5f' % (err_sntg_t/num_iter),
              '| D_loss: %.5f' % (err_D/num_iter),
              '| train acc: %.4f' % (acc_train /num_iter/batch_size))

    elapsed_time = time.time() - since
    print("Target Training completed in {:.2f}s".format(elapsed_time))
    return encoder, classifier


def main(args):

    use_gpu = torch.cuda.is_available()
    if use_gpu: print("Using CUDA")

    epochs = args.epochs
    source_dataset = args.source
    target_dataset =args.target

    if source_dataset == target_dataset:
        print("Same source and target dataset. Exit!")
        exit()

    if source_dataset == 'BOE':
        s_data_dir = 'BOE_split_by_person'
        print(" Loading {} data set as Source".format(s_data_dir))
    elif source_dataset == 'CELL':
        print(" Loading CELL data set as Source")
        s_data_dir = './OCT2017'
    elif source_dataset =='TMI':
        print(" Loading TMI data set as Source ")
        s_data_dir = './TMIdata_split_by_person'

    if target_dataset == 'BOE':
        print(" Loading BOE data set as Target")
        t_data_dir = 'BOE_split_by_person'
    elif target_dataset == 'CELL':
        print(" Loading CELL data set as Target")
        t_data_dir = './OCT2017'
    elif target_dataset =='TMI':
        t_data_dir = 'TMIdata_split_by_person'
        print(" Loading {} data set as Target ".format(t_data_dir))

    TRAIN_S, VAL_S, TEST_S = 'train', 'val','test'
    TRAIN_T, TEST_T ='train', 'test'

    iterations = 5

    # VGG-16 Takes 224x224 images as input, so we resize all of them
    data_transform_s = {
        TRAIN_S: transforms.Compose([
            transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor()]),
        VAL_S: transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor()]),
        TEST_S: transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor()])
    }

    image_dataset_s = {x: datasets.ImageFolder(os.path.join(s_data_dir, x),transform=data_transform_s[x])
        for x in [TRAIN_S, VAL_S, TEST_S]
    }

    dataloader_s = {x: torch.utils.data.DataLoader(image_dataset_s[x], batch_size=batch_size,
            shuffle=True, num_workers=0,  drop_last=True # num_workers = 4 will cause code restart
        )
        for x in [TRAIN_S, VAL_S, TEST_S]
    }

    dataset_sizes_src = {x: len(image_dataset_s[x]) for x in [TRAIN_S, VAL_S, TEST_S]}

    for x in [TRAIN_S, VAL_S, TEST_S]: print("Loaded {} images under Source {}".format(dataset_sizes_src[x], x))

    class_names = image_dataset_s[TRAIN_S].classes
    print("Classes: ",image_dataset_s[TRAIN_S].classes)

    # VGG-16 Takes 224x224 images as input, so we resize all of them
    data_transform_t = {
        TRAIN_T: transforms.Compose([
            transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor()]),
        TEST_T: transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor()])
    }

    image_dataset_t = {x: datasets.ImageFolder(os.path.join(t_data_dir, x), transform=data_transform_t[x])
                      for x in [TRAIN_T, TEST_T]}

    dataloader_t = {x: torch.utils.data.DataLoader(image_dataset_t[x], batch_size=batch_size,shuffle=True, num_workers=0,drop_last=True)
                    for x in [TRAIN_T, TEST_T]}

    dataset_sizes_tgt = {x: len(image_dataset_t[x]) for x in [TRAIN_T, TEST_T]}

    for x in [TRAIN_T, TEST_T]: print("Loaded {} images under Target {}".format(dataset_sizes_tgt[x], x))

    class_names = image_dataset_t[TRAIN_T].classes
    print("Classes: ", image_dataset_t[TRAIN_T].classes)

    test_acc = []
    test_acc_src = []

    for iter in range(1,iterations+1):

        save_name = './model_saved/CAT_' + source_dataset + '_to_' + target_dataset +'_iter' + str(iter)

        # create model
        print('Create VGG16 model.................................................')
        vgg16 = models.vgg16_bn()
        print('vgg16.classifier[6].out_features=', vgg16.classifier[6].out_features) # 1000

        vgg16.load_state_dict(torch.load("vgg16_bn.pth"))

        # Newly created modules have require_grad=True by default
        num_features = vgg16.classifier[6].in_features
        features = list(vgg16.classifier.children())[:-1] # Remove last layer
        features.extend([nn.Linear(num_features, len(class_names))]) # Add our layer with 4 outputs
        vgg16.classifier = nn.Sequential(*features) # Replace the model classifier
        if use_gpu: vgg16.cuda()  # .cuda() will move everything to the GPU side

        # Create Models
        src_encoder = vgg16.features
        src_classifier = vgg16.classifier

        # Create Discriminator
        class_nums =3
        netD = Discriminator(input_features=class_nums).cuda()

        src_encoder,src_classifier = cat_train(src_encoder,src_classifier,netD, dataloader_s[TRAIN_S],dataloader_t[TRAIN_T],epochs)

        print("Test scr_encoder + src_classifier on Source Test dataset")
        src_acc = test(src_encoder, src_classifier, dataloader_s[TEST_S], dataset_sizes_src[TEST_S])
        test_acc_src.append(src_acc)

        print("Test scr_encoder + src_classifier on Target Test dataset")
        tgt_acc = test(src_encoder, src_classifier, dataloader_t[TEST_T], dataset_sizes_tgt[TEST_T])
        test_acc.append(tgt_acc)

        torch.save(src_encoder.state_dict(), save_name + '_encoder.pt')
        torch.save(src_classifier.state_dict(), save_name + '_classifier.pt')

    print('Source  test_acc=', test_acc_src)
    test_acc_avg = sum(test_acc_src) / len(test_acc_src)
    test_acc_var = statistics.stdev(test_acc_src)
    print("Source average test acc: %.4f" % (test_acc_avg), '| Variance test: %.4f' % (test_acc_var))

    print('Target test_acc=', test_acc)
    test_acc_avg = sum(test_acc) / len(test_acc)
    test_acc_var = statistics.stdev(test_acc)
    print("Target Average test acc: %.4f" % (test_acc_avg), '| Variance test: %.4f' % (test_acc_var))
    print("The End")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-s', '--source',
                        help='source dataset, choose from [BOE,CELL,TMI]',
                        type=str,
                        choices=['BOE','CELL','TMI'],
                        default='BOE')

    parser.add_argument('-t', '--target',
                        help='target dataset, choose from [BOE,CELL,TMI]',
                        type=str,
                        choices=['BOE','CELL','TMI'],
                        default='TMI')

    parser.add_argument('-e', '--epochs',
                        help='training epochs',
                        type=int,
                        default=30)

    args = parser.parse_args()
    main(args)
