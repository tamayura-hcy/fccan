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
import mmd
import math

batch_size = 8


class Discriminator(nn.Module):
    """Discriminator model for source domain."""

    def __init__(self, input_dims, hidden_dims, output_dims):
        """Init discriminator."""
        super(Discriminator, self).__init__()

        self.restored = False

        self.layer = nn.Sequential(
            nn.Linear(input_dims, hidden_dims),
            nn.BatchNorm1d(hidden_dims),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Linear(hidden_dims, hidden_dims),
            nn.BatchNorm1d(hidden_dims),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Linear(hidden_dims, output_dims),
        )

    def forward(self, input):
        """Forward the discriminator."""
        out = self.layer(input)
        out = torch.sigmoid(out)
        return out


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


def ddc_train(encoder, classifier, dataloader_train_s, dataloader_train_t, epochs, ddc_loss_type):
    since = time.time()
    """Train classifier for source domain."""
    lr= 0.003

    # setup criterion and optimizer
    optimizer = optim.SGD(list(encoder.parameters()) + list(classifier.parameters()),
                          lr=lr, momentum=0.9)
    criterion_cls = nn.CrossEntropyLoss()

    train_batches = len(dataloader_train_s)

    for epoch in range(epochs):

        encoder.train()
        classifier.train()
        loss_train, loss_val, acc_train, acc_val = 0, 0, 0, 0
        loss_train_cls, loss_train_ddc = 0, 0

        for step, (inputs, labels) in enumerate(dataloader_train_s):
            with torch.no_grad():
                if torch.cuda.is_available():
                    inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

            data_tgt, _ = next(iter(dataloader_train_t))

            with torch.no_grad():
                if torch.cuda.is_available():
                    data_tgt = Variable(data_tgt.cuda())
                else:
                    data_tgt = Variable(data_tgt)

            # zero gradients for optimizer
            optimizer.zero_grad()

            # compute cls loss for critic
            feature_s = encoder(inputs)
            outputs = classifier(feature_s.view(feature_s.size(0),-1))
            _, preds = torch.max(outputs.data, 1)
            loss_cls = criterion_cls(outputs, labels)

            # compute ddc loss
            feature_t = encoder(data_tgt)

            # Squeeze
            feature_s = feature_s.view(feature_s.size(0),-1)
            feature_t = feature_t.view(feature_t.size(0),-1)

            if ddc_loss_type == 'sinhorn':
                loss_ddc = mmd.sinhorn_loss(feature_s, feature_t)
            elif ddc_loss_type == 'mmd_linear':
                loss_ddc = mmd.mmd_linear(feature_s, feature_t)
            elif ddc_loss_type == 'poly_mmd2':
                loss_ddc = mmd.poly_mmd2(feature_s, feature_t)

            # Combine cls loss and ddc loss
            mylambda = 2 / (1 + math.exp(-10 * float(epoch) / epochs)) - 1
            loss = loss_cls + mylambda * loss_ddc

            # optimize source encoder and classifier
            loss.backward()
            optimizer.step()

            loss_train += loss.data.item()
            acc_train += torch.sum(preds == labels.data).item()

            loss_train_cls += loss_cls.data.item()
            loss_train_ddc += loss_ddc.data.item()

            del inputs, labels, preds,feature_s, feature_t
            torch.cuda.empty_cache()

        avg_loss = loss_train / (train_batches * batch_size)
        avg_acc = acc_train / (train_batches * batch_size)
        avg_loss_cls = loss_train_cls / (train_batches * batch_size)
        avg_loss_ddc = loss_train_ddc / (train_batches * batch_size)

        print("Epoch-{} | train loss={:.4f} | train ddc loss={:.4f} | train cls loss={:.4f} | train acc={:.4f}"
              .format(epoch, avg_loss, avg_loss_ddc,avg_loss_cls,avg_acc))

    elapsed_time = time.time() - since
    print("Target Training completed in {:.2f}s".format(elapsed_time))
    return encoder, classifier


def main(args):

    use_gpu = torch.cuda.is_available()
    if use_gpu: print("Using CUDA")

    epochs = args.epochs
    source_dataset = args.source
    target_dataset =args.target
    ddc_loss_type = args.ddcloss

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

    dataloader_t = {x: torch.utils.data.DataLoader(image_dataset_t[x], batch_size=batch_size,shuffle=True, num_workers=0,
                                                   drop_last=True)
                    for x in [TRAIN_T, TEST_T]}

    dataset_sizes_tgt = {x: len(image_dataset_t[x]) for x in [TRAIN_T, TEST_T]}

    for x in [TRAIN_T, TEST_T]: print("Loaded {} images under Target {}".format(dataset_sizes_tgt[x], x))

    class_names = image_dataset_t[TRAIN_T].classes
    print("Classes: ", image_dataset_t[TRAIN_T].classes)

    test_acc = []
    test_acc_src = []

    for iter in range(1,iterations+1):

        save_name = './model_saved/DDC_' + source_dataset + '_to_' + target_dataset +'_iter' + str(iter)

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

        src_encoder,src_classifier = ddc_train(src_encoder,src_classifier,dataloader_s[TRAIN_S],
                                                   dataloader_t[TRAIN_T], epochs, ddc_loss_type)

        print("Test scr_encoder + src_classifier on Source Test dataset")
        src_acc = test(src_encoder,src_classifier,dataloader_s[TEST_S], dataset_sizes_src[TEST_S])
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

    parser.add_argument('-d', '--ddcloss',
                        help='ddc loss type, choose from[sinhorn, mmd_linear, poly_mmd2]',
                        type=str,
                        default='mmd_linear')

    args = parser.parse_args()
    main(args)
