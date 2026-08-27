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


def train_src(encoder, classifier, dataloader_train, dataloader_val, epochs, save_name):
    """Train classifier for source domain."""
    lr= 0.001

    # setup criterion and optimizer
    optimizer = optim.SGD(list(encoder.parameters()) + list(classifier.parameters()),
                          lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    train_batches = len(dataloader_train)
    val_batches = len(dataloader_val)
    best_encoder = copy.deepcopy(encoder.state_dict())
    best_classifier = copy.deepcopy(classifier.state_dict())
    best_acc = 0.0

    for epoch in range(epochs):

        encoder.train()
        classifier.train()
        loss_train, loss_val, acc_train, acc_val = 0, 0, 0, 0

        for step, (inputs, labels) in enumerate(dataloader_train):
            with torch.no_grad():
                if torch.cuda.is_available():
                    inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

            # zero gradients for optimizer
            optimizer.zero_grad()

            # compute loss for critic
            features = encoder(inputs)
            outputs = classifier(features.view(features.size(0),-1))
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)

            # optimize source classifier
            loss.backward()
            optimizer.step()

            loss_train += loss.data.item()
            acc_train += torch.sum(preds == labels.data).item()

            del inputs, labels, preds,features
            torch.cuda.empty_cache()

        avg_loss = loss_train / (train_batches * batch_size)
        avg_acc = acc_train / (train_batches * batch_size)

        encoder.eval()
        classifier.eval()

        for i, data in enumerate(dataloader_val):
            inputs, labels = data

            with torch.no_grad():
                if torch.cuda.is_available():
                    inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                features = encoder(inputs)
                outputs = classifier(features.view(features.size(0), -1))

                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                loss_val += loss.data.item()
                acc_val += torch.sum(preds == labels.data).item()

            del inputs, labels, preds
            torch.cuda.empty_cache()

        avg_loss_val = float(loss_val) / (val_batches * batch_size)
        avg_acc_val = float(acc_val) / (val_batches * batch_size)
        print("Epoch-{} | train loss={:.4f} | train acc={:.4f} | val loss={:.4f} | val acc={:.4f} "
              .format(epoch, avg_loss, avg_acc, avg_loss_val, avg_acc_val))

        if avg_acc_val > best_acc:
            print("Val Acc improved from {} to {}, save model".format(best_acc, avg_acc_val))
            best_acc = avg_acc_val
            best_encoder = copy.deepcopy(encoder.state_dict())
            best_classifier = copy.deepcopy(classifier.state_dict())


    print("Best val acc on Source = {:.4f}".format(best_acc))
    print()
    encoder.load_state_dict(best_encoder)
    classifier.load_state_dict(best_classifier)
    torch.save(encoder.state_dict(), save_name + '_source_encoder.pt')
    torch.save(classifier.state_dict(), save_name + '_source_classifier.pt')
    return encoder, classifier


def train_tgt(src_encoder, tgt_encoder, netD, src_data_loader, tgt_data_loader, save_name, num_epochs=10):
    since =  time.time()
    """Train encoder for target domain."""

    # set train state for Dropout and BN layers
    src_encoder.eval()
    tgt_encoder.train()
    netD.train()

    # setup criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer_tgt = optim.SGD(tgt_encoder.parameters(),lr=0.0001, momentum=0.9)
    optimizer_critic = optim.SGD(netD.parameters(),lr=0.001, momentum=0.9)
    len_data_loader = min(len(src_data_loader), len(tgt_data_loader))

    for epoch in range(num_epochs):
        # zip source and target data pair
        data_zip = enumerate(zip(src_data_loader, tgt_data_loader))
        for step, ((images_src, _), (images_tgt, _)) in data_zip:

            # make images variable
            if torch.cuda.is_available():
                images_src, images_tgt = Variable(images_src.cuda()), Variable(images_tgt.cuda())
            else:
                images_src, images_tgt = Variable(images_src), Variable(images_tgt)

            # zero gradients for optimizer
            optimizer_critic.zero_grad()

            # extract and concat features
            feat_src = src_encoder(images_src)
            feat_tgt = tgt_encoder(images_tgt)
            feat_concat = torch.cat((feat_src, feat_tgt), 0)
            feat_concat = feat_concat.view(feat_concat.size(0),-1)

            # predict on discriminator
            pred_concat = netD(feat_concat.detach())

            # prepare real and fake label
            if torch.cuda.is_available():
                label_src = Variable(torch.ones(feat_src.size(0)).long().cuda())
                label_tgt = Variable(torch.zeros(feat_tgt.size(0)).long().cuda())
            else:
                label_src = Variable(torch.ones(feat_src.size(0)).long())
                label_tgt = Variable(torch.zeros(feat_tgt.size(0)).long())

            label_concat = torch.cat((label_src, label_tgt), 0)

            # compute loss for critic
            loss_critic = criterion(pred_concat, label_concat)
            loss_critic.backward()

            # optimize critic
            optimizer_critic.step()

            pred_cls = torch.squeeze(pred_concat.max(1)[1])
            acc = (pred_cls == label_concat).float().mean()

            # zero gradients for optimizer
            optimizer_critic.zero_grad()
            optimizer_tgt.zero_grad()

            # extract and target features
            feat_tgt = tgt_encoder(images_tgt)
            feat_tgt = feat_tgt.view(feat_tgt.size(0),-1)

            # predict on discriminator
            pred_tgt = netD(feat_tgt)

            # prepare real and fake label
            if torch.cuda.is_available():
                label_tgt = Variable(torch.ones(feat_tgt.size(0)).long().cuda())
            else:
                label_tgt = Variable(torch.ones(feat_tgt.size(0)).long())

            # compute loss for target encoder
            loss_tgt = criterion(pred_tgt, label_tgt)
            loss_tgt.backward()

            # optimize target encoder
            optimizer_tgt.step()

            if ((step + 1) % 5 == 0):
                print("Epoch [{}/{}] Step [{}/{}]:"
                      "d_loss={:.5f} g_loss={:.5f} acc={:.5f}"
                      .format(epoch + 1,
                              num_epochs,
                              step + 1,
                              len_data_loader,
                              loss_critic.item(),
                              loss_tgt.item(),
                              acc.item()))

    elapsed_time = time.time() - since
    print("Target Training completed in {:.2f}s".format(elapsed_time))
    torch.save(netD.state_dict(), save_name+"_netD.pt")
    torch.save(tgt_encoder.state_dict(), save_name+"_target_encoder.pt")
    return tgt_encoder


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
            shuffle=True, num_workers=0,   # num_workers = 4 will cause code restart
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

    dataloader_t = {x: torch.utils.data.DataLoader(image_dataset_t[x], batch_size=batch_size,shuffle=True, num_workers=0,)
                    for x in [TRAIN_T, TEST_T]}

    dataset_sizes_tgt = {x: len(image_dataset_t[x]) for x in [TRAIN_T, TEST_T]}

    for x in [TRAIN_T, TEST_T]: print("Loaded {} images under Target {}".format(dataset_sizes_tgt[x], x))

    class_names = image_dataset_t[TRAIN_T].classes
    print("Classes: ", image_dataset_t[TRAIN_T].classes)

    test_acc = []
    test_acc_src =[]
    for iter in range(1,iterations+1):

        save_name = './model_saved/ADDA_' + source_dataset + '_to_' + target_dataset +'_iter'+str(iter)

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
        input_dims = src_classifier[0].in_features
        netD = Discriminator(input_dims=input_dims, hidden_dims=500,output_dims=2).cuda()

        # Shortcut for saving training time
        if source_dataset == 'BOE':
            source_encoder_name = './result/ADDA_BOE_to_OCT_source_encoder_best.pt'
            source_cls_name = './result/ADDA_BOE_to_OCT_source_classifier_best.pt'
        elif source_dataset == 'CELL':
            source_encoder_name = './result/ADDA_OCT_to_BOE_source_encoder_best.pt'
            source_cls_name = './result/ADDA_OCT_to_BOE_source_classifier_best.pt'
        elif source_dataset == 'TMI':
            source_encoder_name = './result/ADDA_TMI_to_BOE_source_encoder_best.pt'
            source_cls_name = './result/ADDA_TMI_to_BOE_source_classifier_best.pt'

        src_encoder.load_state_dict(torch.load(source_encoder_name))
        src_classifier.load_state_dict(torch.load(source_cls_name))

        # Freeze source encoder and classifier parameters
        for param in src_encoder.parameters():
            param.requires_grad = False
        for param in src_classifier.parameters():
            param.requires_grad = False

        # Train target encoder by GAN
        print("Training encoder for target domain...........................")

        # create model
        print('Create target encoder from VGG16 .............................')
        vgg16_t = models.vgg16_bn()
        if use_gpu: vgg16_t.cuda()  # .cuda() will move everything to the GPU side
        tgt_encoder = vgg16_t.features
        tgt_encoder.load_state_dict(src_encoder.state_dict())

        tgt_encoder = train_tgt(src_encoder,tgt_encoder,netD,dataloader_s[TRAIN_S],dataloader_t[TRAIN_T],save_name,epochs)

        print("Test scr_encoder + src_classifier on Source Test dataset")
        test(src_encoder,src_classifier,dataloader_s[TEST_S], dataset_sizes_src[TEST_S])

        print("Test scr_encoder + src_classifier on Target Test dataset")
        test(src_encoder, src_classifier, dataloader_t[TEST_T], dataset_sizes_tgt[TEST_T])

        print("Test tgt_encoder + src_classifier on Target Test dataset")
        tgt_acc = test(tgt_encoder, src_classifier, dataloader_t[TEST_T], dataset_sizes_tgt[TEST_T])
        test_acc.append(tgt_acc)

    print('test_acc=', test_acc)
    test_acc_avg = sum(test_acc) / len(test_acc)
    test_acc_var = statistics.stdev(test_acc)
    print("Average test acc: %.4f" % (test_acc_avg), '| Variance test: %.4f' % (test_acc_var))

    print("The End")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-s', '--source',
                        help='source dataset, choose from [BOE,CELL,TMI]',
                        type=str,
                        choices=['BOE','CELL','TMI'],
                        default='BOE')

    parser.add_argument('-t', '--target',
                        help='target dataset, choose from [BOE,OCT,TMI]',
                        type=str,
                        choices=['BOE','CELL','TMI'],
                        default='TMI')

    parser.add_argument('-e', '--epochs',
                        help='training epochs',
                        type=int,
                        default=30)

    args = parser.parse_args()
    main(args)
