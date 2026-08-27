import random
seed = 0
random.seed(seed)

import random
import os
import math
from shutil import copyfile
import shutil

data_dir = './dataset_original/BOE_dataset_subject_JPG'

# How many person used for train, val and test
train_ratio = 0.5
val_ratio = 0.25
test_ratio = 1 -train_ratio - val_ratio

dest_folder_train = './BOE_split_by_person/train'
dest_folder_val = './BOE_split_by_person/val'
dest_folder_test = './BOE_split_by_person/test'

classes = os.listdir(data_dir)

for subdir in classes:

    # remove the folder before regenerating the result
    if os.path.exists(os.path.join(dest_folder_train,subdir)):
        shutil.rmtree(os.path.join(dest_folder_train,subdir))

    if os.path.exists(os.path.join(dest_folder_val,subdir)):
        shutil.rmtree(os.path.join(dest_folder_val,subdir))

    if os.path.exists(os.path.join(dest_folder_test,subdir)):
        shutil.rmtree(os.path.join(dest_folder_test,subdir))

    # Create Destination folder for each classes
    if not os.path.exists(os.path.join(dest_folder_train,subdir)):
        os.makedirs(os.path.join(dest_folder_train,subdir))
    if not os.path.exists(os.path.join(dest_folder_val,subdir)):
        os.makedirs(os.path.join(dest_folder_val,subdir))
    if not os.path.exists(os.path.join(dest_folder_test,subdir)):
        os.makedirs(os.path.join(dest_folder_test,subdir))


    sub_data_dir = os.path.join(data_dir,subdir)
    person_count = len(os.listdir(sub_data_dir))

    # Calculate how many person used for train, val, test
    train_count = math.ceil(train_ratio * person_count)
    val_count = math.ceil(val_ratio*person_count)

     # Person Partition
    person_list = os.listdir(sub_data_dir)
    random.shuffle(person_list)
    person_list_train = person_list[0:train_count]
    person_list_val = person_list[train_count:train_count+val_count]
    person_list_test = person_list[train_count+val_count:]
    print("person_list_train =",person_list_train)
    print("person_list_val =", person_list_val)
    print("person_list_test =", person_list_test)

    # Assign the image to train, val, test accordingly
    train_counter, validation_counter, test_counter = 0, 0, 0
    for person in os.listdir(sub_data_dir):
        if person in person_list_train:
            print("Copy {} to train folder".format(person))
            sub_sub_data_dir = os.path.join(sub_data_dir, person)
            filelists = os.listdir(sub_sub_data_dir)
            for filename in filelists:
                # change filename in case same name
                dest_filename = subdir + str(train_counter)+'.jpg'
                dest_folder_train_subdir = os.path.join(dest_folder_train,subdir)
                copyfile(os.path.join(sub_sub_data_dir, filename), os.path.join(dest_folder_train_subdir, dest_filename))
                train_counter += 1

        elif person in person_list_val:
            print("Copy {} to val folder".format(person))
            sub_sub_data_dir = os.path.join(sub_data_dir, person)
            filelists = os.listdir(sub_sub_data_dir)
            for filename in filelists:
                # change filename in case same name
                dest_filename = subdir + str(validation_counter)+'.jpg'
                dest_folder_val_subdir = os.path.join(dest_folder_val, subdir)
                copyfile(os.path.join(sub_sub_data_dir, filename), os.path.join(dest_folder_val_subdir, dest_filename))
                validation_counter += 1
        else:
            print("Copy {} to test folder".format(person))
            sub_sub_data_dir = os.path.join(sub_data_dir, person)
            filelists = os.listdir(sub_sub_data_dir)
            for filename in filelists:
                # change filename in case same name
                dest_filename = subdir + str(test_counter)+'.jpg'
                dest_folder_test_subdir = os.path.join(dest_folder_test, subdir)
                copyfile(os.path.join(sub_sub_data_dir, filename), os.path.join(dest_folder_test_subdir, dest_filename))
                test_counter += 1

    print("Copy {} files to train\{}".format(train_counter,subdir))
    print("Copy {} files to val\{}".format(validation_counter, subdir))
    print("Copy {} files to test\{}".format(test_counter, subdir))

print("End")

