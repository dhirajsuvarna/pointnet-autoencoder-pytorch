# Creates following files
# 1. train_test_split.json - file containing train and test split of dataset
# 2. classlabels.txt - file containing Class Names or Category Names

# output will be of the following format if the "--labeled_split flag is True"
# {
#   train: {
#       file1:label-A
#       file2:label-B
#       ...
#   }
#   test: {
#       file1:label-A
#       file2:label-B
#       ...
#   }
# }

# output will be of the following format if the "--labeled_split flag is False"
# {
#   train: {
#       file1
#       file2
#       ...
#   }
#   test: {
#       file1
#       file2
#       ...
#   }
# }

# inputs
# 1. dataset path 
# it is assumed that the path will contain labled folders
#   label-A
#       file1
#       file2
#   label-B
#       file1
#       file2
# 2. split percent

import os
import random
import math
import json
import argparse


def create_split(iDirPath, iSplitPercent, ext):
    print(f"Creating split of : {iDirPath}")
    if os.path.isdir(iDirPath):
        files = [f for f in os.listdir(iDirPath) if f.endswith(ext)]

        random.shuffle(files)
        files = [os.path.join(iDirPath, f) for f in files]
        
        numFiles = len(files)
        splitIndex = math.floor(numFiles * (iSplitPercent/100.0))

        trainFiles = files[:splitIndex]
        testFiles = files[splitIndex:]

        return trainFiles, testFiles

def create_split_with_labels(datasetPath, outputFilePath, splitPercent, classLabelFilePath, ext):

    train_test_Dict = {"train": {}, "test": {}}
    classLabels = []

    for labelDir in os.listdir(datasetPath):
        labelDirPath = os.path.join(datasetPath, labelDir)
        if os.path.isdir(labelDirPath):
            trainFiles, testFiles = create_split(labelDirPath, splitPercent, ext)
            train_test_Dict["train"].update({f:labelDir for f in trainFiles})
            train_test_Dict["test"].update({f:labelDir for f in testFiles})

            classLabels.append(labelDir)

    with open(outputFilePath, 'w') as outFile:    
        json.dump(train_test_Dict, outFile, indent=4)

    with open(classLabelFilePath, 'w') as classLabelFile:
        for label in classLabels:
            classLabelFile.write('%s\n' %label)


def create_split_without_label(datasetPath, outputFilePath, splitPercent, ext):
    train_test_Dict = {"train": [], "test": []}

    for labelDir in os.listdir(datasetPath):
        labelDirPath = os.path.join(datasetPath, labelDir)
        if os.path.isdir(labelDirPath):
            trainFiles, testFiles = create_split(labelDirPath, splitPercent, ext)
            train_test_Dict["train"] = train_test_Dict["train"] + trainFiles
            train_test_Dict["test"] = train_test_Dict["test"] + testFiles

    with open(outputFilePath, 'w') as outFile:    
        json.dump(train_test_Dict, outFile, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", required=True, help="Path to the Dataset Folder")
    parser.add_argument("--split_percent", type=int, default=80, help="Split Percentage between Train and Test")
    parser.add_argument("--ext", required=True, help="Extension of the files to be included (.pcd)")
    parser.add_argument("--labeled_split", action='store_true', help="Creates split with labels i.e 'Training_Example:Label'")

    args = parser.parse_args()
    print(f"Arguments:{args}")

    datasetPath = args.dataset
    splitPercent = args.split_percent
    ext = args.ext

    outputFilePath = os.path.join(datasetPath, "train_test_split.json")
    classLabelFilePath = os.path.join(datasetPath, "classlabels.txt")

    if args.labeled_split:
        create_split_with_labels(datasetPath, outputFilePath, splitPercent, classLabelFilePath, ext)
    else:
        create_split_without_label(datasetPath, outputFilePath, splitPercent, ext)
