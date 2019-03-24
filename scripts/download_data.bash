#!/bin/bash

kaggle datasets download -d rahul897/catsdogs
mkdir ../data
unzip ./catsdogs.zip -d ../data/
rm ./catsdogs.zip
unzip ../data/test_set.zip -d ../data/
unzip ../data/training_set.zip -d ../data/
rm ../data/test_set.zip
rm ../data/training_set.zip
