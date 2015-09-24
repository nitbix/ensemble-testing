#!/bin/bash

mkdir -p /local/
pushd /local
wget http://deeplearning.net/data/mnist/mnist.pkl.gz
wget http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar zxvf cifar-10-python.tar.gz
wget http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
tar zxvf cifar-100-python.tar.gz
popd

mkdir -p /local/cifar10
./convert_cifar10.py /local/
mkdir -p /local/cifar100
./convert_cifar100.py /local/
