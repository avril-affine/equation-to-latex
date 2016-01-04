#!/bin/sh
conda remove --all --yes -n neural_net_env
conda create --yes -n neural_net_env python=2.7 ipython-notebook
source activate neural_net_env
conda install -y pandas
conda install -y PIL
conda install -y scikit-image
WD=`pwd`
cd ~/anaconda/envs/neural_net_env
pip install -r https://raw.githubusercontent.com/dnouri/nolearn/master/requirements.txt
pip install --upgrade https://github.com/Theano/Theano/archive/master.zip
pip install git+https://github.com/dnouri/nolearn.git@master#egg=nolearn==0.7.git
cd $WD
UNZIPPED=`find data -name mnist_train -type d -maxdepth 1`
if [ "$UNZIPPED" == "" ] 
    then
        unzip data/mnist_train.zip -d data
        rm -rf data/__MACOSX
fi
echo '\n\n\n\n\n\n'
echo 'running environment tests'
python run_nn_test.py