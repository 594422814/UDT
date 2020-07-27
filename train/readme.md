## MatconvNet

1. Download MatConvNet
```
git clone https://github.com/vlfeat/matconvnet.git
```

2. Compile MatConvNet
```
cd matconvnet
run matlab/vl_compilenn
```
3. Change the MatConvNet path in `training_env.m`


## Data

1.Download the training data. (ILSVRC2015 and OxUvA)

2.Data Preprocessing in MATLAB.

```matlab
cd training/dataPreprocessing
```
`run data_preprocessing` for ILSVRC2015 dataset

`run data_preprocessing_LT` for OxUvA long-term tracking dataset

As discussed in the paper, using more data slightly improves the performance

3.Data Preprocessing using the image entropy and HOG-based KCF tracker

Please refer to `getImdbDCFNetUn_entropy.m` and `KCFtracker.m`


## Training for UDT (CVPR version)

1. Train an unsupervised model.
```
run train_UDT.m;
```

## Training for LUDT (IJCV version)

1. Train an unsupervised model.
```
run train_DCFNet_3loss_newdata.m;
```
