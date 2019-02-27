
1. Downloading MatConvNet
```
git clone https://github.com/vlfeat/matconvnet.git
```

2. Compiling MatConvNet
```
cd matconvnet
run matlab/vl_compilenn
```

## Training

1.Download the training data. ([**VID**](data))

2.Data Preprocessing in MATLAB.

```matlab
cd training/dataPreprocessing
run data_preprocessing() or data_preprocessing_LT;
```

3.Train an unsupervised model.
```
run train_UDT.m;
```
