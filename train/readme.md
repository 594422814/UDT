
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
```
`run data_preprocessing` for ILSVRC2015 dataset

`run data_preprocessing_LT` for OxUvA long-term tracking dataset

As discussed in the paper, using more data slightly improves the performance

3.Train an unsupervised model.
```
run train_UDT.m;
```
