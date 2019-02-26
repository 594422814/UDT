function [net, info] = train_UDT(varargin)
%Training unsupervised deep tracker
training_env();
opts.inputSize = 125;
opts.padding = 2; % padding size for crop
opts.gpus = 3;
[opts] = vl_argparse(opts, varargin) ;

opts.expDir = fullfile('..', 'data', sprintf('UDT-net-%d-%1.1f', opts.inputSize, opts.padding)) ;
opts.imdbPath = fullfile('../../DCFNet3.0/data', sprintf('imdb-vid2015-unsuperShort-%d-%1.1f', opts.inputSize, opts.padding), 'imdb.mat');

trainOpts.momentum = 0.9;
trainOpts.weightDecay = 0.0005;
trainOpts.numEpochs = 50;
trainOpts.learningRate = logspace(-2, -5, trainOpts.numEpochs) ; % from SiameseFC
trainOpts.batchSize = 32;
trainOpts.gpus = [opts.gpus]; %only support single gpu
opts.train = trainOpts;

if ~isfield(opts.train, 'gpus')
    opts.train.gpus = [];
elseif numel(opts.train.gpus) ~=0
    gpuDevice(opts.train.gpus);
end

global r;
win_sz = opts.inputSize([1,1]) - [4, 4];
target_sz = opts.inputSize([1,1])/(1+opts.padding);
sigma = sqrt(prod(target_sz))/10;
r = gaussian_shaped_labels(sigma, win_sz);


% --------------------------------------------------------------------
%                                                 Prepare net and data
% --------------------------------------------------------------------
net = init_UDT('inputSize', opts.inputSize, 'padding', opts.padding);

if exist(opts.imdbPath, 'file')
    imdb = load(opts.imdbPath) ;
else
    imdb = getImdbUDT('output_size', opts.inputSize,...
        'padding', opts.padding) ;
    if ~exist(fileparts(opts.imdbPath), 'dir')
        mkdir(fileparts(opts.imdbPath));
    end
    save(opts.imdbPath, '-v7.3', '-struct', 'imdb') ;
end

% --------------------------------------------------------------------
%                                                                Train
% --------------------------------------------------------------------
[net, info] = cnn_train_dag(net, imdb, getBatch(opts), ...
    'expDir', opts.expDir, opts.train, 'val', find(imdb.images.set == 2)) ;

transition_net('expDir', opts.expDir);
transition_net_multi('expDir', opts.expDir);
end

% --------------------------------------------------------------------
function fn = getBatch(opts)
% --------------------------------------------------------------------
bopts = struct('numGpus', numel(opts.train.gpus), 'batchSize', opts.train.batchSize, 'maxStep', 10) ;
fn = @(x,y) getDagNNBatch(bopts,x,y) ;
end

function inputs = getDagNNBatch(opts, imdb, batch)
% --------------------------------------------------------------------
rand_next1 = randi([1, opts.maxStep], size(batch));
rand_next1 = min(rand_next1, imdb.images.up_index(batch));
rand_next2 = randi([1, opts.maxStep], size(batch));
rand_next2 = min(rand_next2, imdb.images.up_index(batch));

if opts.numGpus > 0
    target = gpuArray(single(imdb.images.images(:,:,:,batch)));
    search1 = gpuArray(single(imdb.images.images(:,:,:,batch+rand_next1)));
    search2 = gpuArray(single(imdb.images.images(:,:,:,batch+rand_next2)));
else
    target = single(imdb.images.images(:,:,:,batch));
    search1 = single(imdb.images.images(:,:,:,batch+rand_next1));
    search2 = single(imdb.images.images(:,:,:,batch+rand_next2));
end
target = bsxfun(@minus, target, imdb.images.data_mean);
search1 = bsxfun(@minus, search1, imdb.images.data_mean);
search2 = bsxfun(@minus, search2, imdb.images.data_mean);
global r;
inputs = {'target', target, 'search1', search1, 'search2', search2, 'label', r} ;
end


function labels = gaussian_shaped_labels(sigma, sz)%kcf
[rs, cs] = ndgrid((1:sz(1)) - floor(sz(1)/2), (1:sz(2)) - floor(sz(2)/2));
labels = exp(-0.5 / sigma^2 * (rs.^2 + cs.^2));
labels = circshift(labels, -floor(sz(1:2) / 2) + 1);
assert(labels(1,1) == 1)
end
