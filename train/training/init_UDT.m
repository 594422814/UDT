function net = init_UDT(varargin)
% input:
%       -target :125*125*3*n
%       -search :125*125*3*n
%       -delta_xy :n*2
% output:
%       -response :125*125*1*n(test)
rng('default');
rng(0) ;
opts.inputSize = 125;
opts.padding = 2;
[opts] = vl_argparse(opts, varargin) ;
net = dagnn.DagNN() ;

%% meta
net.meta.normalization.imageSize = [opts.inputSize([1,1]),3];
net.meta.normalization.averageImage = reshape(single([123,117,104]),[1,1,3]);
net.meta.arch = 'DCFNet';
net.meta.inputSize = opts.inputSize;
net.meta.padding = opts.padding;

%% network structure
%% target 
convx1_1 = dagnn.Conv('size', [3 3 3 32], 'pad', 0, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
net.addLayer('convx1_1', convx1_1, {'target'}, {'convx1_1'}, {'conv1f', 'conv1b'}) ;
net.addLayer('relux1_1', dagnn.ReLU(), {'convx1_1'}, {'convx1_1_relu'});

convx1_2 = dagnn.Conv('size', [3 3 32 32], 'pad', 0, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
net.addLayer('convx1_2', convx1_2, {'convx1_1_relu'}, {'convx1_2'}, {'conv2f', 'conv2b'}) ;
net.addLayer('norm_x1', dagnn.LRN('param',[5 1 0.0001/5 0.75]), {'convx1_2'}, {'x'});
%% search 1
convx2_1 = dagnn.Conv('size', [3 3 3 32], 'pad', 0, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
net.addLayer('convx2_1', convx2_1, {'search1'}, {'convx2_1'}, {'conv1f', 'conv1b'}) ;
net.addLayer('relux2_1', dagnn.ReLU(), {'convx2_1'}, {'convx2_1_relu'});

convx2_2 = dagnn.Conv('size', [3 3 32 32], 'pad', 0, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
net.addLayer('convx2_2', convx2_2, {'convx2_1_relu'}, {'convx2_2'}, {'conv2f', 'conv2b'}) ;
net.addLayer('norm_x2', dagnn.LRN('param',[5 1 0.0001/5 0.75]), {'convx2_2'}, {'z1'});
%% search 2
conv1s = dagnn.Conv('size', [3 3 3 32], 'pad', 0, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
net.addLayer('conv1s', conv1s, {'search2'}, {'conv1s'}, {'conv1f', 'conv1b'}) ;
net.addLayer('relu1s', dagnn.ReLU(), {'conv1s'}, {'conv1sx'});

conv2s = dagnn.Conv('size', [3 3 32 32], 'pad', 0, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
net.addLayer('conv2s', conv2s, {'conv1sx'}, {'conv2s'}, {'conv2f', 'conv2b'}) ;
net.addLayer('norm_s', dagnn.LRN('param',[5 1 0.0001/5 0.75]), {'conv2s'}, {'z2'});
feature_sz = opts.inputSize([1,1]) - [4, 4];

target_sz = opts.inputSize([1,1])/(1+opts.padding);
sigma = sqrt(prod(target_sz))/10;

trajectory1 = dagnn.DCF('win_size', feature_sz,'sigma', sigma) ;
net.addLayer('trajectory1', trajectory1, {'x','z1', 'label'}, {'s_label1'}) ;
trajectory2 = dagnn.DCF('win_size', feature_sz,'sigma', sigma) ;
net.addLayer('trajectory2', trajectory2, {'z1','z2', 's_label1'}, {'s_label2'}) ;
trajectory3 = dagnn.DCF('win_size', feature_sz,'sigma', sigma) ;
net.addLayer('trajectory3', trajectory3, {'z2','x', 's_label2'}, {'res'}) ;

ResponseLoss = dagnn.ResponseLossL2('win_size', feature_sz, 'sigma', sigma) ;
net.addLayer('ResponseLoss', ResponseLoss, {'s_label1', 's_label2', 'res', 'label'}, 'objective') ;

% Fill in defaul values
net.initParams();

% % copy parameters
% init_net = load(fullfile('../model', 'DCFNet-net-7-125-2.mat'));
% init_net = dagnn.DagNN.fromSimpleNN(init_net.net);
% for i = 1:length(net.params)
%    for j = 1:length(init_net.params)
%        if strcmp(net.params(i).name, init_net.params(j).name)
%            if length(net.params(i).value) ~= length(init_net.params(j).value)
%                error('dismatch channel number!');
%            end
%            disp(net.params(i).name);
%            net.params(i).value = init_net.params(j).value;
%        end
%    end
% end


%% Save
% netStruct = net.saveobj() ;
% save('../model/cnn_dcf.mat', '-v7.3', '-struct', 'netStruct') ;
% clear netStruct ;

end
