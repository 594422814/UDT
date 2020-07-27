function net = init_DCFNet3loss(varargin)
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
convx1 = dagnn.Conv('size', [3 3 3 32], 'pad', 0, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
net.addLayer('convx1', convx1, {'target'}, {'convx1'}, {'conv1f', 'conv1b'}) ;
net.addLayer('relux1', dagnn.ReLU(), {'convx1'}, {'convx1_relu'});

convx2 = dagnn.Conv('size', [3 3 32 32], 'pad', 0, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
net.addLayer('convx2', convx2, {'convx1_relu'}, {'convx2'}, {'conv2f', 'conv2b'}) ;
net.addLayer('norm_x', dagnn.LRN('param',[5 1 0.0001/5 0.75]), {'convx2'}, {'x'});
%% search 1
convz1_1 = dagnn.Conv('size', [3 3 3 32], 'pad', 0, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
net.addLayer('convz1_1', convz1_1, {'search1'}, {'convz1_1'}, {'conv1f', 'conv1b'}) ;
net.addLayer('reluz1', dagnn.ReLU(), {'convz1_1'}, {'convz1_1_relu'});

convz1_2 = dagnn.Conv('size', [3 3 32 32], 'pad', 0, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
net.addLayer('convz1_2', convz1_2, {'convz1_1_relu'}, {'convz1_2'}, {'conv2f', 'conv2b'}) ;
net.addLayer('norm_z1', dagnn.LRN('param',[5 1 0.0001/5 0.75]), {'convz1_2'}, {'z1'});
%% search 2
convz2_1 = dagnn.Conv('size', [3 3 3 32], 'pad', 0, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
net.addLayer('convz2_1', convz2_1, {'search2'}, {'convz2_1'}, {'conv1f', 'conv1b'}) ;
net.addLayer('reluz2', dagnn.ReLU(), {'convz2_1'}, {'convz2_1_relu'});

convz2_2 = dagnn.Conv('size', [3 3 32 32], 'pad', 0, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
net.addLayer('convz2_2', convz2_2, {'convz2_1_relu'}, {'convz2_2'}, {'conv2f', 'conv2b'}) ;
net.addLayer('norm_z2', dagnn.LRN('param',[5 1 0.0001/5 0.75]), {'convz2_2'}, {'z2'});
%% search 3
convz3_1 = dagnn.Conv('size', [3 3 3 32], 'pad', 0, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
net.addLayer('convz3_1', convz3_1, {'search3'}, {'convz3_1'}, {'conv1f', 'conv1b'}) ;
net.addLayer('reluz3', dagnn.ReLU(), {'convz3_1'}, {'convz3_1_relu'});

convz3_2 = dagnn.Conv('size', [3 3 32 32], 'pad', 0, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
net.addLayer('convz3_2', convz3_2, {'convz3_1_relu'}, {'convz3_2'}, {'conv2f', 'conv2b'}) ;
net.addLayer('norm_z3', dagnn.LRN('param',[5 1 0.0001/5 0.75]), {'convz3_2'}, {'z3'});

feature_sz = opts.inputSize([1,1]) - [4, 4];
target_sz = opts.inputSize([1,1])/(1+opts.padding);
sigma = sqrt(prod(target_sz))/10;

% forward 
trajectory1 = dagnn.DCF('win_size', feature_sz,'sigma', sigma) ;
net.addLayer('trajectory1', trajectory1, {'x','z1', 'label'}, {'s_label1'}) ;
trajectory2 = dagnn.DCF('win_size', feature_sz,'sigma', sigma) ;
net.addLayer('trajectory2', trajectory2, {'z1','z2', 's_label1'}, {'s_label2'}) ;
trajectory3 = dagnn.DCF('win_size', feature_sz,'sigma', sigma) ;
net.addLayer('trajectory3', trajectory3, {'z2','z3', 's_label2'}, {'s_label3'}) ;
% backward 
back1 = dagnn.DCF('win_size', feature_sz,'sigma', sigma) ;
net.addLayer('back1', back1, {'z1','x', 's_label1'}, {'res1'}) ;
back2 = dagnn.DCF('win_size', feature_sz,'sigma', sigma) ;
net.addLayer('back2', back2, {'z2','x', 's_label2'}, {'res2'}) ;
back3 = dagnn.DCF('win_size', feature_sz,'sigma', sigma) ;
net.addLayer('back3', back3, {'z3','x', 's_label3'}, {'res3'}) ;

ResponseLoss2frame = dagnn.ResponseLoss2frame('win_size', feature_sz, 'sigma', sigma) ;
net.addLayer('ResponseLoss2frame', ResponseLoss2frame, {'s_label1', 'res1', 'label'}, 'objective1') ;

ResponseLoss3frame = dagnn.ResponseLoss3frame('win_size', feature_sz, 'sigma', sigma) ;
net.addLayer('ResponseLoss3frame', ResponseLoss3frame, {'s_label1', 's_label2', 'res2', 'label'}, 'objective2') ;

ResponseLoss4frame = dagnn.ResponseLoss4frame('win_size', feature_sz, 'sigma', sigma) ;
net.addLayer('ResponseLoss4frame', ResponseLoss4frame, {'s_label1', 's_label2', 's_label3', 'res3', 'label'}, 'objective3') ;

% Fill in defaul values
net.initParams();

% copy parameters
% init_net = load(fullfile('../model', 'DCFNet-net-7-125-2.mat'));
% 
% % init_net = load(fullfile('../model', 'SelfCFShortfinal-net-125-2epoch_5.mat'));
% init_net = dagnn.DagNN.fromSimpleNN(init_net.net);

%for i = 1:length(net.params)
%   for j = 1:length(init_net.params)
%       disp(net.params(i).name);
%       if strcmp(net.params(i).name, init_net.params(j).name)
%           if length(net.params(i).value) ~= length(init_net.params(j).value)
%               error('dismatch channel number!');
%           end
%           disp(net.params(i).name);
%           net.params(i).value = init_net.params(j).value;
%       end
%   end
%end

% for i = 1:length(net.params)
%     disp(net.params(i).name);
%     disp(init_net.params(i).name);
%     net.params(i).value = init_net.params(i).value;
% end


end
