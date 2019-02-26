function training_env()
if isunix()
    % addpath('users/wangning/Tracking/DCFNet-master/matconvnet/matlab') 
    addpath('/data3/wangning/matconvnet/matlab')    
end

% vl_compilenn('enableGPU', true);

run('vl_setupnn.m') ;

addpath(fullfile('..','utils'));

fftw('planner','patient');
end