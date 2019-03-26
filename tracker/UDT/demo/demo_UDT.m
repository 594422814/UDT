function res = demo_UDT()
addpath(fullfile('..','runfiles'));

init_rect = [129,80,64,78];
img_file = dir('./David/img/*.jpg');
img_file = fullfile('./David/img/', {img_file.name});
subS.init_rect = init_rect;
subS.s_frames = img_file;

param = [];
% use GPU to achieve high speed
param.gpu = false;  
gpuDevice(1);
param.visual = true;

res = run_UDT(subS,0,0,param);
disp(['fps: ',res.fps]);

end