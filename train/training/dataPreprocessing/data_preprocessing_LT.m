function data_preprocessing_LT()
%Data Preprocessiong of long-term tracking benchmark.
clear; close all; clc; clear all;

% change the data path to yours. 
data_path1 = fullfile('/data3/wangning/images/dev');
data_path2 = fullfile('/data3/wangning/images/test');
video_num = 0;
frame_num = 0;
seg = [];
video = [];
subset = [];

%% dev set
subset_path = fullfile(data_path1);
video_name = dir(subset_path);
video_name = sort({video_name.name});

for video_id = 3 : numel(video_name)
    disp(['dev/' num2str(video_id - 2, '%04d')]);
    current_path = [data_path1, '/', video_name{video_id}];
    img = dir([current_path, '/', '*.jpeg']);   
    % for every video, we just select first 1000 frames
    for img_id = 1:min(length(img), 1000)
        video.frame_path{img_id} = [current_path, '/', img(img_id).name];
    end
    current_id = video_num + video_id - 2;
    subset.video{current_id} = video;
end
dev_video_num = current_id;

%% test set
subset_path2 = fullfile(data_path2);
video_name = dir(subset_path2);
video_name = sort({video_name.name});

for video_id = 3 : numel(video_name)
    disp(['test/' num2str(video_id - 2, '%04d')]);
    current_path = [data_path2, '/', video_name{video_id}];
    img = dir([current_path, '/', '*.jpeg']);   
    % for every video, we just select first 1000 frames
    for img_id = 1:min(length(img), 1000)
        video.frame_path{img_id} = [current_path, '/', img(img_id).name];
    end
    current_id = dev_video_num + video_id - 2;
    subset.video{current_id} = video;
end

final_video_num = current_id;
% count total frame number and save the path
for video_id = 1:numel(subset.video)
    video = subset.video{video_id};
    for frame_id = 1:numel(video.frame_path)
        seg{video_id}.path{frame_id} = video.frame_path{frame_id};
        frame_num = frame_num + 1;
    end
end

fprintf('total_video_num: %d', final_video_num);
fprintf('total_frame_num: %d', frame_num);
save(fullfile('..', 'long_term_seg'), 'seg');

end