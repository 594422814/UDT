function data_preprocessing()
% Data Preprocessiong of VID.
clear; close all; clc; clear all;

% change the data path to yours. 
data_path1 = fullfile('/data3/wangning/ILSVRC/ILSVRC2015/Data/VID/train/');
data_path2 = fullfile('/data3/wangning/ILSVRC/ILSVRC2015/Data/VID/val/');
data_path3 = fullfile('/data3/wangning/ILSVRC/ILSVRC2015/Data/VID/test/');

trainset_video_num = 0;
frame_num = 0;
seg = [];
video = [];
subset = [];

%% train set
set_name = dir(fullfile(data_path1, 'ILSVRC2015_VID_train_*'));
set_name = sort({set_name.name});
for set_id = 1:numel(set_name)
    % train set
    subset_path1 = fullfile(data_path1, set_name{set_id});
    video_name = dir(fullfile(subset_path1, 'ILSVRC2015_train_*'));
    video_name = sort({video_name.name});
    
    for video_id = 1:numel(video_name)
        trainset_video_num = trainset_video_num + 1;
        video = [];
        disp([set_name{set_id} '/' num2str(video_id, '%08d')]);
        current_path = [data_path1, set_name{set_id}, '/', video_name{video_id}];
        img = dir([current_path, '/', '*.JPEG']);  
        % for every video, we simply select first 100 frames
        for img_id = 1:min(length(img), 100)
            video.frame_path{img_id} = [current_path, '/', img(img_id).name];
        end
        subset.video{trainset_video_num} = video;
    end
end

%% val set
subset_path2 = fullfile(data_path2);
video_name = dir(subset_path2);
video_name = sort({video_name.name});

for video_id = 3 : numel(video_name)
    disp(['val/' num2str(video_id - 2, '%08d')]);
    current_path = [data_path2, '/', video_name{video_id}];
    img = dir([current_path, '/', '*.JPEG']);   
    for img_id = 1:min(length(img), 100)
        video.frame_path{img_id} = [current_path, '/', img(img_id).name];
    end
    current_id = trainset_video_num + video_id - 2;
    subset.video{current_id} = video;
end
trainval_video_num = current_id;

%% test set
subset_path3 = fullfile(data_path3);
video_name = dir(subset_path3);
video_name = sort({video_name.name});

for video_id = 3 : numel(video_name)
    disp(['test/' num2str(video_id - 2, '%08d')]);
    current_path = [data_path3, '/', video_name{video_id}];
    img = dir([current_path, '/', '*.JPEG']);   
    for img_id = 1:min(length(img), 100)
        video.frame_path{img_id} = [current_path, '/', img(img_id).name];
    end
    current_id = trainval_video_num + video_id - 2;
    subset.video{current_id} = video;
end

video_num = current_id;
% count total frame number and save the path
for video_id = 1:numel(subset.video)
    video = subset.video{video_id};
    for frame_id = 1:numel(video.frame_path)
        seg{video_id}.path{frame_id} = video.frame_path{frame_id};
        frame_num = frame_num + 1;
    end
end

fprintf('total_video_num: %d', video_num);
fprintf('total_frame_num: %d', frame_num);
save(fullfile('..', 'vid_2015_seg'), 'seg');

end