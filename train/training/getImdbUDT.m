function imdb = getImdbUDT(varargin)
opts = [];
opts.dataDir = fullfile('..','data');

opts.visualization = true;
opts.output_size = 125;
opts.padding = 2;
opts = vl_argparse(opts, varargin);

% -------------------------------------------------------------------------
%   full dataset:
%           vid2015: 464873   494211
% -------------------------------------------------------------------------
set_name = {'vid2015'};
num_all_frame = 494211;
%% Be careful!!! It takes a HUGE RAM for fast speed!
imdb.images.set = int8(ones(1, num_all_frame));
imdb.images.set(randperm(num_all_frame, 1000)) = int8(2);
imdb.images.images = zeros(opts.output_size, opts.output_size, 3, num_all_frame, 'uint8');
imdb.images.up_index = zeros(1, num_all_frame, 'double'); % The farthest frame can it touch
now_index = 0;

% -------------------------------------------------------------------------
%                                                                   VID2015
% -------------------------------------------------------------------------
if any(strcmpi(set_name, 'vid2015'))
    disp('VID2015 Data:');
    if exist('vid_2015_seg.mat', 'file')
        load('vid_2015_seg.mat');%% use dataPreprocessing;
    else
        error('You should generate <vid_2015_seg.mat> according ''dataPreprocessing'' at first.')
    end
    videos = seg;
    n_videos = numel(videos);
    for  v = 1:n_videos
        video = videos{v};
        fprintf('%3d / %3d \n', v, n_videos);
        
        img_files = video.path;
        im_frist = vl_imreadjpeg(img_files(1));
        [H, W, ~] = size(im_frist{1});
        img_num = length(img_files);
        % box:(x1, y1, x2, y2) 
        box = round([5*W/12, 5*H/12, 7*W/12, 7*H/12]);
        bbox = repmat(box, [img_num, 1]);       

        im_bank = vl_imreadjpeg(img_files, 'Pack', 'Resize', [H, W], 'numThreads', 32);
        n_frames = size(bbox,1);
        imdb.images.images(:,:,:,now_index+(1:n_frames)) = uint8(...
            imcrop_pad(im_bank{1}, bbox, opts.padding, opts.output_size([1,1])));
        imdb.images.up_index(now_index+(1:n_frames)) = (n_frames:-1:1)-1;
        imdb.images.set(now_index+n_frames) = 4; %should not be selected as x.
        now_index = now_index + n_frames;
    end %%end v
end %%end VID2017

% dataMean = single(mean(imdb.images.images,4));
dataMean(1,1,1:3) = single([123,117,104]);
imdb.images.data_mean(1, 1, 1:3) = dataMean;
imdb.meta.sets = {'train', 'val'} ;

end %%end function
