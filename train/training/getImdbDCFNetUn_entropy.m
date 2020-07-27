function imdb = getImdbDCFNetUn_entropy(varargin)
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
        video = videos{v};fprintf('%3d / %3d \n', v, n_videos);
        
        img_files = video.path;
        img_first = vl_imreadjpeg(img_files(1));
        im_first = uint8(img_first{1}); 
        [H, W, ~] = size(im_first);
        img_num = length(img_files);
        % box:(x1, y1, x2, y2) 
        % bbox position selection
        boxes = zeros(25,4);
        entropy = zeros(25,1);
        boxes(1,:) = round([3*W/12, 3*H/12, 5*W/12, 5*H/12]);
        boxes(2,:) = round([4*W/12, 3*H/12, 6*W/12, 5*H/12]);
        boxes(3,:) = round([5*W/12, 3*H/12, 7*W/12, 5*H/12]);
        boxes(4,:) = round([6*W/12, 3*H/12, 8*W/12, 5*H/12]);
        boxes(5,:) = round([7*W/12, 3*H/12, 9*W/12, 5*H/12]);
        
        boxes(6,:) = round([3*W/12, 4*H/12, 5*W/12, 6*H/12]);
        boxes(7,:) = round([4*W/12, 4*H/12, 6*W/12, 6*H/12]);
        boxes(8,:) = round([5*W/12, 4*H/12, 7*W/12, 6*H/12]);
        boxes(9,:) = round([6*W/12, 4*H/12, 8*W/12, 6*H/12]);
        boxes(10,:) = round([7*W/12, 4*H/12, 9*W/12, 6*H/12]);
        
        boxes(11,:) = round([3*W/12, 5*H/12, 5*W/12, 7*H/12]);
        boxes(12,:) = round([4*W/12, 5*H/12, 6*W/12, 7*H/12]);
        boxes(13,:) = round([5*W/12, 5*H/12, 7*W/12, 7*H/12]);
        boxes(14,:) = round([6*W/12, 5*H/12, 8*W/12, 7*H/12]);
        boxes(15,:) = round([7*W/12, 5*H/12, 9*W/12, 7*H/12]);
        
        boxes(16,:) = round([3*W/12, 6*H/12, 5*W/12, 8*H/12]);
        boxes(17,:) = round([4*W/12, 6*H/12, 6*W/12, 8*H/12]);
        boxes(18,:) = round([5*W/12, 6*H/12, 7*W/12, 8*H/12]);
        boxes(19,:) = round([6*W/12, 6*H/12, 8*W/12, 8*H/12]);
        boxes(20,:) = round([7*W/12, 6*H/12, 9*W/12, 8*H/12]);
        
        boxes(21,:) = round([3*W/12, 7*H/12, 5*W/12, 9*H/12]);
        boxes(22,:) = round([4*W/12, 7*H/12, 6*W/12, 9*H/12]);
        boxes(23,:) = round([5*W/12, 7*H/12, 7*W/12, 9*H/12]);
        boxes(24,:) = round([6*W/12, 7*H/12, 8*W/12, 9*H/12]);
        boxes(25,:) = round([7*W/12, 7*H/12, 9*W/12, 9*H/12]);
        for i = 1:25
            entropy(i) = compute_entropy(im_first, boxes(i,:));
        end
        [~, id] = sort(entropy, 'descend'); 
        % box = round([5*W/12, 5*H/12, 7*W/12, 7*H/12]);
        box = boxes(id(1),:);
        
        target_pos = [box(1)+box(3), box(2)+box(4)]/2;
        target_sz = [box(3)-box(1), box(4)-box(2)];
        tracklet = KCFtracker(img_files, target_pos, target_sz);
        bbox = [tracklet(:,1)-target_sz(1)/2, tracklet(:,2)-target_sz(2)/2, tracklet(:,1)+target_sz(1)/2, tracklet(:,2)+target_sz(2)/2]; 
        
        % bbox = repmat(box, [img_num, 1]);       

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


function patch_entropy = compute_entropy(image, box)

% box: [x1, y1, x2, y2]
pos = [box(1)+box(3), box(2)+box(4)]/2;
sz = ([box(3)-box(1), box(4)-box(2)])*(1); % consider small padding
xs = round(pos(1) + (1:sz(1)) - sz(1)/2);
ys = round(pos(2) + (1:sz(2)) - sz(2)/2);
%extract image
im_crop = image(ys, xs, :);
patch_entropy = entropy(im_crop);

end
