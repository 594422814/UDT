function [positions] = KCFtracker(img_files, pos, target_sz)
 
lambda = 1e-4;
padding = 1.8;
output_sigma_factor = 0.1;
interp_factor = 0.02;
cell_size = 4;
features.gray = false;
features.hog = true;
features.hog_orientations = 9;

%if the target is large, lower the resolution, we don't need that much detail
resize_image = (sqrt(prod(target_sz)) >= 100);  %diagonal size >= threshold
if resize_image,
    pos = floor(pos / 2);
    target_sz = floor(target_sz / 2);
end

%window size, taking padding into account
window_sz = floor(target_sz * (1 + padding));

%create regression labels, gaussian shaped, with a bandwidth
%proportional to target size
output_sigma = sqrt(prod(target_sz)) * output_sigma_factor*2 / cell_size;
y = gaussian_shaped_labels(output_sigma, floor(window_sz / cell_size));
yf = fft2(y);

%store pre-computed cosine window
cos_window = hann(size(yf,1)) * hann(size(yf,2))';	

%note: variables ending with 'f' are in the Fourier domain.
positions = zeros(numel(img_files), 2);  %to calculate precision

for frame = 1:numel(img_files),
    %load image
    im = imread([img_files{frame}]);

    if size(im,3) > 1,
        im = rgb2gray(im);
    end
    if resize_image,
        im = imresize(im, 0.5);
    end

    if frame > 1,
        % patch = get_subwindow(im, roi_pos, window_sz);
        patch = get_subwindow(im, pos, window_sz);
        zf = fft2(get_features(patch, features, cell_size, cos_window));

        %calculate response of the classifier at all shifts
        kzf = linear_correlation(zf, model_xf);
        response = real(ifft2(model_alphaf .* kzf));  %equation for fast detection

        [vert_delta, horiz_delta] = find(response == max(response(:)), 1);

        if vert_delta > size(zf,1) / 2,  %wrap around to negative half-space of vertical axis
            vert_delta = vert_delta - size(zf,1);
        end
        if horiz_delta > size(zf,2) / 2,  %same for horizontal axis
            horiz_delta = horiz_delta - size(zf,2);
        end
        pos = pos + cell_size * [vert_delta - 1, horiz_delta - 1] ;        
    end

    %obtain a subwindow for training at newly estimated target position
    patch = get_subwindow(im, pos, window_sz);
    xf = fft2(get_features(patch, features, cell_size, cos_window));       

    %Kernel Ridge Regression, calculate alphas (in Fourier domain)
    kf = linear_correlation(xf, xf);
    alphaf = yf ./ (kf + lambda);   %equation for fast training

    if frame == 1,  %first frame, train with a single image
        model_alphaf = alphaf;
        model_xf = xf;
    else
        %subsequent frames, interpolate model
        model_alphaf = (1 - interp_factor) * model_alphaf + interp_factor * alphaf;
        model_xf = (1 - interp_factor) * model_xf + interp_factor * xf;
    end

    %save position and timing
    positions(frame,:) = pos;
end

if resize_image,
    positions = positions * 2;
end
    

end


function kf = linear_correlation(xf, yf)
	kf = sum(xf .* conj(yf), 3) / numel(xf);
end


function labels = gaussian_shaped_labels(sigma, sz)

%evaluate a Gaussian with the peak at the center element
[rs, cs] = ndgrid((1:sz(1)) - floor(sz(1)/2), (1:sz(2)) - floor(sz(2)/2));
labels = exp(-0.5 / sigma^2 * (rs.^2 + cs.^2));

%move the peak to the top-left, with wrap-around
labels = circshift(labels, -floor(sz(1:2) / 2) + 1);

%sanity check: make sure it's really at top-left
assert(labels(1,1) == 1)
end


function x = get_features(im, features, cell_size, cos_window)

if features.hog,
    %HOG features, from Piotr's Toolbox
    x = double(fhog(single(im) / 255, cell_size, features.hog_orientations));
    x(:,:,end) = [];  %remove all-zeros channel ("truncation feature")
end

if features.gray,
    %gray-level (scalar feature)
    x = double(im) / 255;

    x = x - mean(x(:));
end

%process with cosine window if needed
if ~isempty(cos_window),
    x = bsxfun(@times, x, cos_window);
end
end


function out = get_subwindow(im, pos, sz)

if isscalar(sz),  %square sub-window
    sz = [sz, sz];
end

xs = floor(pos(2)) + (1:sz(2)) - floor(sz(2)/2);
ys = floor(pos(1)) + (1:sz(1)) - floor(sz(1)/2);

%check for out-of-bounds coordinates, and set them to the values at
%the borders
xs(xs < 1) = 1;
ys(ys < 1) = 1;
xs(xs > size(im,2)) = size(im,2);
ys(ys > size(im,1)) = size(im,1);

%extract image
out = im(ys, xs, :);

end
