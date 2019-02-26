function hf_out = lhs_operation_gpu(hf, samplesf, reg_filter, sample_weights)

% This is the left-hand-side operation in Conjugate Gradient

% Get sizes
num_features = length(hf);
filter_sz = zeros(num_features,2);
for k = 1:num_features
    filter_sz(k,:) = [size(hf{k},1), size(hf{k},2)];
end
[~, k1] = max(filter_sz(:,1));  % Index for the feature block with the largest spatial size
block_inds = 1:num_features;
block_inds(k1) = [];
output_sz = [size(hf{k1},1), 2*size(hf{k1},2)-1];

% Compute the operation corresponding to the data term in the optimization
% (blockwise matrix multiplications)
%implements: A' diag(sample_weights) A f

% sum over all features and feature blocks
sh = sum(bsxfun(@times, samplesf{k1}, hf{k1}), 3);    % assumes the feature with the highest resolution is first
pad_sz = cell(1,1,num_features);
for k = block_inds
    pad_sz{k} = (output_sz - [size(hf{k},1), 2*size(hf{k},2)-1]) / 2;
    
    sh(1+pad_sz{k}(1):end-pad_sz{k}(1), 1+pad_sz{k}(2):end,1,:) = ...
        sh(1+pad_sz{k}(1):end-pad_sz{k}(1), 1+pad_sz{k}(2):end,1,:) + sum(bsxfun(@times, samplesf{k}, hf{k}), 3);
end

% weight all the samples and take conjugate
sh = conj(bsxfun(@times,sample_weights,sh));

% multiply with the transpose
hf_out = cell(1,1,num_features);
hf_out{k1} = conj(sum(bsxfun(@times, sh, samplesf{k1}), 4));
for k = block_inds
    hf_out{k} = conj(sum(bsxfun(@times, sh(1+pad_sz{k}(1):end-pad_sz{k}(1), 1+pad_sz{k}(2):end,1,:), samplesf{k}), 4));
end

% compute the operation corresponding to the regularization term (convolve
% each feature dimension with the DFT of w, and the tramsposed operation)
% add the regularization part
% hf_conv = cell(1,1,num_features);
for k = 1:num_features
    reg_pad = min(size(reg_filter{k},2)-1, size(hf{k},2)-1);
    
    % add part needed for convolution
    hf_conv = cat(2, hf{k}, conj(rot90(hf{k}(:, end-reg_pad:end-1, :), 2)));
    
    % do first convolution
    hf_conv = convn(hf_conv, reg_filter{k});
    
    % do final convolution and put toghether result
    hf_out{k} = hf_out{k} + convn(hf_conv(:,1:end-reg_pad,:), reg_filter{k}, 'valid');
end

end