classdef DCF < dagnn.ElementWise
%DCF  layer
%   Discriminant Correlation Filters(DCF)
%
%   QiangWang, 2016
% -------------------------------------------------------------------------------------------------------------------------
    properties
        win_size = [3,3];
        sigma = 1;
    end
    properties (Transient)
       
        lambda = 1e-4;
    end
    methods
        function outputs = forward(obj, inputs, params)
            
            xf = fft2(inputs{1});% target region
            zf = fft2(inputs{2});% search region
            label = gather(inputs{3});
            gt = gaussian_shaped_labels(obj.sigma, obj.win_size);
            
            if size(label, 4) == 1
                [vert_delta, horiz_delta] = find(label==max(label(:)), 1);
                label = circshift(gt, [vert_delta-1, horiz_delta-1]);
                yf = gpuArray(single(fft2(label)));
            else
                for i = 1:size(label,4)
                    labeli = label(:,:,:,i);
                    [vert_delta, horiz_delta] = find(labeli==max(labeli(:)), 1);
                    if(isreal(vert_delta))&&(isreal(horiz_delta))
                        label(:,:,:,i) = circshift(gt, [vert_delta-1, horiz_delta-1]);
                    else
                        label(:,:,:,i) = gt;
                    end
                end
                yf = gpuArray(single(fft2(label)));
            end
            
            xf_conj = conj(xf);
            [h,w,c,~] = size(xf);
            hwc = h*w*c;
            
            lambda_ = gather(obj.lambda);
            obj.lambda = gpuArray(lambda_);
                        
            kxxf = sum(xf .* xf_conj, 3) ./ hwc;
            alphaf = bsxfun(@rdivide, yf, (kxxf + obj.lambda));
            kzxf = sum(zf .* xf_conj, 3) ./ hwc;
            outputs{1} = real(ifft2(alphaf .* kzxf));
        end
        
        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
            
            dldrf = fft2(derOutputs{1}); 
            xf = fft2(inputs{1});% target region
            zf = fft2(inputs{2});% search region
            
            % normlization label
            label = gather(inputs{3});
            gt = gaussian_shaped_labels(obj.sigma, obj.win_size);
            if size(label, 4) == 1
                [vert_delta, horiz_delta] = find(label==max(label(:)), 1);
                label = circshift(gt, [vert_delta-1, horiz_delta-1]);
                y_f = gpuArray(single(fft2(label)));
            else
                for i = 1:size(label,4)
                    labeli = label(:,:,:,i);
                    [vert_delta, horiz_delta] = find(labeli==max(labeli(:)), 1);
                    if(isreal(vert_delta))&&(isreal(horiz_delta))
                        label(:,:,:,i) = circshift(gt, [vert_delta-1, horiz_delta-1]);
                    else
                        label(:,:,:,i) = gt;
                    end
                end
                y_f = gpuArray(single(fft2(label)));
            end
            
            
            xf_conj = conj(xf);
            
            [h,w,c,~] = size(xf);
            hwc = h*w*c;
            
            kxxf = sum(xf .* xf_conj, 3) ./ hwc +obj.lambda;
            
            alphaf = bsxfun(@rdivide, y_f, kxxf);
            
            dldz = real(ifft2(bsxfun(@times,dldrf.*conj(alphaf),xf)))/hwc;
            kzxf = sum(zf .* xf_conj, 3) ./ hwc;
            
            dldx = real(ifft2(bsxfun(@times,conj(dldrf).*alphaf,zf)-...
            2*bsxfun(@times,xf,real(dldrf.*conj(alphaf.*kzxf)./kxxf))))/hwc;
        
            % zf_kxxf = bsxfun(@rdivide, zf, kxxf);
            % dldy =  real(ifft2(sum(bsxfun(@times,dldrf.*conj(zf_kxxf),xf_conj),3)))/hwc;
            % dldy =  real(ifft2(bsxfun(@times,dldrf.*conj(zf_kxxf),xf)))/hwc;
            
            derInputs{1} = dldx;
            derInputs{2} = dldz;
            derInputs{3} = {}; %dldy;
            derParams = {};
        end
        
        function initYF(obj, useGPU, inputs)
            % yf_ = single(fft2(inputs{3}));
            lambda_ = gather(obj.lambda);
            if useGPU
                % obj.yf = gpuArray(yf_);
                obj.lambda = gpuArray(lambda_);
            else
                % obj.yf = yf_;
                obj.lambda = lambda_;
            end
        end
        
        function obj = reset(obj)
            obj.lambda = 1e-4;
        end
        
        function obj = DCF(varargin)
            obj.load(varargin);
            obj.win_size = obj.win_size;
            obj.sigma = obj.sigma ;
        end 
    end
end

function labels = gaussian_shaped_labels(sigma, sz)%kcf
[rs, cs] = ndgrid((1:sz(1)) - floor(sz(1)/2), (1:sz(2)) - floor(sz(2)/2));
labels = exp(-0.5 / sigma^2 * (rs.^2 + cs.^2));
labels = circshift(labels, -floor(sz(1:2) / 2) + 1);
assert(labels(1,1) == 1)
end