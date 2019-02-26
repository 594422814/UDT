classdef ResponseLossL2 < dagnn.Loss
%ResponseLossL2  layer
%  `ResponseLossL2.forward({r, x*})` computes L2 loss.
%
%  Here the Loss between two matrices is defined as:
%
%     Loss = |r - r_idea|.*|r - r_idea|
%
%  Input 
%       - r    w x h x 1 x N.
%       - x0   N x 2 
%   QiangWang, 2016
% -------------------------------------------------------------------------------------------------------------------------
    properties
        win_size = [3,3];
        sigma = 1;
    end
    properties (Transient)
        y = [];
    end
    methods
        function outputs = forward(obj, inputs, params)
            r1 = inputs{1}; % middle result of Z1
            r2 = inputs{2}; % middle result of Z2
            r3 = inputs{3}; % final label of X
            r4 = inputs{4}; % initial label of X
            
            % original loss
            res_minus = bsxfun(@minus,r3,r4);
            loss = res_minus.*res_minus;
            perSample_loss = sum(sum(sum(loss,1),2),3);
            [~, index] = sort(perSample_loss, 'ascend');
            sample_drop = zeros(1,1,1,size(r1,4));
            % drop unreliable samples
            max_num = round(0.9*size(r1,4));
            sample_drop(index(1:max_num)) = 1;         

            % the motion between X and Z1
            delta1 = bsxfun(@minus,r1,r4);
            % the motion between Z1 and Z2
            delta2 = bsxfun(@minus,r1,r2);
            delta = delta1.*delta1 + delta2.*delta2;
            delta_sum = sum(sum(sum(delta, 1),2),3);
            delta_drop = bsxfun(@times,delta_sum,sample_drop);
            delta_norm = delta_drop./sum(delta_drop(:)) * size(r3,4);
            
            % spatial weight
            spatial = exp(r4);
            weight = bsxfun(@times, delta_norm, spatial);
            % add weight
            loss = loss.*weight;
            
            outputs{1} = sum(loss(:))/size(r3,4);
         
            n = obj.numAveraged ;
            m = n + 1 ;
            obj.average = (n * obj.average + gather(outputs{1})) / m ;
            obj.numAveraged = m ;

        end
        
        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
            r1 = inputs{1}; 
            r2 = inputs{2};
            r3 = inputs{3}; 
            r4 = inputs{4};
            
            % original loss
            res_minus = bsxfun(@minus,r3,r4);
            loss = res_minus.*res_minus;
            perSample_loss = sum(sum(sum(loss,1),2),3);
            [~, index] = sort(perSample_loss, 'ascend');
            sample_drop = zeros(1,1,1,size(r1,4));
            % drop unreliable samples
            max_num = round(0.9*size(r1,4));
            sample_drop(index(1:max_num)) = 1;         

            % the motion between X and Z1
            delta1 = bsxfun(@minus,r1,r4);
            % the motion between Z1 and Z2
            delta2 = bsxfun(@minus,r1,r2);
            delta = delta1.*delta1 + delta2.*delta2;
            delta_sum = sum(sum(sum(delta, 1),2),3);
            delta_drop = bsxfun(@times,delta_sum,sample_drop);
            delta_norm = delta_drop./sum(delta_drop(:)) * size(r3,4);
            
            % spatial weight
            spatial = exp(r4);
            weight = bsxfun(@times, delta_norm, spatial);
                                           
            derInputs{1} = {};
            derInputs{2} = {};
            derInputs{3} = (derOutputs{1}*2/size(r3,4)).*bsxfun(@minus,r3,r4).*weight ;
            derInputs{4} = {};
            derParams = {} ;
        end
        
        function obj = reset(obj)
            obj.y = [] ;
            obj.average = 0 ;
            obj.numAveraged = 0 ;
        end
        
        function obj = ResponseLossL2(varargin)
            obj.load(varargin);
            obj.win_size = obj.win_size;
            obj.sigma = obj.sigma ;
            obj.y = [];
        end

    end

end

