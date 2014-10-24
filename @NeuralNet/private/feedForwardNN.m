function [ actFinal ] = feedForwardNN( obj, data, preActFlag )
%FEEDFORWARDNN
%   Standard feed forward routine for neural networks
%
%   preActFlag : return penultimate layer activations rather than topmost
%                   layer activations
%
% orhanf
%%

if nargin<3
    preActFlag = false;
end

% extract necessary information from caller object
theta     = obj.nnOptTheta;
layers    = obj.params2stack(theta);
    
depth = numel(layers);
z = cell(depth+1,1);      % weighted sum of inputs to unit i in layer l
a = cell(depth+1,1);      % activations
a{1} = data;              % a1 is equal to inputs x
              
for i = (1:depth)
    
    % no dropout regularization for input and output layers
    if (obj.dropOutRatio > 0) && (i~=1) && (i~=depth) 
        z{i+1} = bsxfun(@plus, (layers{i}.w .* obj.dropOutRatio) * a{i}, layers{i}.b);
    else
        z{i+1} = bsxfun(@plus, layers{i}.w * a{i}, layers{i}.b);
    end 
        
    if i==depth
        switch obj.oActFun
            case 'softmax'
                % for softmax layer
                z{i+1} = bsxfun(@minus, z{i+1}, max(z{i+1}));             % this is for numerical issues
                a{i+1} = bsxfun(@rdivide, exp(z{i+1}), sum(exp(z{i+1}))); % predictions - normalized exponential term
            case 'sigmoid'
                a{i+1} = nonLinearity(z{i+1},'sigmoid');
            case 'linear'
                a{i+1} = z{i+1};
        end
    else
        a{i+1} = nonLinearity(z{i+1},obj.nnLayers{i+1});
    end
end

if preActFlag
    actFinal = a{end-1};
else
    actFinal = a{end};
end
    
    
end


