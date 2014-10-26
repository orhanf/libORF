function [ actFinal ] = feedForwardENN( obj, data, preActFlag )
%FEEDFORWARDENN
%   Standard feed forward routine for embedding neural networks
% Inputs
%   obj        : Caller object
%   data       : <nEmbeds,nSamples> discrete data matrix
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
netconfig = obj.netconfig;
layers    = obj.params2stack(theta, netconfig);
nSamples  = size(data,2);    

depth = numel(layers);
z = cell(depth,1);      % weighted sum of inputs to unit i in layer l
a = cell(depth,1);      % activations
a{1} = reshape(layers{1}.w(:,reshape(data,[],1)),...
        obj.nEmbeds*obj.embedSize,nSamples);
              
for i = (2:depth)
    
    if (obj.dropOutRatio > 0) && (i~=1) 
        z{i} = bsxfun(@plus, (layers{i}.w .* obj.dropOutRatio) * a{i-1}, layers{i}.b);
    else
        z{i} = bsxfun(@plus, layers{i}.w * a{i-1}, layers{i}.b);
    end 
        
    if i==depth
        switch obj.oActFun
            case 'softmax'
                % for softmax layer
                z{i} = bsxfun(@minus, z{i}, max(z{i}));             % this is for numerical issues
                a{i} = bsxfun(@rdivide, exp(z{i}), sum(exp(z{i}))); % predictions - normalized exponential term
            case 'sigmoid'
                a{i} = nonLinearity(z{i+1},'sigmoid');
            case 'linear'
                a{i} = z{i};
        end
    else
        a{i} = nonLinearity(z{i},obj.hActFuns(i-1));
    end
end

if preActFlag
    actFinal = a{end-1};
else
    actFinal = a{end};
end
    
    
end


