function [ cost, grads ] = neuralNetCostGPU( obj, layers, grads, data, labels, lambda)
%
%
%
%% Unroll parameters

nSamples = size(data, 2);
tiny     = exp(-30);

%==========================================================================
%       Cost function (forward propagation) calculation with 3 terms
%==========================================================================

depth = numel(layers);
z = cell(depth+1,1);      % weighted sum of inputs to unit i in layer l
a = cell(depth+1,1);      % activations
a{1} = data;              % a1 is equal to inputs x
dMasks(1:depth-1,1)= {1}; % dropout masks, necessary for backprop
    

for i = (1:depth)

    % calculate logit
    z{i+1} = bsxfun(@plus, layers{i}.w * a{i}, layers{i}.b);
        
    % apply element-wise nonlinearity
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
        % apply dropout to the intermediate hidden layers        
        a{i+1} = nonLinearity(z{i+1},obj.nnLayers{i+1});
        if (obj.dropOutRatio > 0) 
            dMasks{i} = gpuArray(rand(size(z{i+1})) > obj.dropOutRatio); % speed this up
            a{i+1} = a{i+1} .* dMasks{i};
        end            
    end
end

switch obj.oActFun
    case 'softmax'
        % Error term of the cost function - this is softmax regressors error
        % term because our output layer is softmax
        J_err = -1/nSamples * (labels(:)' * log(a{end}(:) + tiny));
    case {'sigmoid','linear'}
        J_err = 1/(2*nSamples) * sum( (labels(:)-a{end}(:)) .^ 2);
end

% Regularization term of the cost function
J_reg = 0;
for i=1:numel(layers)
   J_reg = J_reg + sum(sum(layers{i}.w.^2)); 
end

% Sum all the cost terms
cost = J_err + (lambda/2 * J_reg);

%==========================================================================
%       Gradient-partial derivative (backward propagation) calculation
%==========================================================================

if nargout > 1
    
    d = cell(depth+1,1); % deltas
    
    % Compute delta for output layer (tricky)
    switch obj.oActFun
        case 'softmax'
            d{depth+1} = a{end} - labels;
        case {'sigmoid','linear'}
            d{depth+1} = (a{end} - labels) .* (a{end} .* (1 - a{end}));
    end
    
    % Compute deltas for hidden layers
    for i = (depth:-1:2)
        d{i} = (layers{i}.w' * d{i+1}) .* dNonLinearity(a{i},obj.nnLayers{i});
        d{i} = d{i} .* dMasks{i-1}; % gradients should flow only through non-dropped units
    end
    
    % Compute the desired partial derivatives
    for i = (depth:-1:1)
        grads{i}.w = (1/nSamples) * d{i+1} * a{i}';
        grads{i}.b = (1/nSamples) * sum(d{i+1}, 2);
    end
    

end


end
