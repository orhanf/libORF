function [ cost, grad ] = embeddingNeuralNetCost( obj, theta, netconfig, data, labels, lambda)
%
%
%
%% Unroll parameters

% Extract out the "layers"
layers = obj.params2stack(theta, netconfig);

stackgrad = cell(size(layers));
for d = 1:numel(layers)
    stackgrad{d}.w = zeros(size(layers{d}.w));
    stackgrad{d}.b = zeros(size(layers{d}.b));
end

nSamples = size(data, 2);
tiny     = exp(-30);

%==========================================================================
%       Cost function (forward propagation) calculation with 3 terms
%==========================================================================

depth = numel(layers);
dMasks(1:depth-1,1)= {1}; % dropout masks, necessary for backprop
z = cell(depth,1);      % weighted sum of inputs to unit i in layer l
a = cell(depth,1);      % activations

% THIS PART IS CRUCIAL FOR EMBEDDING NEURAL NET----------------------------
% use look-up table (embedding layer weights) and map input words to the
% embedding space, then concatenate all embeddings. This will construct the
% input to our forthcoming neural network, input dimension to the neural
% net part is <embedSize*nEmbeds, nSamples>

a{1} = reshape(layers{1}.w(:,reshape(data,[],1)),...
        obj.nEmbeds*obj.embedSize,nSamples);
%--------------------------------------------------------------------------    

for i = (2:depth)

    % calculate logit
    z{i} = bsxfun(@plus, layers{i}.w * a{i-1}, layers{i}.b);
        
    % apply element-wise nonlinearity
    if i==depth
        switch obj.oActFun
            case 'softmax'
                % for softmax layer
                z{i} = bsxfun(@minus, z{i}, max(z{i}));             % this is for numerical issues
                a{i} = bsxfun(@rdivide, exp(z{i}), sum(exp(z{i}))); % predictions - normalized exponential term
            case 'sigmoid'
                a{i} = nonLinearity(z{i},'sigmoid');
            case 'linear'
                a{i} = z{i};
        end
    else
        % apply dropout to the intermediate hidden layers        
        a{i} = nonLinearity(z{i},obj.hActFuns(i-1));
        if (obj.dropOutRatio > 0) 
            dMasks{i} = rand(size(z{i})) > obj.dropOutRatio; 
            a{i} = a{i} .* dMasks{i};
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
for i=2:depth,    J_reg = J_reg + sum(layers{i}.w(:) .^ 2 ); end
J_reg = lambda/2 * J_reg;

% Sum all the cost terms
cost = J_err + J_reg;

%==========================================================================
%       Gradient-partial derivative (backward propagation) calculation
%==========================================================================

if nargout > 1
    
    d = cell(depth,1); % deltas
    
    % Compute delta for output layer (tricky)
    switch obj.oActFun
        case 'softmax'
            d{depth} = a{end} - labels;
        case {'sigmoid','linear'}
            d{depth} = (a{end} - labels) .* (a{end} .* (1 - a{end}));
    end
    
    % Compute deltas for hidden layers
    for i = (depth:-1:3)
        d{i-1} = (layers{i}.w' * d{i}) .* dNonLinearity(a{i-1},obj.hActFuns(i-2));
        d{i-1} = d{i-1} .* dMasks{i-1}; % gradients should flow only through non-dropped units
    end
    
    % Compute delta for embedding-hidden layer
    d{1} = layers{2}.w' * d{2};
    
    % Now calculate embedding weight gradients
    M = eye(size(obj.trainLabels,1));
    for i = 1:obj.nEmbeds
       stackgrad{1}.w = stackgrad{1}.w + ...
         (M(:, data(i, :)) * (d{1}(1 + (i - 1) * obj.embedSize : i * obj.embedSize, :)'))';
    end
    stackgrad{1}.w = stackgrad{1}.w / nSamples;


    % Compute the desired partial derivatives
    for i = (depth:-1:2)
        stackgrad{i}.w = (1/nSamples) * d{i} * a{i-1}';
        stackgrad{i}.b = (1/nSamples) * sum(d{i}, 2);
    end
    
    % Roll gradient vector
    grad = obj.stack2params(stackgrad);
end


end
