function [ cost, grad ] = stackedAECostSM(obj, theta, hiddenSize, numClasses,...
                                        netconfig, lambda, data, labels)
                                         
% stackedAECost: Takes a trained softmaxTheta and a training data set with labels,
% and returns cost and gradient using a stacked autoencoder model. Used for
% finetuning.
                                         
% theta: trained weights from the autoencoder
% hiddenSize:  the number of hidden units *at the 2nd layer*
% numClasses:  the number of categories
% netconfig:   the network configuration of the stack
% lambda:      the weight regularization penalty
% data: Our matrix containing the training data as columns.  So, data(:,i) is the i-th training example. 
% labels: A vector containing labels, where labels(i) is the label for the
% i-th training example


%% Unroll softmaxTheta parameter

% We first extract the part which compute the softmax gradient
softmaxTheta = reshape(theta(1:hiddenSize*numClasses), numClasses, hiddenSize);

% Extract out the "stack"
stack = obj.params2stack(theta(hiddenSize*numClasses+1:end), netconfig);
stackgrad = cell(size(stack));
for d = 1:numel(stack)
    stackgrad{d}.w = zeros(size(stack{d}.w));
    stackgrad{d}.b = zeros(size(stack{d}.b));
end

nSamples = size(data, 2);
groundTruth = full(sparse(labels, 1:nSamples, 1));

%==========================================================================
%       Cost function (forward propagation) calculation with 3 terms
%==========================================================================

    depth = numel(stack);
    z = cell(depth+1,1); % weighted sum of inputs to unit i in layer l
    a = cell(depth+1,1); % activations
    a{1} = data;         % a1 is equal to inputs x

    for layer = (1:depth)
      z{layer+1} = stack{layer}.w * a{layer} + repmat(stack{layer}.b, 1, size(a{layer},2));
      a{layer+1} = sigmoid(z{layer+1});
    end

    M = softmaxTheta * a{depth+1};              % softmax regressor terms
    M = bsxfun(@minus, M, max(M));              % this is for numerical issues
    p = bsxfun(@rdivide, exp(M), sum(exp(M)));  % predictions - normalized exponential term

    
    % Error term of the cost function - this is softmax regressors error
    % term because our output layer is softmax
    J_err = groundTruth(:)' * log(p(:));

    % Regularization term of the cost function - we penalize only softmax
    % parameters for regularization
    J_reg = lambda/2 * sum(softmaxTheta(:) .^ 2);
    
    % Sum all the cost terms
    cost = -1/nSamples * J_err + J_reg;


%==========================================================================
%       Gradient-partial derivative (backward propagation) calculation 
%==========================================================================

if nargout > 1 
      
    % This is exactly same as softmax gradient - input of the softmax layer
    % is activations for the last layer a{depth+1} (it was x in softmax)
    softmaxThetaGrad = -1/nSamples * (groundTruth - p) * a{depth+1}' + lambda * softmaxTheta;

    d = cell(depth+1,1); % deltas

    % Compute delta for output layer (tricky) -(GradJ).*fprime(zn) -> GradJ
    % == theta'(I-P)
    d{depth+1} = -(softmaxTheta' * (groundTruth - p)) .* a{depth+1} .* (1-a{depth+1});

    % Compute deltas for hidden layers
    for layer = (depth:-1:2)
      d{layer} = (stack{layer}.w' * d{layer+1}) .* a{layer} .* (1-a{layer});
    end

    % Compute the desired partial derivatives
    for layer = (depth:-1:1)
      stackgrad{layer}.w = (1/nSamples) * d{layer+1} * a{layer}';
      stackgrad{layer}.b = (1/nSamples) * sum(d{layer+1}, 2);
    end
    
    % Roll gradient vector
    grad = [softmaxThetaGrad(:) ; obj.stack2params(stackgrad)];
end

end


function sigm = sigmoid(x)  
    sigm = 1 ./ (1 + exp(-x));
end
