function [ cost, grad ] = stackedAECostSVM(obj, theta, nFeatures, nClasses,...
                                        netconfig, data, Y, C)
%   Inputs
%     obj      : caller object
%     theta    : trained weights from the autoencoders and linearSVM
%     nFeatures: number of features
%     nClasses : number of classes
%     netconfig: configuration of the network produced (internap structure)
%     data     : <nFeatures x nSamples> matrix containing the training data
%     Y        : <nSamples x 1> class label vector
%     C        : regularization parameter (similar to C parameter of libSVM)
%
%   Outputs
%     cost   : objective cost
%     grad   : gradient vector for parameters
%
%   Note that startup code adapted from ufldl, svm cost function and
%   gradient is similar to liblinear.
%
% orhanf - (c) 2013 Orhan FIRAT <orhan.firat@ceng.metu.edu.tr>
%% Unroll svm theta parameter

% We first extract the part which compute the svm gradient
svmTheta = reshape(theta(1:nFeatures*nClasses), [nClasses, nFeatures])';

% Extract out the "stack"
stack = obj.params2stack(theta(nFeatures*nClasses+1:end), netconfig);
stackgrad = cell(size(stack));
for d = 1:numel(stack)
    stackgrad{d}.w = zeros(size(stack{d}.w));
    stackgrad{d}.b = zeros(size(stack{d}.b));
end

nSamples = size(data, 2);

%==========================================================================
%       Cost function (forward propagation) calculation with 1 term
%==========================================================================

    depth = numel(stack);
    z = cell(depth+1,1); % weighted sum of inputs to unit i in layer l
    a = cell(depth+1,1); % activations
    a{1} = data;         % a1 is equal to inputs x

    % feed forward
    for layer = (1:depth)
      z{layer+1} = stack{layer}.w * a{layer} + repmat(stack{layer}.b, 1, size(a{layer},2));
      a{layer+1} = sigmoid(z{layer+1});
    end
        
    % calculate margin
    margin = max(0, 1 - Y .* (a{depth+1}' * svmTheta));
        
    % Error term of the cost function - this is linear svm error
    % term because our output layer is linear svm
    cost = sum((0.5 * sum(svmTheta.^2)) + C*mean(margin.^2));
        
%==========================================================================
%       Gradient-partial derivative (backward propagation) calculation 
%==========================================================================

if nargout > 1 
      
    % This is exactly same as linear svm gradient
    svmThetaGrad = svmTheta - 2*C/nSamples * (a{depth+1} * (margin .* Y));

    d = cell(depth+1,1); % deltas

    % Compute delta for output layer - TRICKY (actually we're taking
    % partial derivatives of the error function with respect to the inputs
    % of the error function which are the activations of the last layer,
    % different part from svmThetaGrad is just here where we were
    % differentiating wrt parameters theta)
    d{depth+1} = - 2*C * svmTheta * (margin .* Y)';
    
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
    grad = [svmThetaGrad(:) ; obj.stack2params(stackgrad)];
end

end


function sigm = sigmoid(x)  
    sigm = 1 ./ (1 + exp(-x));
end
