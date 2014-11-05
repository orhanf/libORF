function [pred] = stackedAEPredict(obj, theta, hiddenSize, numClasses, netconfig, data)
                                         
% stackedAEPredict: Takes a trained theta and a test data set,
% and returns the predicted labels for each example.
                                         
% theta: trained weights from the autoencoder
% visibleSize: the number of input units
% hiddenSize:  the number of hidden units *at the 2nd layer*
% numClasses:  the number of categories
% data: Our matrix containing the training data as columns.  So, data(:,i) is the i-th training example. 

% This code should produce the prediction vector 
% pred, where pred(i) is argmax_c P(y(c) | x(i)).
 
%% Unroll theta parameter

% We first extract the part which compute the softmax gradient
softmaxTheta = reshape(theta(1:hiddenSize*numClasses), numClasses, hiddenSize);

% Extract out the "stack"
stack = obj.params2stack(theta(hiddenSize*numClasses+1:end), netconfig);

    depth = numel(stack);
    z = cell(depth+1,1); % weighted sum of inputs to unit i in layer l
    a = cell(depth+1,1); % activations
    a{1} = data;         % a1 is equal to inputs x

    for layer = (1:depth)
      z{layer+1} = stack{layer}.w * a{layer} + repmat(stack{layer}.b, 1, size(a{layer},2));
      a{layer+1} = sigmoid(z{layer+1});
    end

    M = softmaxTheta * a{depth+1};              
    
    % We do not need to normalize 
    [dummy, pred] = max(M);

end


function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end
