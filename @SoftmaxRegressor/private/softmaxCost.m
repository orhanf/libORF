function [cost, grad] = softmaxCost(obj, theta, nClasses, nFeatures, lambda, x, y)
%
% obj       - caller obj (this)
% theta     - parameter vector
% nClasses  - number of classes
% nFeatures - number of features
% lambda    - weight decay parameter (regularization)
% x         - the n x m data matrix, where each column data(:, i) corresponds to
%               a single test set
% y         - an m x nClasses matrix containing the labels corresponding for the input data
%
% orhanf
%%

    % Unroll the parameters from theta
    theta = reshape(theta, nClasses, nFeatures);
    m     = size(x, 2);
    thetagrad = zeros(nClasses, nFeatures);
    
    M = theta * x;    
    M = bsxfun(@minus, M, max(M, [], 1));   % This is for numberical overflow           
    
    expM     = exp(M);                              % exponential term
    probs    = bsxfun(@rdivide, expM , sum(expM));  % predictions - normalized exponential term
    logProbs = log( probs );                        % log predictions    
    
    % regularization term for cost function
    G = (0.5*lambda).*(norm(theta(:))^2) ./ m;   
    
    % calculate cost function
    cost = -(1/m) .* sum( y(:) .* logProbs(:) ) + G; 
    
    % calculate derivative
    thetagrad = -(1/m) .*  (x * (y - probs)')' + (lambda .* thetagrad);
           
    % Unroll the gradient matrices into a vector for minFunc
    grad = thetagrad(:);

    
end

