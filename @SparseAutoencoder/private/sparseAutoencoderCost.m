function [cost,grad] = sparseAutoencoderCost(obj, theta, visibleSize, hiddenSize, ...
                                             lambda, lambdaL1, sparsityParam, beta, data)
% obj           : caller object
% visibleSize   : the number of input units 
% hiddenSize    : the number of hidden units 
% lambda        : weight decay parameter (regularization parameter)
% lambdaL1      : L1 penalty parameter (regularization parameter)
% sparsityParam : The desired average activation for the hidden units
%   (choose close to zero as far as you want your model to be sparse)
% beta          : weight of sparsity penalty term
% data          : <n x m> matrix containing the training data.  So, data(:,i) is the i-th training example. 
  
% The input theta is a vector (because minFunc expects the parameters to be a vector). 
% We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this 
% follows the notation convention of the lecture notes. 

% Current implementation assumes there exist only 3 layers, 1st layer is
% the input layer, 2nd layer is the hidden layer (where the number of nodes
% (units-neurons) in this layer is smaller than input unit size) and the
% 3rd layer is the output layer. For further implementation of deep
% learning please wrap a1,z1 ... named parameters in a for loop and
% parametrize respectively. 
% 
%
% orhanf (starter code is adapted from UFLDL class exercises)
%%

% (starter code starts here) 
W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);

% (starter code ends here) ... orhanf

%% ========================================================================
% Compute the cost/optimization objective J_sparse(W,b) for the Sparse Autoencoder,
%  	and the corresponding gradients W1grad, W2grad, b1grad, b2grad.
%
% W1grad, W2grad, b1grad and b2grad should be computed using backpropagation.
% Note that W1grad has the same dimensions as W1, b1grad has the same dimensions
% as b1, etc.  Your code should set W1grad to be the partial derivative of J_sparse(W,b) with
% respect to W1.  I.e., W1grad(i,j) should be the partial derivative of J_sparse(W,b) 
% with respect to the input parameter W1(i,j).  Thus, W1grad should be equal to the term 
% [(1/m) \Delta W^{(1)} + \lambda W^{(1)}] in the last block of pseudo-code in Section 2.2 
% of the lecture notes (and similarly for W2grad, b1grad, b2grad).
% 
% Stated differently, if we were using batch gradient descent to optimize the parameters,
% the gradient descent update to W1 would be W1 := W1 - alpha * W1grad, and similarly for W2, b1, b2. 
% 
% this comment is taken from UFLDL class exercises - orhanf

nSamples = size(data,2);

%==========================================================================
%       Cost function (forward propagation) calculation with 3 terms
%==========================================================================

a1 = data;                         % a1 is equal to inputs x
z2 = bsxfun(@plus, W1 * a1, b1);   % z2 is weigted sum of a1
a2 = nonLinearity(z2,obj.hActFun); % a2 is sigmoid output of z3
z3 = bsxfun(@plus, W2 * a2 ,b2);   % z3 is weigted sum of a2
a3 = nonLinearity(z3,0);           % a3 is equal to h

% Squared error term of the cost function
squared_err = ( a3 - data ) .^ 2 ;
J_err = sum(squared_err(:)) / (2*nSamples);

% Regularization term of the cost function
J_reg = (lambda/2) .* ( sum(W1(:).^2) + sum(W2(:).^2));

% L1 penalty for transition weights
J_regL1 = (lambdaL1) * (sum(abs(W1(:))) + sum(abs(W2(:))) ) 

% Sparsity term of the cost functin
p_hat = mean(a2,2);
J_spr = beta .* sum(KL(sparsityParam,p_hat));

% Sum all the cost terms
cost = J_err + J_reg + J_spr + J_regL1;


%==========================================================================
%       Gradient-partial derivative (backward propagation) calculation 
%==========================================================================
if nargout>1
    % Compute delta for output layer 
    beta_term = beta .* ( - sparsityParam ./ p_hat  + ( 1-sparsityParam ) ./ ( 1 - p_hat ));

    delta3 = -( data - a3 ) .* dNonLinearity(a3,0);
    delta2 = bsxfun(@plus, W2' * delta3, beta_term) .* dNonLinearity(a2,obj.hActFun);

    % Compute the desired partial derivatives
    W2grad = (1/nSamples) * delta3 * a2' + ( lambda .* W2 ) + (lambdaL1 .* sign(W2));
    b2grad = (1/nSamples) * sum(delta3,2);

    W1grad = (1/nSamples) * delta2 * a1' + ( lambda .* W1 ) + (lambdaL1 .* sign(W1));
    b1grad = (1/nSamples) * sum(delta2,2);

    %-------------------------------------------------------------------
    % After computing the cost and gradient, we will convert the gradients back
    % to a vector format (suitable for minFunc).  Specifically, we will unroll
    % gradient matrices into a vector.

    grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];
end

end


%% ------------------------------------------------------------------------
% Here's an implementation of the Kullback-Leibler divergence function, 
% for the costs.  This inputs two (row or column) vectors (say (p1, p2, p3),
% (q1, q2, q3)) and returns (KL(pq1), KL(pq2), KL(pq3)). 
function kldiv = KL(p,q)  
    kldiv = p .* log( p./q ) + (1-p) .* log( (1-p) ./ (1-q) );
    kldiv(isnan(kldiv))=0;
end

