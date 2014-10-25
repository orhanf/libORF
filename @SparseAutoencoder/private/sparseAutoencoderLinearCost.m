function [cost,grad] = sparseAutoencoderLinearCost(obj, theta, visibleSize, hiddenSize, ...
                                                        lambda, sparsityParam, beta, data)
% TODO : add function comments here
%
%
%
% orhanf
%%

W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);

nSamples = size(data,2);

%%
%==========================================================================
%       Cost function (forward propagation) calculation with 3 terms
%==========================================================================

a1 = data;                         % a1 is equal to inputs x
z2 = bsxfun(@plus, W1 * a1, b1);   % z2 is weigted sum of a1
a2 = nonLinearity(z2,obj.hActFun); % a2 is sigmoid output of z3
z3 = bsxfun(@plus, W2 * a2, b2);   % z3 is weigted sum of a2
a3 = nonLinearity(z3,3);           % a3 is equal to z3 because of linear decoding

% Squared error term of the cost function
squared_err = ( a3 - data ) .^ 2 ;
J_err = sum(squared_err(:)) / (2*nSamples);

% Regularization term of the cost function
J_reg = (lambda/2) .* ( sum(W1(:).^2) + sum(W2(:).^2));

% Sparsity term of the cost functin
p_hat = mean(a2,2);
p_hat(p_hat==1) = 1-eps;
p_hat(p_hat==0) = eps;

J_spr = beta .* sum(KL(sparsityParam,p_hat));

% Sum all the cost terms
cost = J_err + J_reg + J_spr;

%%
%==========================================================================
%       Gradient-partial derivative (backward propagation) calculation 
%==========================================================================
if nargout>1
    % Compute delta for output layer 
    beta_term = beta .* ( - sparsityParam ./ p_hat  + ( 1-sparsityParam ) ./ ( 1 - p_hat ));

    delta3 = -( data - a3 );
    delta2 = bsxfun(@plus, W2' * delta3, beta_term) .*  dNonLinearity(a2,obj.hActFun);

    % Compute the desired partial derivatives
    W2grad = (1/nSamples) * delta3 * a2' + ( lambda .* W2 );
    b2grad = (1/nSamples) * sum(delta3,2);

    W1grad = (1/nSamples) * delta2 * a1' + ( lambda .* W1 );
    b1grad = (1/nSamples) * sum(delta2,2);


    %-------------------------------------------------------------------
    % After computing the cost and gradient, we will convert the gradients back
    % to a vector format (suitable for minFunc).  Specifically, we will unroll
    % gradient matrices into a vector.
    grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];

end

end

function kldiv = KL(p,q)  
    q(q==0) = eps;
    q(q==1) = 1-eps;
    
    kldiv = p .* log( p./q ) + (1-p) .* log( (1-p) ./ (1-q) );
    kldiv(isnan(kldiv))=0;
end

