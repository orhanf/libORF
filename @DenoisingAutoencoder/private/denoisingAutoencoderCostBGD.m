function [ cost, grad ] = denoisingAutoencoderCostBGD( obj, theta, x )
%DENOISINGAUTOENCODERCOST - Calculates cost and gradients of a denoising
% autoencoder cost with an L2 weight decay on transition weights (biases 
% are not penalized). Two noise sources can be applied, additive gaussian
% noise or a dropping mask, with parameters obj.nu and obj.drop
% respectively. Error is measured using either mean squared error or cross
% entropy. Tied weights are allowed, in case of tied weights W2 and W2grad
% will be empty.
%
% This function is used for batch gradient descent with an external
% minimization function, eg. minFunc.
%
% Inputs  
%   obj   : caller object
%   theta : parameter vector to be updated
%   x     : uncorrupted input mini-batch
%
% Outputs
%   cost : scaler indicating denoising autoencoder cost
%   grad : gradient vector
%
% orhanf
%%

% extract parameters
W1 = reshape(theta(1:obj.hiddenSize*obj.visibleSize), obj.hiddenSize, obj.visibleSize);
W2 = reshape(theta(obj.hiddenSize*obj.visibleSize+1:2*obj.hiddenSize*obj.visibleSize), obj.visibleSize, obj.hiddenSize);
b1 = theta(2*obj.hiddenSize*obj.visibleSize+1:2*obj.hiddenSize*obj.visibleSize+obj.hiddenSize);
b2 = theta(2*obj.hiddenSize*obj.visibleSize+obj.hiddenSize+1:end);

% get helpers
nSamples = size(x,2);

% save clean input
x0 = x;

% add noise - additive gaussian noise to inputs
x = x + (obj.nu * randn(size(x)));

% apply drop mask - set some inputs to zero
x = binornd(1,1-obj.drop,size(x)) .* x;

%==========================================================================
%       Cost function (forward propagation) calculation with 3 terms
%==========================================================================

z2 = bsxfun(@plus, W1*x, b1);           % pre-activation of hidden
a2 = nonLinearity(z2, obj.hActFun);             % hidden representation

if obj.tiedWeights
    z3 = W1' * a2 + repmat(b2, 1, nSamples); % pre-activation of output 
else
    z3 = W2 * a2 + repmat(b2, 1, nSamples);  % pre-activation of output 
end

a3 = nonLinearity(z3, obj.vActFun);             % reconstruction     

if obj.errFun==0        % Squared error term of the cost function    
    squared_err = ( a3 - x0 ) .^ 2 ;
    J_err = sum(squared_err(:)) / (2*nSamples);
else                    % Cross entropy for the cost function 
    J_err = -mean(sum(x0 .* log(max(a3,eps)) + (1-x0) .* log(max(1-a3,eps))));
end
    
% Regularization term of the cost function
J_reg = (obj.lambda/2) .* ( sum(W1(:).^2) + sum(W2(:).^2));

% Sparsity term of the cost functin
rho_hat = mean(a2,2);
J_spr = obj.beta .* sum(KL(obj.rho, rho_hat));

% Sum all the cost terms
cost = J_err + J_reg + J_spr;


%==========================================================================
%       Gradient-partial derivative (backward propagation) calculation 
%==========================================================================
if nargout>1
    % Compute delta for output layer     
    if obj.errFun == 1 || obj.vActFun == 3
        delta3 = -( x0 - a3 );    
    else        
        delta3 = -( x0 - a3 ) .* dNonLinearity(a3, obj.vActFun);    
    end 

    % beta term for sparsity regularization
    beta_term = obj.beta .* ( - obj.rho ./ rho_hat  + ( 1-obj.rho ) ./ ( 1 - rho_hat ));
    
    % Compute the desired partial derivatives
    if obj.tiedWeights
        delta2 = bsxfun(@plus, W1 * delta3, beta_term) .* dNonLinearity(a2, obj.hActFun);
        W1grad = (1/nSamples) * ((a2 * delta3') + (delta2 * x' )) + (obj.lambda .* W1);
        W2grad = [];
    else
        delta2 = bsxfun(@plus, W2' * delta3, beta_term) .* dNonLinearity(a2, obj.hActFun);
        W1grad = (1/nSamples) * delta2 * x' + ( obj.lambda .* W1 );    
        W2grad = (1/nSamples) * delta3 * a2' + ( obj.lambda .* W2 );
    end
    
    b1grad = (1/nSamples) * sum(delta2,2);
    b2grad = (1/nSamples) * sum(delta3,2);
                
    grad = [W1grad(:); W2grad(:); b1grad(:); b2grad(:)];
    
end


end

function kldiv = KL(p,q)  
    kldiv = p .* log( p./q ) + (1-p) .* log( (1-p) ./ (1-q) );
    kldiv(isnan(kldiv))=0;
end