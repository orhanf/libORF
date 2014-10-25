function [ cost, grad ] = contractiveAutoencoderCostSGD( obj, x )
%CONTRACTIVEAUTOENCODERCOST - Calculates cost and gradients of a contractive
% autoencoder cost with an L2 weight decay on transition weights (biases 
% are not penalized). 
%  Error is measured using either mean squared error or cross
% entropy. Tied weights are allowed, in case of tied weights W2 and W2grad
% will be empty.
%
% This function is used in the inner loop of mini-batch stochastic gradient
% descent directly.
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
%
%   TODO : implement add bias option  
%   TODO : improve speed by vectorizing for-loops
%
% orhanf
%%

% get helpers
nSamples = size(x,2);

%==========================================================================
%       Cost function (forward propagation) calculation with 3 terms
%==========================================================================

z2 = bsxfun(@plus, obj.W1 * x, obj.b1);   % pre-activation of hidden
a2 = nonLinearity(z2, obj.hActFun);       % hidden representation
z3 = bsxfun(@plus, obj.W1' * a2, obj.b2); % pre-activation of output 
a3 = nonLinearity(z3, obj.vActFun);       % reconstruction     

if obj.errFun==0        % Squared error term of the cost function    
    squared_err = ( a3 - x ) .^ 2 ;
    J_err = sum(squared_err(:)) / (2*nSamples);
else                    % Cross entropy for the cost function 
    J_err = -mean(sum(x .* log(max(a3,eps)) + (1-x) .* log(max(1-a3,eps))));
end
    
% Regularization term of the cost function
J_reg = (obj.lambda/2) .* sum(obj.W1(:).^2);

% Frobenius norm error term 
J_cont = mean(sum(bsxfun(@times,(a2 .* (1-a2)).^2 , sum(obj.W1.^2,2))));

% Sum all the cost terms
cost = J_err + J_reg + J_cont;


%==========================================================================
%       Gradient-partial derivative (backward propagation) calculation 
%==========================================================================
if nargout>1
    % Compute delta for output layer     
    if obj.errFun == 1 || obj.vActFun == 3
        delta3 = -( x - a3 );    
    else        
        delta3 = -( x - a3 ) .* dNonLinearity(a3, obj.vActFun);    
    end
    
    % Compute the desired partial derivatives
    delta2 = (obj.W1 * delta3) .* dNonLinearity(a2, obj.hActFun);
    W1grad = (1/nSamples) * ((a2 * delta3') + (delta2 * x' )) + (obj.lambda .* obj.W1);
    
    % Compute gradient of the contractive term - HIGHLY NON-OPTIMIZED------
    c1 = zeros(size(obj.W1,1),size(obj.W1,2),nSamples);    
    W1SumOfSqr = repmat(sum(obj.W1.^2,2),1,obj.visibleSize);
    for sampleIdx = 1:nSamples
        c1(:,:,sampleIdx) = bsxfun(@times, (obj.W1 + ((1 - 2 * a2(:,sampleIdx)) * x(:,sampleIdx)'  .* W1SumOfSqr )),...
            2 * a2(:,sampleIdx).^2 .* (1 - a2(:,sampleIdx)).^2);
    end     
    c = mean(c1,3);    
    W1grad = W1grad + obj.contrLevel * c;  
    %----------------------------------------------------------------------
    
%     b1grad = (1/nSamples) * sum(delta2,2);
%     b2grad = (1/nSamples) * sum(delta3,2);
                
    grad.W1grad = W1grad;
    
% TODO add bias option    
%     grad.b1grad = b1grad;
%     grad.b2grad = b2grad;

end


end

