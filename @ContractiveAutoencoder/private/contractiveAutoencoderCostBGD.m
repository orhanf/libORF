function [ cost, grad ] = contractiveAutoencoderCostBGD( obj, theta, x )
%CONTRACTIVEAUTOENCODERCOST - Calculates cost and gradients of a contractive
% autoencoder cost with an L2 weight decay on transition weights (biases 
% are not penalized). 
%  Error is measured using either mean squared error or cross
% entropy. Tied weights are allowed, in case of tied weights W2 and W2grad
% will be empty.
%
% This function is used in the batch gradient descent scheme, when user
% supplies a minimization function.
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

% extract parameters
W1 = reshape(theta(1:obj.hiddenSize*obj.visibleSize), obj.hiddenSize, obj.visibleSize);

% TODO add bias option
if obj.addBias 
    b1 = theta(obj.hiddenSize*obj.visibleSize+1:obj.hiddenSize*obj.visibleSize+obj.hiddenSize);
    b2 = theta(obj.hiddenSize*obj.visibleSize+obj.hiddenSize+1:end);
else
    b1 = zeros(obj.hiddenSize,1);
    b2 = zeros(obj.visibleSize,1);
end

% get helpers
nSamples  = size(x,2);

%==========================================================================
%       Cost function (forward propagation) calculation with 3 terms
%==========================================================================

z2 = W1 * x + repmat(b1, 1,nSamples);    % pre-activation of hidden
a2 = nonLinearity(z2, obj.hActFun);      % hidden representation
z3 = W1' * a2 + repmat(b2, 1, nSamples); % pre-activation of output
a3 = nonLinearity(z3, obj.vActFun);      % reconstruction

if obj.errFun==0        % Squared error term of the cost function
    squared_err = ( a3 - x ) .^ 2 ;
    J_err = sum(squared_err(:)) / (2*nSamples);
else                    % Cross entropy for the cost function
    J_err = -mean(sum(x .* log(max(a3,eps)) + (1-x) .* log(max(1-a3,eps))));
end

% Regularization term of the cost function
J_reg = (obj.lambda/2) .* ( sum(W1(:).^2));


% Frobenius norm error term -----------------------------------------------
contC = zeros(nSamples,1);
for sampleIdx = 1:nSamples
    for hidIdx = 1:obj.hiddenSize
        currCost = (a2(hidIdx,sampleIdx) * (1-a2(hidIdx,sampleIdx))).^2 * sum(W1(hidIdx,:).^2);
        contC(sampleIdx) = contC(sampleIdx) + currCost;
    end
end
J_cont = mean(contC);
%--------------------------------------------------------------------------

% Sum all the cost terms
cost = J_err + J_reg + J_cont;

clear('contC','squared_err','z2','z3');

%==========================================================================
%       Gradient-partial derivative (backward propagation) calculation
%==========================================================================
if nargout>1
    
    c1 = zeros(size(W1,1),size(W1,2),nSamples); 
    
    % Compute delta for output layer
    if obj.errFun == 1 || obj.vActFun == 3
        delta3 = -( x - a3 );
    else
        delta3 = -( x - a3 ) .* dNonLinearity(a3, obj.vActFun);
    end
    
    % Compute the desired partial derivatives
    delta2 = (W1 * delta3) .* dNonLinearity(a2, obj.hActFun);
    W1grad = (1/nSamples) * ((a2 * delta3') + (delta2 * x' )) + (obj.lambda .* W1);
    
    clear('a3','theta')
    
    % Compute gradient of the contractive term - HIGHLY NON-OPTIMIZED------    
%     tt = zeros(size(W1,1),size(W1,2),nSamples); 
%     rr = zeros(size(W1,1),size(W1,2),nSamples); 
    W1SumOfSqr = repmat(sum(obj.W1.^2,2),1,obj.visibleSize);
    for sampleIdx = 1:nSamples
        rr = (W1 + ((1 - 2 * a2(:,sampleIdx)) * x(:,sampleIdx)'  .* W1SumOfSqr ) );
        tt = repmat(2 * a2(:,sampleIdx).^2 .* (1 - a2(:,sampleIdx)).^2,1,obj.visibleSize);        
        c1(:,:,sampleIdx) = tt .* rr;
    end    
%     t = mean(tt,3);    
%     r = mean(rr,3);    
    c = mean(c1,3);    
       
%     dd = zeros(size(W1,1),size(W1,2),nSamples); 
%     for sampleIdx = 1:nSamples
%         dd(:,:,sampleIdx) = (W1 + ((1 - 2 * a2(:,sampleIdx)) * x(:,sampleIdx)' .* repmat(sum(W1.^2,2),1,obj.visibleSize)) );        
%     end   
%     sum(dd,3)
%     
%     aa = (W1 + ( (1 - 2 * a2) * x' .* repmat(sum(W1.^2,2),1,obj.visibleSize) ) /100 );
%     bb = repmat(mean((2 * a2.^2 .* (1 - a2).^2),2),1,obj.visibleSize) ;        
%     cc = aa .* bb  ;     
%     
%     W_cae1 = bsxfun(@times, W1, mean(a2 .* (1 - a2).^2, 2));
%     W_cae2 = W1.^2 .* ( (...
%         (1 - 2 * a2) .* a2 .* (1 - a2).^2 ...
%     ) * x' / nSamples);
%     W_cae = W_cae1 + W_cae2;
%     c = W_cae;
    %----------------------------------------------------------------------
    
    W1grad = W1grad + obj.contrLevel * c;            

% TODO add bias option    
    b1grad = (1/nSamples) * sum(delta2,2);
    b2grad = (1/nSamples) * sum(delta3,2);
    
    % vectorize gradient            
    grad = W1grad(:);
    
% TODO add bias option    
    if obj.addBias 
        grad = [grad(:); b1grad(:); b2grad(:)];
    end        
    
end


end

