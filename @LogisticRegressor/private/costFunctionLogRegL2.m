function [ J, grad ] = costFunctionLogRegL2(obj, theta, X, y, lambda)
%
%   Logistic Regression Cost function with L2 regularization
%
% orhanf
%%

    % Initialize some model parameters
    m = length(y); % number of training examples

    g = inline('1.0 ./ (1.0 + exp(-z))'); % logistic function
                
    z = X * theta;
    h = g(z);        
    
    % Cost function - this cost is also called cross entropy
    J = ((1/m) .* ( (-y' * log(h)) - ((1-y)' * log(1-h)) ) ) + ...
        ((lambda/(2*m)) * (norm(theta(2:end))^2));
                   
    % Gradient
    G = (lambda/m) .* theta;                % Regularizatin term for gradient
    G(1) = 0;                               % No regularization for bias
    grad = ((1/m) .* X' * ( h - y )) + G;   % Calculate gradient  
    

end

