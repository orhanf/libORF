function [ J, grad ] = costFunctionLinRegL2(obj, theta, X, y, lambda)
%COSTFUNCTÝONLÝNREGL2 
%
%   Linear Regression Cost function with L2 regularization
%
% orhanf
%%
  

    m = length(y); % number of training examples
                    
    % Cost function
    J = (0.5/m) .* ...
        (...
            (X * theta - y)' * ( X * theta - y) + ...       % original cost function
            (lambda .* (theta(2:end)' * theta(2:end))) ...  % regularization term 
        );
    
    % Gradient
    grad0 = ((1/m) .* (X(:,1)' * ((X * theta) - y)));   % original gradient for theta0
                                                        % no regularization term for theta0                                                       
                                                 
    gradj = ((1/m) .* (X(:,2:end)' * ((X * theta) - y))) + ...  % original gradient
        (lambda/m).*theta(2:end);                               % regularization term

    grad = [grad0; gradj];
    
    
end

