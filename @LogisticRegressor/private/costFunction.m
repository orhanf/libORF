function [J, grad] = costFunction(obj, theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta


    % Initialize some useful values
    m = length(y); % number of training examples

    g = inline('1.0 ./ (1.0 + exp(-z))');   % Logistic function
                
    z = X * theta;                          % Logit
%     z = bsxfun(@minus, z, max(z, [], 1));   % This is for numberical overflow           
    
    h = g(z);   % Hypothesis      
   
    % Cost function - this cost is also called cross entropy
    J = (1/m) .* ( (-y' * log(h)) - ((1-y)' * log(1-h)) );

    % Gradient
    grad = (1/m) .* X' * ( h - y );            

end

