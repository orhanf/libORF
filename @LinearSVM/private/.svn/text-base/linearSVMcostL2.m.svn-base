function [ cost, grad ] = linearSVMcostL2(obj, theta, X, Y, C)
%   Inputs
%     obj    : caller object
%     theta  : parameter vector
%     X      : <n x m> matrix containing the training data.
%              So, data(:,i) is the i-th training example.
%     Y      : <m x 1> class label vector
%     C      : regularization parameter (similar to C parameter of libSVM)
%
%   Outputs
%     cost   : objective cost
%     grad   : gradient vector for parameters
%
%   Note that this implementation is similar to liblinear
%
% orhanf - (c) 2013 Orhan FIRAT <orhan.firat@ceng.metu.edu.tr>
%%

% get helpers
nSamples  = size(X,1);
nFeatures = size(X,2);
nClasses  = size(Y,2);

% re-organise weight into a matrix
theta = reshape(theta, [nFeatures, nClasses]);

% calculate margin
margin = max(0, 1 - Y .* (X*theta));

% Regularization term of the cost function L1 term
J_regL1 = obj.lambdaL1/(2*nSamples) * sum(abs(theta(:)));

% calculate cost
% cost = (0.5 * sum(theta.^2)) + C*mean(margin.^2);
cost = (0.5 * sum(theta.^2)) + C*sum(margin.^2);
cost = sum(cost) + J_regL1;

% calculate gradient and vectorise it 
if nargout>1
%     grad = theta - 2*C/nSamples * (X' * (margin .* Y));
    grad = theta - (2*C * (X' * (margin .* Y)) + (obj.lambdaL1 .* sign(theta)));
    grad = grad(:);
end

end

