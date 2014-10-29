function [ cost, grad] = penalizedKernelL2_matrix(obj, w,K,nCols,gradFunc,lambda,varargin )
% Adds kernel L2-penalization to a loss function, when the weight vector
%   is actually a matrix with nCols columns (and the kernel is
%   block-diagonal with respect to the columns)
% (you can use this instead of always adding it to the loss function code)
% 
%   This function is adapted from pmtk-toolbox and slightly enhanced for
%   speed-up
% - orhanf

if nargout <= 1
    [cost] = gradFunc(w,varargin{:});
elseif nargout == 2
    [cost,grad] = gradFunc(w,varargin{:});
end

nInstances = size(K,1);
w = reshape(w,[nInstances nCols]);

% calculate the cost
cost = cost + lambda * sum(sum((K*w) .* w));

% calculate gradient if necessary
if nargout > 1
    grad = reshape(grad,[nInstances nCols]);   
    grad = grad + 2 * lambda .* (K*w);
    grad = grad(:);
end

end

