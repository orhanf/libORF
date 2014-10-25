function y = dNonLinearity( x, ntype )
%DNONLINEARITY Derivative of elementwise nonlinearities
%
% Inputs  
%   x     : input
%   ntype : 0-sigmoid, 1-tanh, 2-relu, 3-linear
%
% Outputs
%   y     : output
%
% orhanf
%%

switch ntype
    case 0 % sigmoid
        y =  x .* (1 - x);
    case 1 % tanh
        y = 1 - x.^2;
    case 2 % relu
        y = zeros(size(x));
        y(x>0) = 1;        
    case 3 % linear
        y = ones(size(x));
end


end

