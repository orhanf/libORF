function y = nonLinearity(x, ntype )
%NONLINEARITY , elementwise non-linearity function
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
    case 0  % sigmoid
        y = 1 ./ (1+exp(-x));
    case 1  % hyperbolic tangent
        y = tanh(x);
    case 2  % rectified linear unit
        y = max(x,0);
    case 3  % linear activation
        y = x;
end
    
end

