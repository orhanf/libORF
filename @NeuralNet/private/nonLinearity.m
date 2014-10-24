function [y, switches] = nonLinearity(x, layerOpt)
%NONLINEARITY , elementwise non-linearity function
%
% Inputs  
%   x     : input
%   ntype : 0-sigmoid, 1-tanh, 2-relu, 3-linear, 4-scaledTanh, 5-maxout
%
% Outputs
%   y     : output
%
% orhanf
%%

y = [];
switches = [];

switch layerOpt.act
    case 'sigmoid'      % sigmoid
        y = 1 ./ (1+exp(-x));
    case 'tanh'         % hyperbolic tangent
        y = tanh(x);
    case 'relu'         % rectified linear unit
        y = max(x,0);
    case 'linear'       % linear activation
        y = x;
    case 'stanh'        % scaled tanh
        y = 1.7159 * tanh(2/3 .* x);
    case 'maxout'       % maxout
        [y, switches] = ConvUtils.maxout_Fprop(x,layerOpt);
end
    
end

