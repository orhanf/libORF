function y = dNonLinearity( x, layerOpt )
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

switch layerOpt.act
    case 'sigmoid'      % sigmoid
        y =  x .* (1 - x);
    case 'tanh'         % hyperbolic tangent
        y = 1 - x.^2;
    case 'relu'         % rectified linear
        y = zeros(size(x));
        y(x>0) = 1;        
    case 'linear'       % linear
        y = ones(size(x));
    case 'stanh'        % scaled tanh
           y = 1.7159 * 2/3 * (1 - tanh( 2/3 * x).^2);
    case 'maxout'       % maxout 
        y = zeros(size(x));
        y(x~=0) = 1;                 
end


end

