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

if iscell(ntype), ntype = ntype{1,1}; end

switch ntype
    case {'sigmoid',0}      % sigmoid
        y =  x .* (1 - x);
    case {'tanh',   1}      % hyperbolic tangent
        y = 1 - x.^2;
    case {'relu',   2}      % rectified linear
        y = zeros(size(x));
        y(x>0) = 1;        
    case {'linear', 3}      % linear
        y = ones(size(x));
end


end

