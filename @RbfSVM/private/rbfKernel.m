function kernelMat = rbfKernel(obj, x1, x2, sigma )
%RBFKERNEL 
%
%   - original code is from pmtk toolbox, this is an adapted version tuned
%       for time and memory complexities.
%
% orhanf  
%%

Z = 1/sqrt(2*pi*sigma^2);
squaredEucDist = pdist2(x1',x2').^2;
kernelMat = Z*exp(-squaredEucDist/(2*sigma^2));

end

