function [ cost, grad ] = rbfSVMcost(obj, theta, x, y , nClasses)
%   Inputs
%     obj      : caller object
%     theta    : parameter vector
%     X        : <nSamples x nSamples> matrix containing the training data.
%     y        : <nSamples x nClasses> class label vector
%     nClasses : number of classes numel(unique(y))
%
%   Outputs
%     cost   : objective cost
%     grad   : gradient vector for parameters
%
%   Note that this implementation is adapted from pmtk-toolbox and slightly
%   enhanced for speed-up
%
%       cost = sum_k ( max(0, 1 + <w_k,x> - <w_y,x> ) )
%
% orhanf 
%%

% get helpers
y = logical(y+1)*(1:nClasses)';
nFeatures = size(x,1);
nSamples  = size(y,1);
classVec  = 1:nClasses;

% re-organise weight into a matrix
theta = reshape(theta, [nFeatures, nClasses]);

cost = 0;
grad = zeros(nFeatures,nClasses);

for i = 1:nSamples
    
    yThis = y(i);
    
    % calculate cost
    mSub1  = bsxfun(@minus, theta(:,classVec(classVec~=yThis)), theta(:,yThis));    
    mSub2  = x(i,:)*mSub1 + 1;     % this is error
    mSub2(mSub2<0) = 0;            % take max(0,err)
    cost   = cost + sum(mSub2.^2); % accumulate error
    
    % calculate gradient
    ngrad = bsxfun(@times,2*mSub2,x(i,:)');
    grad(:,classVec(classVec~=yThis)) = grad(:,classVec(classVec~=yThis)) + ngrad;
    grad(:,yThis) = grad(:,yThis) - sum(ngrad,2);
    
end
grad = grad(:);


end

