function [params] = stack2params(obj, stack, transposeFlag)

% Converts a "stack" structure into a flattened parameter vector and also
% stores the network configuration. This is useful when working with
% optimization toolboxes such as minFunc.
%
% [params] = stack2params(stack)
%
% stack - the stack structure, where stack{1}.w = weights of first layer
%                                    stack{1}.b = weights of first layer
%                                    stack{2}.w = weights of second layer
%                                    stack{2}.b = weights of second layer
%                                    ... etc.
% - starter code from UFLDL

if nargin<3
    transposeFlag = false;
end

% Setup the compressed param vector
params = [];
for d = 1:numel(stack)    
    if obj.addBias
        params = [params ; seralizeMat(stack{d}.w,transposeFlag) ; stack{d}.b(:) ];                        
    else
        params = [params ; seralizeMat(stack{d}.w,transposeFlag)];
    end            
end

end

function s = seralizeMat(m,transposeFlag)
    if transposeFlag
        m = m';        
    end
    s = m(:);
end