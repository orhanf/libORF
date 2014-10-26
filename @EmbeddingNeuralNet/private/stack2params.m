function [params, netconfig] = stack2params(obj, stack, transposeFlag)

% Converts a "stack" structure into a flattened parameter vector and also
% stores the network configuration. This is useful when working with
% optimization toolboxes such as minFunc.
%
% [params, netconfig] = stack2params(stack)
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
    
    % This can be optimized. But since our stacks are relatively short, it
    % is okay
    if obj.addBias && d ~= 1
        params = [params ; seralizeMat(stack{d}.w,transposeFlag) ; stack{d}.b(:) ];        
    else
        params = [params ; seralizeMat(stack{d}.w,transposeFlag)];
    end    
    
end

if nargout > 1
    % Setup netconfig
    if numel(stack) == 0
        netconfig.inputsize = 0;
        netconfig.layersizes = {};
    else
        netconfig.inputsize = size(stack{1}.w, 2);
        netconfig.layersizes = size(stack{1}.w,1)*obj.nEmbeds;
        for d = 2:numel(stack)
            netconfig.layersizes = [netconfig.layersizes ; size(stack{d}.w,1)];
        end
    end
end

end

function s = seralizeMat(m,transposeFlag)
    if transposeFlag
        m = m';        
    end
    s = m(:);
end