function stack = params2stack(obj, params)

% Converts a flattened parameter vector into a nice "stack" structure 
% for us to work with. This is useful when you're building multilayer
% networks.
%
% stack = params2stack(params)
%
% params - flattened parameter vector
%
% Starter code from UFLDL

% Map the params (a vector into a stack of weights)
depth = numel(obj.nnLayers)-1;
stack = cell(depth,1);
prevLayerSize = obj.inputSize; % the size of the previous layer
curPos = double(1);                  % mark current position in parameter vector

for d = 1:depth
    % Create layer d
    stack{d} = struct;

    % Extract weights
    wlen = double(obj.nnLayers{d+1}.nNeuron * prevLayerSize);
    stack{d}.w = reshape(params(curPos:curPos+wlen-1),prevLayerSize, obj.nnLayers{d+1}.nNeuron)';
    curPos = curPos+wlen;

    % Extract bias
    if obj.addBias
        blen = double(obj.nnLayers{d+1}.nNeuron);
        stack{d}.b = reshape(params(curPos:curPos+blen-1), obj.nnLayers{d+1}.nNeuron, 1);
        curPos = curPos+blen;
    else
        stack{d}.b = zeros(obj.nnLayers{d+1}.nNeuron, 1);
    end
    
    % Set previous layer size
    prevLayerSize = obj.nnLayers{d+1}.outSize;
end

end