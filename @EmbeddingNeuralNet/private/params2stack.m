function stack = params2stack(obj, params, netconfig)

% Converts a flattened parameter vector into a nice "stack" structure 
% for us to work with. This is useful when you're building multilayer
% networks.
%
% stack = params2stack(params, netconfig)
%
% params - flattened parameter vector
% netconfig - auxiliary variable containing 
%             the configuration of the network
%
% Starter code from UFLDL

% Map the params (a vector into a stack of weights)
depth = numel(netconfig.layersizes);
stack = cell(depth,1);

% first extract out embedding weights, note that embedding weights does not
% have a bias component
stack{1}.w = reshape(params(1:obj.embedSize*netconfig.inputsize),...
    [obj.embedSize,netconfig.inputsize]);
stack{1}.b = 0; % nouse
prevLayerSize = obj.embedSize*obj.nEmbeds; % the size of the previous layer
curPos = 1+numel(stack{1}.w) ;       % mark current position in parameter vector

for d = 2:depth
    % Create layer d
    stack{d} = struct;

    % Extract weights    
    wlen = double(netconfig.layersizes(d) * prevLayerSize);
    stack{d}.w = reshape(params(curPos:curPos+wlen-1),netconfig.layersizes(d),prevLayerSize);    
    curPos = curPos+wlen;

    % Extract bias
    if obj.addBias
        blen = double(netconfig.layersizes(d));
        stack{d}.b = reshape(params(curPos:curPos+blen-1), netconfig.layersizes(d), 1);
        curPos = curPos+blen;
    else
        stack{d}.b = zeros(netconfig.layersizes(d), 1);
    end
    
    % Set previous layer size
    prevLayerSize = netconfig.layersizes(d);
end

end