function [ J grad ] = rbmCost(obj, theta, visibleSize, hiddenSize, X, weightDecay)
%RBMCOST Summary of this function goes here
%   Detailed explanation goes here
%
% orhanf
%%

% extract out W,b,c
W = reshape(theta(1:visibleSize*hiddenSize),[hiddenSize, visibleSize]);
b = theta(visibleSize*hiddenSize+1:visibleSize*hiddenSize+hiddenSize);
c = theta(visibleSize*hiddenSize+1+hiddenSize:end);

m = size(X,2);

% UP PHASE --------------------------------------------------------
visStatesUP = double(X > obj.sample_bernoulli(X));
hidProbsUP  = sigmoid(W * visStatesUP + repmat(b, [1, m])); % hidden activations
hidStatesUP = double(hidProbsUP>obj.sample_bernoulli(hidProbsUP));  % hidden states (bernoulli sampling)

% DOWN PHASE ------------------------------------------------------
visProbsDOWN  = sigmoid(W' * hidStatesUP + repmat(c, [1, m]));   % visible activations
visStatesDOWN = double(visProbsDOWN>obj.sample_bernoulli(visProbsDOWN)); % this is the reconstruction (sampling)
hidProbsDOWN  = sigmoid(W * visStatesDOWN + repmat(b, [1, m]));  % go up again

% COMPUTE GRADIENT ------------------------------------------------        
posAssociation = hidProbsUP  * visStatesUP';        
negAssociation = hidProbsDOWN * visStatesDOWN';

% UPDATE WEIGHTS --------------------------------------------------
gradW = (posAssociation - negAssociation)  / m; % TODO : no weight decay
gradb = sum(hidProbsUP  - hidProbsDOWN, 2) / m;
gradc = sum(visStatesUP - visStatesDOWN,2) / m;

J = sum(sum((visStatesUP - visProbsDOWN) .^ 2)) / m;

grad = [gradW(:); gradb(:); gradc(:)];

end

