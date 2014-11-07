function [params, costs] = train_BB(obj)
%==================================================================
% Train a binary visible and binary hidden (both bernoulli) RBM
% using contrastive divergence for one phase.
%
% orhanf
%==================================================================

data = obj.x;
nSamples  = size(data,2);
batchsize = min(obj.batchSize,nSamples);
nBatches  = floor(nSamples / batchsize);

costs  = zeros(obj.nEpochs,1);
params = struct('W',[],'b',[],'c',[]);

% initialize weights
if isempty(obj.randSource)
    W = 0.1 * rand(obj.hiddenSize, obj.visibleSize);
else
    W = 0.1 * (obj.sample_bernoulli(zeros(obj.hiddenSize, obj.visibleSize), ...
                prod([obj.hiddenSize, obj.visibleSize])) * 2 - 1) ;
end
b = zeros(obj.hiddenSize,1);    % biases to hidden units
c = zeros(obj.visibleSize,1);   % biases to visible units

gradW = zeros(size(W));
gradb = zeros(size(b));
gradc = zeros(size(c));

for epoch=1:obj.nEpochs
    
    % shuffle dataset
    if isempty(obj.randSource)
        sampleIdx = randperm(nSamples);
    else
        sampleIdx = 1:nSamples;
    end
    
    cost = 0;
    for batch = 1:nBatches
        
        % index calculation for current batch
        batchIdx = sampleIdx((batch - 1) * batchsize + 1 : batch * batchsize);
        
        % UP PHASE --------------------------------------------------------
        visStatesUP = data(:,batchIdx);                                     % this is the reality
        hidProbsUP  = sigmoid(W * visStatesUP + repmat(b, [1, batchsize])); % hidden activations
        hidStatesUP = double(hidProbsUP>obj.sample_bernoulli(hidProbsUP));  % hidden states (bernoulli sampling)
        
        % DOWN PHASE ------------------------------------------------------
        visProbsDOWN  = sigmoid(W' * hidStatesUP + repmat(c, [1, batchsize]));   % visible activations
        visStatesDOWN = double(visProbsDOWN>obj.sample_bernoulli(visProbsDOWN)); % this is the reconstruction (sampling)
        hidProbsDOWN  = sigmoid(W * visStatesDOWN + repmat(b, [1, batchsize]));  % go up again
        hidStatesDOWN = double(hidProbsDOWN>obj.sample_bernoulli(hidProbsDOWN)); % hidden states (bernoulli sampling)
        
        % COMPUTE GRADIENT ------------------------------------------------
        if obj.reduceNoise                                                  % use probabilities rather than binary states
            posAssociation = hidProbsUP  * visStatesUP';
            negAssociation = hidProbsDOWN * visProbsDOWN';
        else                                                                % use binary states
            posAssociation = hidStatesUP * visStatesUP';
            negAssociation = hidStatesDOWN * visStatesDOWN';
        end
        grad = (posAssociation - negAssociation);
        
        % UPDATE WEIGHTS --------------------------------------------------
        gradW = obj.momentum * gradW + obj.learningRate * ( grad / batchsize - obj.weightDecay * W);
        W = W + gradW;
        if obj.addBias
            gradb = obj.momentum * gradb + obj.learningRate * sum(hidProbsUP  - hidProbsDOWN, 2) / batchsize;
            gradc = obj.momentum * gradc + obj.learningRate * sum(visStatesUP - visStatesDOWN,2) / batchsize;
            b = b + gradb;
            c = c + gradc;
        end
        
        % CALCULATE RECONSTRUCTION COST -----------------------------------
        cost = cost + sum(sum((visStatesUP - visProbsDOWN) .^ 2)) / batchsize;
    end
    
    fprintf('Epoch:[%d/%d] - Reconstruction error:[%.6f]\n',epoch,obj.nEpochs,cost);
    costs(epoch) = cost;
end

params.W = W;
params.b = b;
params.c = c;

end
