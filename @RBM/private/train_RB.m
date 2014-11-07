function [params, costs] = train_RB(obj)
%==================================================================
% Train a real valued visible betwee [0,1] and binary hidden RBM
% using contrastive divergence for one phase.
%
% orhanf
%==================================================================

nSamples  = size(obj.x,2);
batchsize = min(obj.batchSize,nSamples);
nBatches  = floor(nSamples / batchsize);

costs  = zeros(obj.nEpochs,1);
params = struct('W',[],'b',[],'c',[]);

% initialize weights
if isempty(obj.randSource)
    W = 0.1 * randn(obj.hiddenSize, obj.visibleSize);
else
    W = 0.1 * (obj.sample_bernoulli(zeros(obj.hiddenSize, obj.visibleSize), ...
        prod([obj.hiddenSize, obj.visibleSize])) * 2 - 1) ;
end
b = zeros(obj.hiddenSize,1);    % biases to hidden units
c = zeros(obj.visibleSize,1);   % biases to visible units


if ~isempty(obj.minfunc)  % use external minimization function
    
    sampleIdx = randperm(nSamples);
    
    for batch = 1:nBatches
        
        batchIdx = sampleIdx((batch - 1) * batchsize + 1 : batch * batchsize);
        
        theta = [W(:); b(:); c(:)];

        [opttheta, cost] = obj.minfunc( @(p) obj.rbmCost(p, obj.visibleSize, obj.hiddenSize, ...
                                                                 obj.x(:,batchIdx), obj.weightDecay), ...
                                                        theta);
        % extract out W,b,c
        W = reshape(opttheta(1:obj.visibleSize*obj.hiddenSize),[obj.hiddenSize, obj.visibleSize]);
        b = opttheta(obj.visibleSize*obj.hiddenSize+1:obj.visibleSize*obj.hiddenSize+obj.hiddenSize);
        c = opttheta(obj.visibleSize*obj.hiddenSize+1+obj.hiddenSize:end);
        
    end
    
else	% use mini-batch gradient descent with shuffling
    
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
            % actually following line is the only difference with BB version, we
            % actually binarize the data
            visStatesUP = double(obj.x(:,batchIdx) > obj.sample_bernoulli(obj.x(:,batchIdx)));
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
            
            % do not sample for hidden units from reconstruction - actually
            % sampling the hidden state that results from the "reconstruction"
            % visible state is useless: it does not change the expected value
            % of the gradient estimate that CD-1 produces; it only increases
            % its variance. More variance means that we have to use a smaller
            % learning rate, and that means that it'll learn more slowly; in
            % other words, we don't want more variance, especially if it
            % doesn't give us anything pleasant to compensate for that
            % slower learning
            if obj.useCondProb
                negAssociation = hidProbsDOWN * visStatesDOWN';
            end
            
            grad = (posAssociation - negAssociation);
            
            % UPDATE WEIGHTS --------------------------------------------------
            gradW = obj.momentum * gradW + obj.learningRate * ( grad / batchsize - obj.weightDecay * W); % TODO : inspect weight decay
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
    
end

params.W = W;
params.b = b;
params.c = c;

end
