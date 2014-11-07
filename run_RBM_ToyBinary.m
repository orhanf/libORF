%% ========================================================================
%   This script contains a sample run for 
%
%   Notes to programmers:
%       - Provided demo uses USPS collection of handwritten digits dataset
%       - USPS dataset consist of gray level identity sized character
%           images in the range [0,1]
%       - For any extension, please consider scaling/pre-processing or
%           normalizing your data beforehand
%       - Have fun:)
%
% orhanf - (c) 2013 Orhan FIRAT <orhan.firat@ceng.metu.edu.tr>
%==========================================================================


%% ========================================================================
% Let us first train an RBM to model binary toy data
%==========================================================================

% we want experiments to be replicable, thus we need a deterministic
% randomness (!) source, for the results to be same- uncomment two lines
randSource = load('data/randomnessSource.mat');
randSource = randSource.randSource;

% toy data
x = [[1,1,1,0,0,0];...
    [1,0,1,0,0,0];...
    [1,1,1,0,0,0];...
    [0,0,1,1,1,0];...
    [0,0,1,1,0,0];...
    [0,0,1,1,1,0]]';

% initialize rbm object
rbm1 = RBM();

% set proper fields
rbm1.x = x;
rbm1.hiddenSize   = 2;
rbm1.visibleSize  = 6;
rbm1.nEpochs      = 5000;
rbm1.learningRate = 0.1;
rbm1.reduceNoise  = true;  % use probabilities rather than binary states
rbm1.momentum     = 0.4;     % do not use momentum
rbm1.batchSize    = 6;     % update for each sample
rbm1.weightDecay  = 0;     % no regularization
% rbm1.randSource   = randSource; 

% train model generatively
[model, costs] = rbm1.train_model;

% visualise weights
fprintf('-------------------------------------------------------------\n');
fprintf('Visualise weights\n\tbiases\t hidden1\thidden2\n');
disp([0 model.b'; model.c, model.W']);
fprintf('-------------------------------------------------------------\n');

% let us feed data and observe hidden activations
[hiddenStates, hiddenProbs, E1] = rbm1.push_visible_data([[0,0,0,1,1,0];...
                                                         [0,1,1,1,1,0];...
                                                         [1,1,0,0,1,0];...
                                                         [1,1,1,1,1,1]]');
fprintf('Hidden states :\n');
disp(hiddenStates');
fprintf('Hidden probabilities:\n');
disp(hiddenProbs');
fprintf('Energy for given samples:[%.6f]\n',E1);
fprintf('-------------------------------------------------------------\n');

% also let us sample visible units
[visibleStates, visibleProbs, E2] = rbm1.push_hidden_data([[0,0];...
                                                          [1,0];...
                                                          [1,0];...
                                                          [1,0];...
                                                          [0,1];...
                                                          [0,1];...
                                                          [0,1];...
                                                          [1,1]]');
fprintf('Visible states :\n');
disp(visibleStates');
fprintf('Visible probabilities:\n');
disp(visibleProbs');
fprintf('Energy for given samples:[%.6f]\n',E2);
fprintf('-------------------------------------------------------------\n');

% now sample data from the model (day-dream or alternating gibbs-sampling)
nSamples = 5;
[sampledStates, sampledProbs, E3] = rbm1.sample_data(nSamples);

fprintf('Sampled states :\n');
disp(sampledStates');
fprintf('Sampled probabilities:\n');
disp(sampledProbs');
fprintf('Energy for given sampled configurations:[%.6f]\n',E3);
fprintf('-------------------------------------------------------------\n');

clear rbm1


