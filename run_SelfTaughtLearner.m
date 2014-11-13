%% Load data and seperate it into unlabeled, training and test parts
% Use first 5 class to train softmax and last 5 class to train sparse
% autoencoder respectively. Also seperate first 5 class samples into two
% for training and testing the final self-taught learner model

% Load MNIST database files
mnistData   = loadMNISTImages('data/train-images.idx3-ubyte');
mnistLabels = loadMNISTLabels('data/train-labels.idx1-ubyte');

% Set Unlabeled Set (All Images)

% Simulate a Labeled and Unlabeled set
labeledSet   = find(mnistLabels >= 0 & mnistLabels <= 4);
unlabeledSet = find(mnistLabels >= 5);

numTrain = round(numel(labeledSet)/2);
trainSet = labeledSet(1:numTrain);
testSet  = labeledSet(numTrain+1:end);

unlabeledData = mnistData(:, unlabeledSet);

trainData   = mnistData(:, trainSet);
trainLabels = mnistLabels(trainSet)' + 1; % Shift Labels to the Range 1-5

testData   = mnistData(:, testSet);
testLabels = mnistLabels(testSet)' + 1;   % Shift Labels to the Range 1-5

% Output Some Statistics
fprintf('# examples in unlabeled set: %d\n', size(unlabeledData, 2));
fprintf('# examples in supervised training set: %d\n\n', size(trainData, 2));
fprintf('# examples in supervised testing set: %d\n\n', size(testData, 2));

clear mnistData mnistLabels labeledSet unlabeledSet numTrain trainSet testSet


%% Set model parameters

% Model parameters for sparse autoencoder and softmax regressor
nFeatures       = 28 * 28; % Number of nodes in the input layer of sparse autoencoder
hiddenSize      = 200;     % Number of nodes in the hidden layer of sparse autoencoder
sparsityParam   = 0.1;     % desired average activation of the hidden units for sparse autoencoder.
lambdaSA        = 3e-3;    % weight decay parameter for sparse autoencoder(regularization)
lambdaSM        = 1e-4;    % weight decay parameter for sorftmax regressor(regularization)
beta            = 3;       % weight of sparsity penalty term for sparse autoencoder


% Model parameters for sparse autoencoder
optionsSA.visibleSize   = nFeatures;
optionsSA.hiddenSize    = hiddenSize;
optionsSA.sparsityParam = sparsityParam;
optionsSA.lambda        = lambdaSA;
optionsSA.beta          = beta;

% Model parameters for softmax regressor
optionsSM.lambda = lambdaSM;


%% Initialize Self-Taught Learner object

% Set model parameters of SelfTaughtLearner object 
options.optionsSA     = optionsSA;
options.optionsSM     = optionsSM;
options.trainData     = trainData;
options.trainLabels   = trainLabels;
options.unlabeledData = unlabeledData;

STL = SelfTaughtLearner(options);


%% Set optimizer parameters for SA and SM in STL

% Parameters for optimizers (same optimizer will be used for both SA and SM with the same options)
foptions.Method  = 'lbfgs';
foptions.maxIter = 400;	  
foptions.display = 'on';

STL.SA.minfunc = @(funJ,theta)minFunc(funJ, theta, foptions);
STL.SM.minfunc = @(funJ,theta)minFunc(funJ, theta, foptions);


%% Train model, first trains autoencoder then softmax regressor

% Train both SA and SM
[optthetaSA optthetaSM] = STL.train_model;


% Visualize hidden weights of SA
W1 = reshape(optthetaSA(1:hiddenSize * nFeatures), hiddenSize, nFeatures);
Utility.display_images(W1',[],[],14);


%% Test model

pred = STL.predict_samples(testData);

% Classification Score
fprintf('Test Accuracy: %f%%\n', 100*mean(pred(:) == testLabels(:)));



