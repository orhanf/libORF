%% Load data

addpath(genpath(pwd))

% Load MNIST database files
trainData = loadMNISTImages('data/train-images.idx3-ubyte');
trainLabels = loadMNISTLabels('data/train-labels.idx1-ubyte');

trainLabels(trainLabels == 0) = 10; % Remap 0 to 10 since labels need to start from 1

%% Set model parameters

%--------------------------------------------------------------------------
nFeatures       = 28 * 28; % Number of nodes in the input layer of sparse autoencoder
hiddenSizeL1    = 200;     % Layer 1 Hidden Size
hiddenSizeL2    = 200;     % Layer 2 Hidden Size
sparsityParam   = 0.1;     % desired average activation of the hidden units. for sparse autoencoders
lambda          = 3e-3;    % weight decay parameter for both softmax regressor and sparse autoencoders
beta            = 3;       % weight of sparsity penalty term for sparse autoencoders
%--------------------------------------------------------------------------

% Set model parameters of first layer sparse autoencoder object with an options struct
optionsSA{1}.visibleSize   = nFeatures;
optionsSA{1}.hiddenSize    = hiddenSizeL1;
optionsSA{1}.sparsityParam = sparsityParam;
optionsSA{1}.lambda        = lambda;
optionsSA{1}.beta          = beta;

% Set model parameters of first layer sparse autoencoder objects with
% options structs
%  -hidden layer activations of the first autoencoder will be fed to the
%   second autoencoder's input layer, therefore no need to specify number of
%   input units for the second autoencoder, it will be set to the number of
%   hidden layers of the first autoencoder internally, 
%       optionsSA{2}.hiddenSize    = hiddenSizeL2;
%  -this is also the case for input data of second autoencoder which is set
%   internally.
optionsSA{2}.hiddenSize    = hiddenSizeL2;
optionsSA{2}.sparsityParam = sparsityParam;
optionsSA{2}.lambda        = lambda;
optionsSA{2}.beta          = beta;

% Model parameters for softmax regressor
optionsSM.lambda = 1e-4;

% Model parameters for Stacked Autoencoder object
optionsSAE.optionsSA   = optionsSA;
optionsSAE.optionsSM   = optionsSM;
optionsSAE.trainData   = trainData;
optionsSAE.trainLabels = trainLabels;
optionsSAE.lambda      = lambda;

% Optimizer options
foptions.Method  = 'lbfgs';                          
foptions.maxIter = 400;	  
foptions.display = 'on';


%% Initialize StackedAutoencoder object which has 2 autoencoders and a softmax classifier
SAE = StackedAutoencoder(optionsSAE);


%% Train model, this involves training Autoencoders in order and then training the Softmax at last, seperately
% This step is also called as "Greedy Layer-wise Training"

% Set optimizers for each layer, softmax and fine-tuning
SAE.SA(1).minfunc = @(funJ,theta)minFunc(funJ, theta, foptions);
SAE.SA(2).minfunc = @(funJ,theta)minFunc(funJ, theta, foptions);
SAE.SM.minfunc    = @(funJ,theta)minFunc(funJ, theta, foptions);

% Start training
stackedAETheta = SAE.train_model;


%% Fine-tune trained model

% Set optimizer for Stacked Autoencoder 
SAE.minfunc       = @(funJ,theta)minFunc(funJ, theta, foptions);
stackedAEOptTheta = SAE.fine_tune_model;


%% Perform prediction with test data

% load test data and corresponding labels
testData = loadMNISTImages('data/t10k-images.idx3-ubyte');
testLabels = loadMNISTLabels('data/t10k-labels.idx1-ubyte');

testLabels(testLabels == 0) = 10; % Remap 0 to 10

% Predict with greedy layer-wise training parameters
pred1 = SAE.predict_samples(testData, stackedAETheta);

% Predict with fine-tuned parameters
pred2 = SAE.predict_samples(testData, stackedAEOptTheta);

acc1 = mean(testLabels(:) == pred1(:));
acc2 = mean(testLabels(:) == pred2(:));

fprintf('Before Finetuning Test Accuracy: %0.3f%%\n', acc1 * 100);
fprintf('After Finetuning Test Accuracy: %0.3f%%\n', acc2 * 100);

% Before Finetuning Test Accuracy: 94.770%
% After Finetuning Test Accuracy: 98.010%




