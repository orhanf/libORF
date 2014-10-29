%% ========================================================================
%   This script contains a sample run for deep learning framework with 
%    stacked autoencoders as internal layers and a linear svm as the 
%    output layer. All of the classes used in this script are implemented
%    in libORF except optimization function "minFunc". 
%
%   Notes to programmers:
%       - Provided demo uses MNIST dataset
%       - MNIST dataset consist of gray level identity sized character
%           images in the range [0,1]
%       - For any extension, please consider scaling/pre-processing or
%           normalizing your data beforehand
%       - Linear SVM in the output layer is a scalable version for big
%           datasets, having lots of samples. For time and memory
%           complexities primal is used and optimized with a quasi-newton
%           method, hence expect some minor weakening in accuracy
%       - Have fun:)
%
% orhanf - (c) 2013 Orhan FIRAT <orhan.firat@ceng.metu.edu.tr>
%==========================================================================
%% Load data

addpath(genpath(pwd))

% Load MNIST database files
trainData = loadMNISTImages('data/train-images.idx3-ubyte');
trainLabels = loadMNISTLabels('data/train-labels.idx1-ubyte');

trainData = trainData(:,1:5000);
trainLabels = trainLabels(1:5000);

trainLabels(trainLabels == 0) = 10; % Remap 0 to 10 since labels need to start from 1

%% Set model parameters

%--------------------------------------------------------------------------
nFeatures       = 28 * 28; % Number of nodes in the input layer of sparse autoencoder
hiddenSizeL1    = 200;     % Layer 1 Hidden Size
hiddenSizeL2    = 200;     % Layer 2 Hidden Size
sparsityParam   = 0.1;     % desired average activation of the hidden units. for sparse autoencoders
lambda          = 3e-3;    % weight decay parameter for both softmax regressor and sparse autoencoders
beta            = 3;       % weight of sparsity penalty term for sparse autoencoders
svmCost         = 500;     % cost parameter of linear SVM
svmCostFineTune = 50;     % cost parameter of linear SVM
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

% Model parameters for linear svm
optionsSVM.C = svmCost;

% Model parameters for Stacked Autoencoder object
optionsSAE.optionsSA   = optionsSA;
optionsSAE.optionsSVM  = optionsSVM;
optionsSAE.trainData   = trainData;
optionsSAE.trainLabels = trainLabels;
optionsSAE.lambda      = lambda;

% Optimizer options
foptions.Method  = 'lbfgs';                          
foptions.maxIter = 400;	  
foptions.display = 'on';


%% Initialize StackedAutoencoder object which has 2 autoencoders and a linear SVM classifier
SAE = StackedAutoencoder(optionsSAE);


%% Train model, this involves training Autoencoders in order and then training the linear SVM at last, seperately
% This step is also called as "Greedy Layer-wise Training"

% Set optimizers for each layer, linear SVM and fine-tuning
SAE.SA(1).minfunc = @(funJ,theta)minFunc(funJ, theta, foptions);
SAE.SA(2).minfunc = @(funJ,theta)minFunc(funJ, theta, foptions);
SAE.SVM.minfunc   = @(funJ,theta)minFunc(funJ, theta, foptions);

% Start training
stackedAETheta = SAE.train_model;


%% Fine-tune trained model

% Set optimizer for Stacked Autoencoder 
SAE.minfunc       = @(funJ,theta)minFunc(funJ, theta, foptions);
SAE.SVM.C         = svmCostFineTune;
stackedAEOptTheta = SAE.fine_tune_model;


%% Perform prediction with test data

% load test data and corresponding labels
testData = loadMNISTImages('data/t10k-images.idx3-ubyte');
testLabels = loadMNISTLabels('data/t10k-labels.idx1-ubyte');

testData = testData(:,1:5000);
testLabels = testLabels(1:5000);

testLabels(testLabels == 0) = 10; % Remap 0 to 10 since labels need to start from 1

% Predict with greedy layer-wise training parameters
pred1 = SAE.predict_samples(testData, stackedAETheta);

% Predict with fine-tuned parameters
pred2 = SAE.predict_samples(testData, stackedAEOptTheta);

acc1 = mean(testLabels(:) == pred1(:));
acc2 = mean(testLabels(:) == pred2(:));

fprintf('Before Finetuning Test Accuracy: %0.3f%%\n', acc1 * 100);
fprintf('After Finetuning Test Accuracy: %0.3f%%\n', acc2 * 100);

% Before Finetuning Test Accuracy: 90.530%
% After Finetuning Test Accuracy: 94.410%





