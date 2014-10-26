%% Load dataset and labels
addpath(genpath('minfunc_2012/'));
images = loadMNISTImages('data/train-images.idx3-ubyte');
labels = loadMNISTLabels('data/train-labels.idx1-ubyte');
labels(labels==0) = 10; % Remap 0 to 10 


%% Gradient Checking
% Check numberical gradient - note it may take several minutes in full
% model, in this implementation libORF achieved  1.0714e-08 difference by
% calculating difference as norm(numgrad-grad)/norm(numgrad+grad) which is
% reasonably small implying gradient implentation is correct
% Uncomment if you want to check
% options.x      = rand(10,50);
% options.y      = [ones(1,25), ones(1,25)+1];
% options.lambda = 0;
% SM2 = SoftmaxRegressor(options);
% [numgrad, grad] = SM2.check_numerical_gradient();


%% Softmax object initialization
% Set model parameters of sparse autoencoder object with an options struct
options.lambda = 3e-3;
options.x      = images; 
options.y      = labels;     

% Initialize softmaxRegressor object
SM = SoftmaxRegressor(options);

% Training
% Set minimization function for softmax object
foptions.Method  = 'lbfgs';                          
foptions.maxIter = 400;	  
foptions.display = 'on';
SM.minfunc = @(funJ,theta)minFunc(funJ, theta, foptions);

% Start training our SM with minFunc as optimizer - note this step takes
% less than a minute in a desktop for 100 iterations and resulting function
% value is 2.25717e-01
SM.train_model;


% Test
% Predict new samples with trained model, which is calculated by libORF as 
% Accuracy: 94.150% Error: 2.12963e-01
pred = SM.predict_samples(images);
acc  = mean(labels(:) == pred(:));
fprintf('Accuracy: %0.3f%%\n', acc * 100);


%% Now train a model using gradient descent

% Set model parameters of sparse autoencoder object with an options struct
options.lambda   = 3e-3;
options.x        = images; 
options.y        = labels;     
options.nIters   = 500;
options.alpha    = 1;
options.momentum = 0.9;
options.useGPU   = true;

% Initialize softmaxRegressor object
SM = SoftmaxRegressor(options);

SM.train_model;

% Test
% Predict new samples with trained model, which is calculated by libORF as 
% Accuracy: 93.263% Error: 2.49285e-01
pred = SM.predict_samples(images);
acc  = mean(labels(:) == pred(:));
fprintf('Accuracy: %0.3f%%\n', acc * 100);

