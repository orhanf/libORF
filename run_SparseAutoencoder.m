%% Load data 
addpath(genpath('data/'));
images = loadMNISTImages('data/train-images.idx3-ubyte');


%% Check gradient implementation first for correctness by finite diff.

options.visibleSize   = 10;
options.hiddenSize    = 5;
options.sparsityParam = 0.1;
options.lambda        = 0;
options.beta          = 3;
options.x             = rand(10,100); % use first 10k images

% Initialize sparse autoencoder object
SA0 = SparseAutoencoder(options);

SA0.check_numerical_gradient;

clear SA0

%% Initialize sparse autoencoder object
% Set model parameters of sparse autoencoder object with an options struct
options.visibleSize   = 28*28;
options.hiddenSize    = 196;
options.sparsityParam = 0.1;
options.lambda        = 3e-3;
options.beta          = 3;
options.x             = images(:,1:10000); % use first 10k images

% Initialize sparse autoencoder object
SA = SparseAutoencoder(options);

% Display first 100 samples
Utility2.display_images(SA.x(:,1:100));


%% Training 
% Set minimization function for SA object
foptions.Method  = 'lbfgs';                          
foptions.maxIter = 400;	  
foptions.display = 'on';
SA.minfunc = @(funJ,theta)minFunc(funJ, theta, foptions);

% Start training our SA
SA.train_model;

% Display resulting hidden layer features
W1 = reshape(SA.theta(1:SA.hiddenSize*SA.visibleSize), SA.hiddenSize, SA.visibleSize);
Utility2.display_images(W1'); 

clear SA

%% Train using Stochastic Gradient Descent this time, with similar model

% Set model parameters of sparse autoencoder object with an options struct
options.visibleSize   = 28*28;
options.hiddenSize    = 196;
options.sparsityParam = 0.1;
options.lambda        = 3e-3;
options.beta          = 3;
options.x             = images(:,1:10000); % use first 10k images
options.nEpochs       = 50;
options.useAdaDelta   = true;

% Initialize sparse autoencoder object
SA = SparseAutoencoder(options);

% Start training our SA
SA.train_model;

% Display resulting hidden layer features
W1 = reshape(SA.theta(1:SA.hiddenSize*SA.visibleSize), SA.hiddenSize, SA.visibleSize);
Utility2.display_images(W1'); 

clear SA

