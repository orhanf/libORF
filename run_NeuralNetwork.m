%% ========================================================================
%   This script contains a sample run for standard neural network with 
%    sigmoid/tanh/relu neurons at the internal layers and a softmax as the 
%    output layer. 
%   
%   Examples in the script:
%       - Training regimes using nEpochs, nIters and external optimizer (LBFGS)
%       - Multiple layers
%       - Effect of Dropout regularization
%       - Pre-training using autoencoders
%       - Effect of AdaDelta
%       - GPU examples
%       - Different activation functions
%       - L1 and L2 regularizations
%       - Maxout 
%
%   Notes to programmers:
%       - Provided 1st demo uses USPS handwritten digits dataset
%       - USPS dataset consist of gray level identity sized character
%           images in the range [0,1]
%       - For any extension, please consider scaling/pre-processing or
%           normalizing your data beforehand
%       - The rest of the examples use MNIST dataset which is again in the
%           range [0,1]
%       -If normalization is not applicable for your case, consider
%           changing activation functions, cost function and initialization
%       - Have fun:)
%
% orhanf - (c) 2013 Orhan FIRAT <orhan.firat@ceng.metu.edu.tr>
%==========================================================================

%% Load data

addpath(genpath(pwd))

data = Utility.load_fields('./data/USPS.mat',{'data'});


%% Check gradient implementation first for correctness by finite diff.

% necessary parameters
options.trainData   = rand(10,2);    % Training data
options.trainLabels = data.training.targets(:,1:2);   % Training labels

% architectural parameters
options.nnLayers    = { struct('type','input', 'nNeuron',10),...
                        struct('type','fc',    'nNeuron',10, 'act','maxout', 'poolSize',2),...
                        struct('type','output','nNeuron',10, 'act','softmax')};

% optional parameters - also set by default
options.lambdaL2    = 0;     % Weight decay parameter (for L2 regularization)

% initialize object
NN0 = NeuralNet(options);

NN0.check_numerical_gradient;

clear NN0

%% set parameters of neural net object and initialize it 

% necessary parameters
options.trainData   = data.training.inputs;    % Training data
options.trainLabels = data.training.targets;   % Training labels
options.cvData      = data.validation.inputs;  % Cross validation data
options.cvLabels    = data.validation.targets; % Cross validation labels

testData   = data.test.inputs;
testLabels = (1:10) * data.test.targets;
clear data

% architectural parameters
options.nnLayers    = { struct('type','input', 'nNeuron',256),...
                        struct('type','fc',    'nNeuron',200, 'act','sigmoid'),...
                        struct('type','output','nNeuron',10,  'act','softmax')};   

% optional parameters - also set by default
options.nIters    = 1000;  % Number of iterations in each epoch
options.batchSize = 100;   % Mini-batch size.
options.alpha     = 0.35;  % Learning rate
options.momentum  = 0.9;   % Momentum        
options.lambdaL2    = 0;     % Weight decay parameter (for L2 regularization)
options.stdInitW  = 0.01;  % Standard deviation of the normal distribution which is sampled to get the initial weights
options.silent    = false; % Display cost in each iteration etc.     
options.addBias   = false; % Do not add bias to input and hidden layers

% initialize object
NN = NeuralNet(options);


% train neural net using mini-batch gradient descent
% first let us run using number of iterations regime
NN.train_model;


% test trained neural net on test data
pred = NN.predict_samples(testData);

% Classification Score - you should expect something around 91.888889%
fprintf('Test Accuracy: %f%%\n', 100*mean(pred(:) == testLabels(:)));


%% re-run training using epochs regime (same with nIters == 1000)

NN.trainingRegime = 2;
NN.nEpochs = 100;  
NN.reset;
NN.train_model;


% now test trained models with second nn
pred = NN.predict_samples(testData);

% Classification Score - you should expect someting aroud 92.288889%
fprintf('Test Accuracy: %f%%\n', 100*mean(pred(:) == testLabels(:)));


%% let us see what happens when we add bias

NN.addBias = true;
NN.trainingRegime = 2;
NN.nEpochs = 100;  
NN.reset;
NN.train_model;


% now test trained models with third nn
pred = NN.predict_samples(testData);

% Classification Score - you should expect someting aroud 92.711111%
fprintf('Test Accuracy: %f%%\n', 100*mean(pred(:) == testLabels(:)));

clear NN


%% Clear everything and try MNIST dataset this time, with dropout and 
%  different activation units in the intermediate layers. Note that, this
%  example uses entire MNIST dataset hence takes some time. You may
%  consider shrinking the dataset size for trainData,trainLabels.

clear('options','testData','testLabels','pred'); 

% load MNIST 
images = loadMNISTImages('data/train-images.idx3-ubyte');
labels = loadMNISTLabels('data/train-labels.idx1-ubyte');
labels(labels==0) = 10; % Remap 0 to 10 
labels = Utility.convert_to_one_of_k_encoding(labels);

testData   = images(:,51001:end);
testLabels = (1:10) * labels(:,51001:end);

% necessary parameters
options.trainData   = images(:,1:50000);      % Training data
options.trainLabels = labels(:,1:50000);      % Training labels
options.cvData      = images(:,50001:51000);  % Cross validation data
options.cvLabels    = labels(:,50001:51000);  % Cross validation labels

clear('images','labels');

% architectural parameters
options.nnLayers    = { struct('type','input', 'nNeuron',784),...
                        struct('type','fc',    'nNeuron',200, 'act','sigmoid'),...
                        struct('type','fc',    'nNeuron',200, 'act','sigmoid'),...
                        struct('type','output','nNeuron',10,  'act','softmax')};

% optional parameters - also set by default
options.alpha     = 0.35;  % Learning rate
options.lambdaL2  = 0;     % Weight decay parameter (for L2 regularization)
options.nEpochs   = 5;     % 5 full sweeps over dataset

% initialize object
NN = NeuralNet(options);
NN.train_model;

% now test trained models with third nn
pred = NN.predict_samples(testData);

% Classification Score - you should expect someting aroud  97.055556%
fprintf('Test Accuracy: %f%%\n', 100*mean(pred(:) == testLabels(:)));

clear NN


%% Let us train a similar model but with less data 

% necessary parameters
options.trainData   = images(:,1:5000);       % Training data , 5k
options.trainLabels = labels(:,1:5000);       % Training labels
options.cvData      = images(:,50001:51000);  % Cross validation data , 1k
options.cvLabels    = labels(:,50001:51000);  % Cross validation labels

% optional parameters - also set by default
options.alpha        = 0.35;  % Learning rate
options.lambdaL2     = 0;     % Weight decay parameter (for L2 regularization)
options.nEpochs      = 50;    % 5 full sweeps over dataset

% initialize object
NN = NeuralNet(options);
NN.train_model;

% now test trained models with third nn
pred = NN.predict_samples(testData);

% Classification Score - you should expect someting aroud  93.100000%
fprintf('Test Accuracy: %f%%\n', 100*mean(pred(:) == testLabels(:)));

clear NN


%% This time let us train the previous model using dropout

% necessary parameters
options.trainData   = images(:,1:5000);       % Training data , 5k
options.trainLabels = labels(:,1:5000);       % Training labels
options.cvData      = images(:,50001:51000);  % Cross validation data , 1k
options.cvLabels    = labels(:,50001:51000);  % Cross validation labels

% optional parameters - also set by default
options.alpha        = 1;  % Learning rate - dropout necessitates larger stepsizes
options.lambdaL2     = 0;     % Weight decay parameter (for L2 regularization)
options.nEpochs      = 50;    % 50 full sweeps over dataset
options.dropOutRatio = 0.5;   % Set half of the neurons to zero in hidden layers

% initialize object
NN = NeuralNet(options);
NN.train_model;

% now test trained models with third nn
pred = NN.predict_samples(testData);

% Classification Score - you should expect someting aroud  94.800000%
fprintf('Test Accuracy: %f%%\n', 100*mean(pred(:) == testLabels(:)));

clear NN

%% This time with a big dataset

% necessary parameters
options.trainData   = images(:,1:50000);      % Training data
options.trainLabels = labels(:,1:50000);      % Training labels
options.cvData      = images(:,50001:51000);  % Cross validation data
options.cvLabels    = labels(:,50001:51000);  % Cross validation labels

% architectural parameters
options.nnLayers    = { struct('type','input', 'nNeuron',784),...
                        struct('type','fc',    'nNeuron',200, 'act','sigmoid'),...
                        struct('type','fc',    'nNeuron',200, 'act','sigmoid'),...
                        struct('type','output','nNeuron',10,  'act','softmax')};

% optional parameters - also set by default
options.alpha     = 1;      % Learning rate
options.lambdaL2  = 0;      % Weight decay parameter (for L2 regularization)
options.nEpochs   = 10;     % 5 full sweeps over dataset
options.dropOutRatio = 0.5; % Set half of the neurons to zero in hidden layers

% initialize object
NN = NeuralNet(options);
NN.train_model;

% now test trained models with third nn
pred = NN.predict_samples(testData);

% Classification Score - you should expect someting aroud  97.211111%
fprintf('Test Accuracy: %f%%\n', 100*mean(pred(:) == testLabels(:)));

clear NN


%% We have come so far, now let us try pre-training our neural network, 
%   using unsupervised feature learning with autoencoders. Our neural
%   network that has 2 hidden layers, will be trained by stacking
%   autoencoders. The first (bottom) autoencoder will be a denoising 
%   autoencoder, trained using SGD, the second (upper) autoencoder will be
%   sparse autoencoder trained using BGD. You may specify further options,
%   and also use a contractive autoencoder as well.

% load MNIST 
images = loadMNISTImages('data/train-images.idx3-ubyte');
labels = loadMNISTLabels('data/train-labels.idx1-ubyte');
labels(labels==0) = 10; % Remap 0 to 10 
labels = Utility.convert_to_one_of_k_encoding(labels);

testData   = images(:,51001:end);
testLabels = (1:10) * labels(:,51001:end);

% necessary parameters
options.trainData   = images(:,1:5000);      % Training data - 5k
options.trainLabels = labels(:,1:5000);      % Training labels
options.cvData      = images(:,50001:51000);  % Cross validation data - 1k
options.cvLabels    = labels(:,50001:51000);  % Cross validation labels

% architectural parameters
options.nnLayers    = { struct('type','input', 'nNeuron',784),...
                        struct('type','fc',    'nNeuron',200, 'act','sigmoid'),...
                        struct('type','fc',    'nNeuron',150, 'act','sigmoid'),...
                        struct('type','output','nNeuron',10,  'act','softmax')};

% optional parameters - also set by default
options.alpha        = 0.35; % Learning rate
options.lambdaL2       = 0;    % Weight decay parameter (for L2 regularization)
options.nEpochs      = 10;   % 5 full sweeps over dataset
options.dropOutRatio = 0.5;  % Set half of the neurons to zero in hidden layers

% initialize object
NN = NeuralNet(options);

% now pre-train - note that if nothing is given, libORF trains
% DenoisingAutoencoders for all layers with SGD by default. The AEoptions
% below seem cumbersome but passing an empty argument or no argument will 
% also suffice, the AEoptions here is just for illustration. The classes
% given in the 'aeTypes' argument must implement an .encodeX(data) function
% and a .get_parameters_as_stack() function.
AEoptions = struct( 'aeType', {@DenoisingAutoencoder,@SparseAutoencoder},...   %specify type
                    'aeOpt' , {struct('alpha',1,...                            % options of the first autoencoder
                                       'momentum',.7,...
                                       'useAdaDelta',true),...
                                struct('sparsityParam',.1,...                   % options of the second autoencoder
                                       'lambdaL2',3e-3,...
                                       'minfunc',@(a,b)minFunc(a,b,struct('maxIter',400)))});
                
NN.pre_train_model(options.trainData, AEoptions);

NN.train_model;

% now test trained models with third nn
pred = NN.predict_samples(testData);

% Classification Score - you should expect someting aroud 93.277778%
fprintf('Test Accuracy: %f%%\n', 100*mean(pred(:) == testLabels(:)));

clear NN


%% Same model without pre-training

% initialize object
NN = NeuralNet(options);

NN.train_model;

% now test trained models with third nn
pred = NN.predict_samples(testData);

% Classification Score - you should expect someting aroud 87.988889%
fprintf('Test Accuracy: %f%%\n', 100*mean(pred(:) == testLabels(:)));

clear NN


%% Same model without pre-training but using adadelta

% optional parameters - also set by default
options.useAdaDelta  = true; 
options.adaDeltaRho  = 0.95; 
options.initFanInOut = true; % this is VERY important. If you do not initialize 
                             % your weights respectively (i.e. too small),
                             % then you will observe an increase in the
                             % training error and classification results
                             % will be at chance level. .initFanInOut
                             % ensures the weights are initialized
                             % adaptively to the architecture in a
                             % plausible interval. Note that, by setting
                             % useAdaDelta=true, NeuralNet automatically
                             % sets initFanInOut=true, this is just to
                             % emphasize the importance :) 

% initialize object
NN = NeuralNet(options);

NN.train_model;

% now test trained models with third nn
pred = NN.predict_samples(testData);

% Classification Score - you should expect someting aroud 90.388889%
fprintf('Test Accuracy: %f%%\n', 100*mean(pred(:) == testLabels(:)));

clear NN


%% Now, try a second order method for optimization using Batch Gradient
%   Descent, namely L-BFGS of minFunc

% necessary parameters
options.trainData   = images(:,1:5000);      % Training data - 5k
options.trainLabels = labels(:,1:5000);      % Training labels
options.cvData      = images(:,50001:51000);  % Cross validation data - 1k
options.cvLabels    = labels(:,50001:51000);  % Cross validation labels

% architectural parameters
options.nnLayers    = { struct('type','input', 'nNeuron',784),...
                        struct('type','fc',    'nNeuron',200, 'act','sigmoid'  ),...
                        struct('type','output','nNeuron',10,  'act','softmax')};
                    
% optional parameters - also set by default
options.lambdaL2       = 0;    % Weight decay parameter (for L2 regularization)

% Set minimization function for SA object
foptions.Method  = 'cg';                          
foptions.maxIter = 400;	  
foptions.display = 'on';
options.minfunc = @(funJ,theta)minFunc(funJ, theta, foptions);

% initialize object
NN = NeuralNet(options);

NN.train_model;

% now test trained models with third nn
pred = NN.predict_samples(testData);

% Classification Score - you should expect someting aroud 94.233333%
fprintf('Test Accuracy: %f%%\n', 100*mean(pred(:) == testLabels(:)));

clear NN


%% Test a GPU version and compare speed

% load MNIST 
images = loadMNISTImages('data/train-images.idx3-ubyte');
labels = loadMNISTLabels('data/train-labels.idx1-ubyte');
labels(labels==0) = 10; % Remap 0 to 10 
labels = Utility.convert_to_one_of_k_encoding(labels);

testData   = images(:,51001:end);
testLabels = (1:10) * labels(:,51001:end);

% necessary parameters
options.trainData   = images(:,1:5000);      % Training data - 5k
options.trainLabels = labels(:,1:5000);      % Training labels
options.cvData      = images(:,50001:51000);  % Cross validation data - 1k
options.cvLabels    = labels(:,50001:51000);  % Cross validation labels

% architectural parameters
options.nnLayers    = { struct('type','input', 'nNeuron',784),...
                        struct('type','fc',    'nNeuron',200, 'act','sigmoid'  ),...
                        struct('type','fc',    'nNeuron',150, 'act','sigmoid'),...
                        struct('type','output','nNeuron',10,  'act','softmax')};

% optional parameters - also set by default
options.alpha        = 0.35; % Learning rate
options.lambdaL2       = 0;    % Weight decay parameter (for L2 regularization)
options.nEpochs      = 10;   % 5 full sweeps over dataset
options.dropOutRatio = 0;  % Set half of the neurons to zero in hidden layers
options.useGPU       = true;
options.useAdaDelta  = true;

% clear gpu
g = gpuDevice;
reset(g);

tic 
% initialize object and train using gpu
NN  = NeuralNet(options);
NN.train_model;
toc

tic
options.useGPU = false;
NN2 = NeuralNet(options);
NN2.train_model;
toc

% You should observe 2x speed-up compared to CPU version in MatlabR2013a

% now test trained models with third nn
pred = NN.predict_samples(testData);
pred2 = NN2.predict_samples(testData);

% Classification Score - you should expect someting aroud :
fprintf('Test Accuracy: %f%%\n', 100*mean(pred(:)  == testLabels(:))); % 92.655556%
fprintf('Test Accuracy: %f%%\n', 100*mean(pred2(:) == testLabels(:))); % 92.844444%

clear NN NN2

%% Scaled hyperbolic tangent - sigmoid - relu hidden units

% load MNIST 
images = loadMNISTImages('data/train-images.idx3-ubyte');
labels = loadMNISTLabels('data/train-labels.idx1-ubyte');
labels(labels==0) = 10; % Remap 0 to 10 
labels = Utility.convert_to_one_of_k_encoding(labels);

testData   = images(:,51001:end);
testLabels = (1:10) * labels(:,51001:end);

% necessary parameters
options.trainData   = images(:,1:5000);      % Training data - 5k
options.trainLabels = labels(:,1:5000);      % Training labels
options.cvData      = images(:,50001:51000);  % Cross validation data - 1k
options.cvLabels    = labels(:,50001:51000);  % Cross validation labels

% architectural parameters
options.nnLayers    = { struct('type','input', 'nNeuron',784),...
                        struct('type','fc',    'nNeuron',200, 'act','stanh'  ),...
                        struct('type','fc',    'nNeuron',150, 'act','sigmoid'),...
                        struct('type','fc',    'nNeuron',50,  'act','relu'),...
                        struct('type','output','nNeuron',10,  'act','softmax')};

% optional parameters - also set by default
options.alpha        = 0.35; % Learning rate
options.lambdaL2       = 0;    % Weight decay parameter (for L2 regularization)
options.nEpochs      = 10;   % 5 full sweeps over dataset
options.dropOutRatio = 0;    % Set half of the neurons to zero in hidden layers
options.useAdaDelta  = true;

% initialize object and train using gpu
NN  = NeuralNet(options);
NN.train_model;

% You should observe 2x speed-up compared to CPU version in MatlabR2013a

% now test trained models with third nn
pred = NN.predict_samples(testData);

% Classification Score - you should expect someting aroud : 93.711111%
fprintf('Test Accuracy: %f%%\n', 100*mean(pred(:)  == testLabels(:))); % 

% 'stanh','sigmoid','relu' => 93.955556%

clear NN


%% Rectified linear and tanh workout

% load MNIST 
addpath(genpath('data'));
images = loadMNISTImages('data/train-images.idx3-ubyte');
labels = loadMNISTLabels('data/train-labels.idx1-ubyte');
labels(labels==0) = 10; % Remap 0 to 10 
labels = Utility.convert_to_one_of_k_encoding(labels);

testData   = images(:,51001:end);
testLabels = (1:10) * labels(:,51001:end);
cvData     = images(:,50001:51000);  % Cross validation data - 1k
cvLabels   = labels(:,50001:51000);  % Cross validation labels

% necessary parameters
options.trainData   = images(:,1:5000);      % Training data - 5k
options.trainLabels = labels(:,1:5000);      % Training labels
options.cvData      = cvData;
options.cvLabels    = cvLabels;

% architectural parameters
options.nnLayers    = { struct('type','input', 'nNeuron',784),...
                        struct('type','fc',    'nNeuron',400, 'act','relu'  ),...
                        struct('type','output','nNeuron',10,  'act','softmax')};

% optional parameters - also set by default
options.alpha        = 0.35; % Learning rate
options.lambdaL2       = 0;    % Weight decay parameter (for L2 regularization)
options.nEpochs      = 10;   % 5 full sweeps over dataset
options.dropOutRatio = 0.5;    % Set half of the neurons to zero in hidden layers
options.useAdaDelta  = true;
% options.useGPU       = true;

% initialize object and train using gpu
NN  = NeuralNet(options);
NN.train_model;

% now test trained models with third nn
predTE = NN.predict_samples(testData);
predCV = NN.predict_samples(cvData);
fprintf('CV   Accuracy: %f%%\n', 100*mean(predCV(:)  == [(1:10) * cvLabels ]')); % 
fprintf('Test Accuracy: %f%%\n', 100*mean(predTE(:)  == testLabels(:))); % 

% 95.055556% with 0.5-dropout, 4000 neurons, 10 Epochs, 0.35 alpha with adadelta, 0 weight-decay
% 95.244444% with 0.5-dropout, 2000-1000 neurons, 10 Epochs, 0.35 alpha with adadelta, 0 weight-decay
clear NN


%% Try a second order method for optimization using Stochastic Gradient
%   Descent, namely L-BFGS of minFunc

% necessary parameters
addpath(genpath('data'))
addpath(genpath('minFunc_2012'))
images = loadMNISTImages('data/train-images.idx3-ubyte');
labels = loadMNISTLabels('data/train-labels.idx1-ubyte');
labels(labels==0) = 10; % Remap 0 to 10 
labels = Utility.convert_to_one_of_k_encoding(labels);

testData   = images(:,51001:end);
testLabels = (1:10) * labels(:,51001:end);

options.trainData   = images(:,1:5000);      % Training data - 5k
options.trainLabels = labels(:,1:5000);      % Training labels
options.cvData      = images(:,50001:51000);  % Cross validation data - 1k
options.cvLabels    = labels(:,50001:51000);  % Cross validation labels

% architectural parameters
options.nnLayers    = { struct('type','input', 'nNeuron',784),...                        
                        struct('type','fc',    'nNeuron',200, 'act','sigmoid'),...
                        struct('type','output','nNeuron',10,  'act','softmax')};

% optional parameters - also set by default
options.lambdaL2     = 0;    % Weight decay parameter (for L2 regularization)
options.nEpochs      = 10;
options.batchSize    = 100;
options.trainingRegime = 4;

% Set minimization function for SA object
foptions.Method  = 'lbfgs';                          
foptions.maxIter = 400;	  
foptions.display = 'off';
options.minfunc = @(funJ,theta)minFunc(funJ, theta, foptions);

% initialize object
NN = NeuralNet(options);

NN.train_model;

% now test trained models with third nn
pred = NN.predict_samples(testData);

% Classification Score - you should expect someting aroud 94.233333%
fprintf('Test Accuracy: %f%%\n', 100*mean(pred(:) == testLabels(:)));

clear NN


%% L1 and L2 regularizations at work

% load MNIST 
addpath(genpath('data'));
images = loadMNISTImages('data/train-images.idx3-ubyte');
labels = loadMNISTLabels('data/train-labels.idx1-ubyte');
labels(labels==0) = 10; % Remap 0 to 10 
labels = Utility.convert_to_one_of_k_encoding(labels);

testData   = images(:,51001:end);
testLabels = (1:10) * labels(:,51001:end);
cvData     = images(:,50001:51000);  % Cross validation data - 1k
cvLabels   = labels(:,50001:51000);  % Cross validation labels

% necessary parameters
options.trainData   = images(:,1:5000);      % Training data - 5k
options.trainLabels = labels(:,1:5000);      % Training labels
options.cvData      = cvData;
options.cvLabels    = cvLabels;

% architectural parameters
options.nnLayers    = { struct('type','input', 'nNeuron',784),...
                        struct('type','fc',    'nNeuron',400, 'act','relu'  ),...
                        struct('type','output','nNeuron',10,  'act','softmax')};

% optional parameters - also set by default
options.alpha        = 0.35; % Learning rate
options.lambdaL1     = 0; 
options.lambdaL2     = 1e-3;    
options.nEpochs      = 10;   
options.useAdaDelta  = true;

% initialize object and train 
NN  = NeuralNet(options);
NN.train_model;
weightsL2 = NN.nnOptTheta;

% now test trained models with third nn
predTE = NN.predict_samples(testData);
predCV = NN.predict_samples(cvData);
fprintf('CV   Accuracy: %f%%\n', 100*mean(predCV(:)  == [(1:10) * cvLabels ]')); 
fprintf('Test Accuracy: %f%%\n', 100*mean(predTE(:)  == testLabels(:))); 

clear NN

options.lambdaL1 = 1e-3; 
options.lambdaL2 = 0;
NN  = NeuralNet(options);
NN.train_model;
weightsL1 = NN.nnOptTheta;

predTE = NN.predict_samples(testData);
predCV = NN.predict_samples(cvData);
fprintf('CV   Accuracy: %f%%\n', 100*mean(predCV(:)  == [(1:10) * cvLabels ]')); 
fprintf('Test Accuracy: %f%%\n', 100*mean(predTE(:)  == testLabels(:))); 

clear NN

options.lambdaL1 = 1e-3; 
options.lambdaL2 = 1e-3;
NN  = NeuralNet(options);
NN.train_model;
weightsL1L2 = NN.nnOptTheta;

predTE = NN.predict_samples(testData);
predCV = NN.predict_samples(cvData);
fprintf('CV   Accuracy: %f%%\n', 100*mean(predCV(:)  == [(1:10) * cvLabels ]')); 
fprintf('Test Accuracy: %f%%\n', 100*mean(predTE(:)  == testLabels(:))); 

clear NN

figure,hist(weightsL2,min(weightsL2)-1:.001:max(weightsL2)+1),title('L2 regularized weights')
figure,hist(weightsL1,min(weightsL1)-1:.001:max(weightsL1)+1),title('L1 regularized weights')
figure,hist(weightsL1L2,min(weightsL1L2)-1:.001:max(weightsL1L2)+1),title('L2-L1 regularized weights')

%% Try maxout hidden layer

% load MNIST 
addpath(genpath('data'));
images = loadMNISTImages('data/train-images.idx3-ubyte');
labels = loadMNISTLabels('data/train-labels.idx1-ubyte');
labels(labels==0) = 10; % Remap 0 to 10 
labels = Utility.convert_to_one_of_k_encoding(labels);

testData   = images(:,51001:end);
testLabels = (1:10) * labels(:,51001:end);


% necessary parameters
options.trainData   = images(:,1:5000);       % Training data - 5k
options.trainLabels = labels(:,1:5000);       % Training labels
options.cvData      = images(:,50001:51000);  % Cross validation data - 1k
options.cvLabels    = labels(:,50001:51000);  % Cross validation labels

% architectural parameters
options.nnLayers    = { struct('type','input', 'nNeuron',784),...
                        struct('type','fc',    'nNeuron',800, 'act','maxout', 'poolSize',2),...                        
                        struct('type','output','nNeuron',10,  'act','softmax')};

% optional parameters - also set by default
options.alpha        = 0.1;   % Learning rate
options.lambdaL2     = 0;     % Weight decay parameter (for L2 regularization)
options.lambdaL1     = 0;     % Weight decay parameter (for L1 regularization)
options.nEpochs      = 20;    % 5 full sweeps over dataset
options.dropOutRatio = 0;     % Set half of the neurons to zero in hidden layers
options.useAdaDelta  = false; % it seems adaDelta does not help for maxout
% options.useGPU       = true;

% initialize object and train using gpu
NN  = NeuralNet(options);
NN.train_model;

% now test trained models with third nn
predTE = NN.predict_samples(testData);
predCV = NN.predict_samples(options.cvData);
fprintf('CV   Accuracy: %f%%\n', 100*mean(predCV(:)  == [(1:10) * options.cvLabels ]')); % 
fprintf('Test Accuracy: %f%%\n', 100*mean(predTE(:)  == testLabels(:))); % 

% dropout=0 alpha=1e-1 adadelta=false maxout-400 nEpochs=20
% CV   Accuracy: 94.300000%
% Test Accuracy: 95.211111%
% dropout=0 alpha=1e-1 adadelta=false maxout-800 nEpochs=20
% CV   Accuracy: 94.300000%
% Test Accuracy: 95.211111%
% dropout=0 alpha=1e-1 adadelta=false maxout-800,800 nEpochs=20
% CV   Accuracy: 94.400000%
% Test Accuracy: 95.722222%
clear NN

