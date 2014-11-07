%%

%% Load data

addpath(genpath(pwd))

% we want experiments to be replicable, thus we need a deterministic
% randomness (!) source, for the results to be same- uncomment two lines
randSource = load('data/randomnessSource.mat');
randSource = randSource.randSource;


%% ========================================================================
% Let us first train an RBM to model a smaller set of hand-written digits
%==========================================================================

% load USPS dataset
data        = Utility.load_fields('./data/USPS.mat',{'data'});
trainData   = data.training.inputs;       % Training data
testData    = data.test.inputs;           % Test data
testLabels  = (1:10) * data.test.targets; % Test labels
clear data

% initialize rbm object
rbm1 = RBM();

% set proper fields   
rbm1.x = trainData;
rbm1.hiddenSize   = 300;   % Note that we learn a large #of hidden units
rbm1.visibleSize  = 16*16; % Patch size for USPS
rbm1.nEpochs      = 100;
rbm1.learningRate = 0.1;
rbm1.reduceNoise  = false; % Use binary states
rbm1.momentum     = 0.9;   % Use momentum
rbm1.batchSize    = 100;   % Update for 100 samples
rbm1.weightDecay  = 0;     % No Regularization
rbm1.statesMode   = 'RB';
rbm1.randSource   = randSource; 
rbm1.addBias      = false; % Do not add bias 
rbm1.useCondProb  = false; % Use conditional probability for improved training

% train model generatively
[model, costs] = rbm1.train_model;

% visualize hidden weights
figure, display_network(rbm1.rbm_W',false);
figure, hold on;
plot(costs, 'b');
legend('training');
ylabel('loss');
xlabel('iteration number');
hold off;

% unwroll to a neural net

% train neural net 

% perform prediction

clear rbm1 costs testData trainData testLabels


%% ========================================================================
% Now train an RBM to model MNIST hand-written digits
%==========================================================================

% Load MNIST dataset files
trainData = double(Utility.load_fields('./data/mnist_uint8.mat',{'train_x'}))' / 255;

% initialize rbm object
rbm2 = RBM();

% set proper fields   
rbm2.x = trainData;
rbm2.hiddenSize   = 256;   % Note that we learn a large #of hidden units
rbm2.visibleSize  = 28*28; % Patch size for MNIST
rbm2.nEpochs      = 10;
rbm2.learningRate = 1;
rbm2.reduceNoise  = true;  % Use binary states
rbm2.momentum     = 0;     % No momentum
rbm2.batchSize    = 100;   % Update for every 100 samples
rbm2.weightDecay  = 0.001; 
rbm2.statesMode   = 'RB';
rbm2.addBias      = true;  % Add bias 
rbm2.useCondProb  = false; % Dont use conditional probability for improved training

% train model generatively
[model, costs] = rbm2.train_model;

% visualize hidden weights
Utility2.display_images(rbm2.rbm_W');




