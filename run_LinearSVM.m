%% Load dataset and labels
addpath(genpath('minFunc_2012/'));
addpath(genpath('data/'));
images = loadMNISTImages('data/train-images.idx3-ubyte');
labels = loadMNISTLabels('data/train-labels.idx1-ubyte');

% images = images(:,1:1000);
% labels = labels(1:1000);

labels(labels==0) = 10; % Remap 0 to 10 

%% Check gradient correctness

% Set model parameters with an options struct
optionsSVM.C = 1;
optionsSVM.x = rand(100,5); 
optionsSVM.y = [1 1 1 2 2];     
optionsSVM.method = 0;  
optionsSVM.lambdaL1 = 1e-3;

% Initialize softmaxRegressor and linear SVM objects
SVM0 = LinearSVM(optionsSVM);
SVM0.check_numerical_gradient; % should get  1.7720e-12
clear SVM0

%% LinearSVM and Softmax object initializations

% Set model parameters with an options struct
optionsSM.lambda = 3e-3;
optionsSM.x      = images; 
optionsSM.y      = labels;     

optionsSVM.C = 5;
optionsSVM.x = images; 
optionsSVM.y = labels;     
optionsSVM.method = 0;  

% Initialize softmaxRegressor and linear SVM objects
SM   = SoftmaxRegressor(optionsSM);
SVM1 = LinearSVM(optionsSVM);


%% Training

% Set minimization function for both linearSVM and softmax objects
foptionsSM.Method  = 'lbfgs';                          
foptionsSM.maxIter = 100;	  
foptionsSM.display = 'on';

foptionsSVM.Method  = 'lbfgs';                          
foptionsSVM.maxIter = 500;	  
foptionsSVM.display = 'on';

SM.minfunc     = @(funJ,theta)minFunc(funJ, theta, foptionsSM);
SVM1.minfunc = @(funJ,theta)minFunc(funJ, theta, foptionsSVM);

% SM.train_model;
SVM1.train_model;


%% Test
% Predict new samples with trained model, which is calculated by libORF as 

% Accuracy 91.442%
pred = SVM1.predict_samples(images);
acc  = mean(labels(:) == pred(:));
fprintf('SVM Accuracy: %0.3f%%\n', acc * 100);

% Accuracy: 93.895%
% pred = SM.predict_samples(images);
% acc  = mean(labels(:) == pred(:));
% fprintf('SM Accuracy: %0.3f%%\n', acc * 100);

clear SVM1 SM
