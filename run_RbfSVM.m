%% Load dataset and labels
addpath(genpath('minFunc_2012/'));
addpath(genpath('data/'));
imagesOrj = loadMNISTImages('data/train-images.idx3-ubyte');
labelsOrj = loadMNISTLabels('data/train-labels.idx1-ubyte');

images = imagesOrj(:,1:2000);
labels = labelsOrj(1:2000);

labels(labels==0) = 10; % Remap 0 to 10 


%% rbf-SVM object initialization

% Set model parameters with an options struct
optionsSVM.C = 1; % this is actually lambda, C will be set to 1/lambda, so you should set this to 1/<paramYouChoose>
optionsSVM.g = 2; % this is the sigma parameter in the radial basis function
optionsSVM.x = images; 
optionsSVM.y = labels;     
optionsSVM.method = 0;  

% Initialize 
SVM = RbfSVM(optionsSVM);


%% Training

% Set minimization function for rbfSVM 
foptionsSVM.Method  = 'lbfgs';                          
foptionsSVM.maxIter = 500;	  
foptionsSVM.display = 'on';

SVM.minfunc = @(funReg,theta,kernel,nClasses,funJ,C)minFunc(funReg, theta, foptionsSVM, kernel, nClasses, funJ,C);

SVM.train_model;


%% Test

% Predict new samples with trained model, which is calculated by libORF as 
% Accuracy 83.400%
pred = SVM.predict_samples(imagesOrj(:,4001:4500));
acc  = mean(labelsOrj(4001:4500) == pred(:));
fprintf('rbf SVM Accuracy: %0.3f%%\n', acc * 100);

clear SVM


%% Compare it with a linear SVM

options2.C = .05;
options2.x = images; 
options2.y = labels;     

SVM1 = LinearSVM(options2);
SVM1.minfunc = @(funJ,theta)minFunc(funJ, theta, foptionsSVM);
SVM1.train_model;

% Accuracy 79.200%
pred = SVM1.predict_samples(imagesOrj(:,4001:4500));
acc  = mean(labelsOrj(4001:4500) == pred(:));
fprintf('linear SVM Accuracy: %0.3f%%\n', acc * 100);

clear SVM1

