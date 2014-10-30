%% ------------------------------------------------------------------------
%  Multi-nomial Naive Bayes Example - Document classification using 
%  bag-of-words approach

disp('Classification demo for spam documents with Multinomial Naive Bayes:');

% read training data (discrete) and labels (discrete)
nSamplesTR = 700;   % number of e-mails to be classified as spam/non-spam
nFeatures  = 2500;  % number of words in the dictionary
M = dlmread('data/train-features.txt', ' ');
train_matrix = full(sparse(M(:,1), M(:,2), M(:,3), nSamplesTR, nFeatures));
train_labels = dlmread('data/train-labels.txt');

clear M

% Initialize Naive Bayes object

% Set model parameters of naive bayes object with an options struct
options.train_matrix = train_matrix;
options.train_labels = train_labels;
options.isGaussianNB = false;

% Initialize multinomial naive bayes object
MNB = NaiveBayesGM(options);

% Fit a Model Using Training Data
MNB.train_model;

% Load Test Data 
nSamplesTE = 260;
M = dlmread('data/test-features.txt', ' ');
test_matrix = full(sparse(M(:,1), M(:,2), M(:,3), nSamplesTE, nFeatures));
test_labels = dlmread('data/test-labels.txt');

clear M 

% Test Model
predicted_labels = MNB.predict_labels(test_matrix);


% Compute test error

%Print out error statistics on the test set
numdocs_wrong  = sum(xor(predicted_labels, test_labels))
fraction_wrong = numdocs_wrong/nSamplesTE
overall_accuracy = 1 - fraction_wrong 
confusion_matrix = confusionmat(test_labels,predicted_labels)

clear all


%% ------------------------------------------------------------------------
% Gaussian Naive Bayes Example - fisheriris dataset

disp('Classification demo for fisheriris with Gaussian Naive Bayes:');

load fisheriris
train_matrix = meas;
train_labels = cell2mat(cellfun(@(x)find(cellfun(@(y)strcmp(x,y),unique(species))),species,'UniformOutput',false));

clear species meas


% Initialize Gaussian Naive Bayes object

% Set model parameters of naive bayes object with an options struct
options.train_matrix = train_matrix;
options.train_labels = train_labels;
options.isGaussianNB = true;

% Initialize gaussian naive bayes object
GNB = NaiveBayesGM(options);

% Fit a Model Using Training Data
GNB.train_model;

% Test Model
predicted_labels = GNB.predict_labels(train_matrix);

% Compute test error

%Print out error statistics on the test set
numsamples_wrong = sum(abs(predicted_labels - train_labels)>0)
fraction_wrong   = numsamples_wrong/size(train_matrix,1)
overall_accuracy = 1 - fraction_wrong 
confusion_matrix = confusionmat(train_labels,predicted_labels)

clear all


%% ------------------------------------------------------------------------
% Kernel Naive Bayes Example - fisheriris dataset

disp('Classification demo for fisheriris with Kernel Naive Bayes:');


load fisheriris
train_matrix = meas;
train_labels = cell2mat(cellfun(@(x)find(cellfun(@(y)strcmp(x,y),unique(species))),species,'UniformOutput',false));

clear species meas

% Initialize Gaussian Naive Bayes object

% Set model parameters of naive bayes object with an options struct
options.train_matrix = train_matrix;
options.train_labels = train_labels;
options.isKernelNB = true;

% Initialize gaussian naive bayes object
KNB = NaiveBayesGM(options);

% Fit a Model Using Training Data
KNB.train_model;

% Test Model
predicted_labels = KNB.predict_labels(train_matrix);

% Compute test error

%Print out error statistics on the test set
numsamples_wrong = sum(abs(predicted_labels - train_labels)>0)
fraction_wrong   = numsamples_wrong/size(train_matrix,1)
overall_accuracy = 1 - fraction_wrong 
confusion_matrix = confusionmat(train_labels,predicted_labels)

clear all

