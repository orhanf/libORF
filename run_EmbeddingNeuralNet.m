%% load data for word embedding problem

[data] = Utility.load_fields('data/wordEmbeddingDataset.mat',{'data'});
trainData   = data.trainData(1:3,:);
trainLabels = data.trainData(4,:);
testData    = data.testData(1:3,:);
testLabels  = data.testData(4,:);
cvData      = data.validData(1:3,:);
cvLabels    = data.validData(4,:);
vocabulary  = data.vocab;
clear data


%% set parameters of embedding neural net object and initialize it 

% necessary parameters
options.trainData   = trainData;   % Training data
options.trainLabels = trainLabels; % Training data
options.vocabulary  = vocabulary;  % Dictionary (optional)
options.hiddenSizes = [200];       % Array for hidden unit sizes, excluding input, embedding and output layers
options.embedSize   = 50;          % Number of units in each of the embedding layer parts
options.nEmbeds     = 3;           % Number of embedding replications size(trainData,1)-1
options.outputSize  = 250;         % Number of units in the output layer as it is softmax       
options.inputSize   = 250;         % Number of units in one of the input layers (note there are nEmbeds*inputSize input units in total)

% optional parameters
options.nEpochs     = 10;        % Number of passes through the dataset              
options.alpha       = 0.1;       % Learning rate
options.lambda      = 0;         % No weight decay
options.useAdaDelta = true;      % Employ a scheduling for learning rate
options.keepRecords  = 0;        % This will make training faster with less diagnosis tools

% initialize object
ENN = EmbeddingNeuralNet(options);


%% train embedding neural net using mini-batch gradient descent
tic
ENN.train_model;
fprintf(1,'Elapsed time in training [%f]sec\n',toc);

%% evaluate model on the test set

pred = ENN.predict_samples(testData);

% Classification Score - you should expect something around 36.900876%
fprintf('Test Accuracy: %f%%\n', 100*mean(pred(:) == testLabels(:)));


%% now let us get some close words in some vicinity of query word

% get 10 nearest words
ENN.get_closest_words('day',10)
ENN.get_closest_words('he',10)
ENN.get_closest_words('money',10)
ENN.get_closest_words('life',10)


%% calculate the distance between two words

ENN.get_word_distances('he','she')   % this should be around 1.3181
ENN.get_word_distances('he','years') % this should be around 6.0591


%% Try a deeper model and see the effect

options.hiddenSizes = [200,200]; % two hidden layers with equal sizes

% initialize object
ENN2 = EmbeddingNeuralNet(options);

tic
ENN2.train_model;
fprintf(1,'Elapsed time in training [%f]sec\n',toc);

% evaluate model on the test set

pred = ENN2.predict_samples(testData);

% Classification Score - you should expect something around 37.356124%
fprintf('Test Accuracy: %f%%\n', 100*mean(pred(:) == testLabels(:)));

ENN2.get_closest_words('day',10)

%% Try the same model with different non-linearities

options.hiddenSizes = [200,200];
options.hActFuns    = {'tanh','tanh'};

ENN3 = EmbeddingNeuralNet(options);

tic
ENN3.train_model;
fprintf(1,'Elapsed time in training [%f]sec\n',toc);

% evaluate model on the test set

pred = ENN3.predict_samples(testData);

% Classification Score - you should expect something around 32.348394%
fprintf('Test Accuracy: %f%%\n', 100*mean(pred(:) == testLabels(:)));

ENN3.get_closest_words('day',10)

%% Try the same model with different non-linearities

options.hiddenSizes = [400];
options.hActFuns    = {'relu'};

ENN3_2 = EmbeddingNeuralNet(options);

tic
ENN3_2.train_model;
fprintf(1,'Elapsed time in training [%f]sec\n',toc);

% evaluate model on the test set

pred = ENN3_2.predict_samples(testData);

% Classification Score - you should expect something around 36.761295%
fprintf('Test Accuracy: %f%%\n', 100*mean(pred(:) == testLabels(:)));

ENN3_2.get_closest_words('day',10)

clear ENN3_2

%% Now let us try a second order method with a Batch Gradient Descent 
%   regime. Note that, since our dataset is large, this will take long

addpath(genpath('minFunc_2012')); % add optimizer to Matlab path

options.hiddenSizes    = 200;
options.hActFuns       = {'sigmoid'};
options.trainingRegime = 2;

% Set minimization function for SA object
foptions.Method  = 'lbfgs';                          
foptions.maxIter = 100;	  
foptions.display = 'on';
options.minfunc  = @(funJ,theta)minFunc(funJ, theta, foptions);

ENN4 = EmbeddingNeuralNet(options);

tic
ENN4.train_model;
fprintf(1,'Elapsed time in training [%f]sec\n',toc);

% evaluate model on the test set

pred = ENN4.predict_samples(testData);

% Classification Score - you should expect something around 37.993901%
% after [3664.848964]sec ~ 1hour
fprintf('Test Accuracy: %f%%\n', 100*mean(pred(:) == testLabels(:)));

ENN4.get_closest_words('day',10)


%% Employ cross validation data and stop training when validation error 
%   increases, simply early stopping. Note that, this will slow down
%   training. After each epoch, CV data will be used for validation and if
%   the CV data is too much then this will result a slowing down. Early
%   stopping does not mean stopping training early but taking the
%   parameters that has the lowest CV error after training.


options.trainingRegime  = 1;
options.cvData          = cvData;   % Cross validation data
options.cvLabels        = cvLabels; % Cross validation labels
options.isEarlyStopping = true;
options.nEpochs         = 20;

ENN5 = EmbeddingNeuralNet(options);

tic
ENN5.train_model;
fprintf(1,'Elapsed time in training [%f]sec\n',toc);

pred = ENN5.predict_samples(testData);

% Classification Score - you should expect something around 37.457052%
% after [451.116359]sec
fprintf('Test Accuracy: %f%%\n', 100*mean(pred(:) == testLabels(:)));


%% Faster version
options.alpha        = .1;
options.nEpochs      = 10;
options.keepRecords  = 0;

ENN6 = EmbeddingNeuralNet(options);

tic
ENN6.train_model;
fprintf(1,'Elapsed time in training [%f]sec\n',toc);

pred = ENN6.predict_samples(testData);

% Classification Score - you should expect something around  37.040457%
% after [177.278588]sec
fprintf('Test Accuracy: %f%%\n', 100*mean(pred(:) == testLabels(:)));


%% Empirical stuff - try minibatch L-BFGS

addpath(genpath('minFunc_2012'));

options.trainingRegime = 3;
options.batchSize      = 1000;
options.minfunc  = @(funJ,theta)minFunc(funJ, theta, struct('Method','lbfgs',...
                                                            'maxIter',100,...
                                                            'display','off'));
ENN7 = EmbeddingNeuralNet(options);

tic
ENN7.train_model;
fprintf(1,'Elapsed time in training [%f]sec\n',toc);

pred = ENN7.predict_samples(testData);

% Classification Score - you should expect something around  
% after 
fprintf('Test Accuracy: %f%%\n', 100*mean(pred(:) == testLabels(:)));


%% Again mini-batch L-BFGS

addpath(genpath('minFunc_2012'));

options.nEpochs        = 5;
options.trainingRegime = 3;
options.batchSize      = 5000;
options.useAdaDelta    = false;
options.momentum       = 0;
options.minfunc  = @(funJ,theta)minFunc(funJ, theta, struct('Method','lbfgs',...
                                                            'maxIter',50,...
                                                            'display','off'));
ENN8 = EmbeddingNeuralNet(options);

tic
ENN8.train_model;
fprintf(1,'Elapsed time in training [%f]sec\n',toc);

pred = ENN8.predict_samples(testData);

% Classification Score - you should expect something around  
% after 
fprintf('Test Accuracy: %f%%\n', 100*mean(pred(:) == testLabels(:)));

ENN8.get_closest_words('day',10)


