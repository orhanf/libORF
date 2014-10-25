%%
% This script presents a demo run for contractive autoencoders with various
% architectures and options on MNIST. 
% 


%% Load data 
addpath(genpath('data/'));
images = loadMNISTImages('data/train-images.idx3-ubyte');


%% Let us start by checking our gradient calculation numerically

% Initialize sparse autoencoder object
CAE0 = ContractiveAutoencoder(struct(  'visibleSize',20,... % input size
                                        'hiddenSize' ,3,... % hidden size
                                        'lambda'     ,0,...
                                        'contrLevel' ,1,...
                                        'addBias'    ,false,...                                          
                                        'x'          ,rand(20,100)));

% load commonVars.mat                                
% CAE0.x = x;
% CAE0.W1 = W1;

% this should be very small, we obtained 3.0328e-10                                
CAE0.check_numerical_gradient;
                                
clear CAE0


%% First let us train a standard contractive autoencoder with a default 
%   setting. Train using SGD for 100 epochs with momentum. Hidden and
%   output activations are both sigmoid and cost function is mse.

% Initialize sparse autoencoder object
CAE1 = ContractiveAutoencoder(struct(  'visibleSize',28*28,...  % input size
                                        'hiddenSize' ,196,...   % hidden size
                                        'lambda'     ,0,...     % weight decay
                                        'contrLevel' ,.3,...    % contraction level
                                        'momentum'   ,0,...     % grad speed
                                        'addBias'    ,false,...  
                                        'alpha'      ,1,...     % learning rate
                                        'nEpochs'    ,100,...
                                        'x'          ,images(:,1:10000)));

% Start training our DA
CAE1.train_model;

% Display resulting hidden layer features
Utility2.display_images(CAE1.W1'); 

clear CAE1


%% Now let us train a standard contractive autoencoder with a second order 
%   method. Train using BGD. Hidden and output activations are both 
%   sigmoid and cost function is mse.

addpath(genpath('minFunc_2012'));

% Initialize sparse autoencoder object
CAE2 = ContractiveAutoencoder(struct(  'visibleSize',28*28,... % input size
                                        'hiddenSize' ,196,...   % hidden size
                                        'lambda'     ,.0,...   % weight decay 
                                        'contrLevel' ,0.3,...   % drop level
                                        'addBias'    ,false,...  
                                        'minfunc'    ,@(funJ,theta)minFunc(funJ, theta, struct('maxIter',100,'method','cg')),...
                                        'x'          ,images(:,1:1000)));

% Start training our DA
CAE2.train_model;

% Display resulting hidden layer features
Utility2.display_images(CAE2.W1'); 

clear CAE2


