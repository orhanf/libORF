%%
% This script presents a demo run for denoising autoencoders with various
% architectures and options on MNIST. 
% 


%% Load data 
addpath(genpath('data/'));
images = loadMNISTImages('data/train-images.idx3-ubyte');


%% Let us start by checking our gradient calculation numerically

% Initialize sparse autoencoder object
DA0 = DenoisingAutoencoder(struct(  'visibleSize',20,... % input size
                                    'hiddenSize' ,3,...   % hidden size
                                    'drop'       ,0,...    % drop level
                                    'lambda'     ,0,...
                                    'x'          ,rand(20,10)));

% this should be very small, we obtained 4.6655e-10                                
DA0.check_numerical_gradient;
                                
clear DA0


%% First let us train a standard denoising autoencoder with a default 
%   setting. Train using SGD for 100 epochs with momentum. Hidden and
%   output activations are both sigmoid and cost function is mse.

% Initialize sparse autoencoder object
DA1 = DenoisingAutoencoder(struct(  'visibleSize',28*28,... % input size
                                    'hiddenSize' ,196,...   % hidden size
                                    'drop'       ,.5,...    % drop level
                                    'momentum'   ,.7,...    % grad speed
                                    'alpha'      ,1,...     % learning rate
                                    'x'          ,images(:,1:10000)));

% Start training our DA
DA1.train_model;

% Display resulting hidden layer features
Utility2.display_images(DA1.W1'); 

clear DA1

%% Now let us train a standard denoising autoencoder with a default 
%   setting but with a second order method this time, namely; l-BFGS. 
%   Hidden and output activations are both sigmoid and cost function is mse

addpath(genpath('minFunc_2012'));

% Initialize sparse autoencoder object
DA2 = DenoisingAutoencoder(struct(  'visibleSize',28*28,... % input size
                                    'hiddenSize' ,196,...   % hidden size
                                    'lambda'     ,.03,...   % weight decay 
                                    'drop'       ,0.4,...   % drop level
                                    'minfunc'    ,@(funJ,theta)minFunc(funJ, theta, struct('maxIter',100)),...
                                    'x'          ,images(:,1:10000)));

% Start training our DA
DA2.train_model;

% Display resulting hidden layer features
Utility2.display_images(DA2.W1'); 

clear DA2


%% This time try tied weights meaning that W2 = W1 

% Initialize sparse autoencoder object
DA3 = DenoisingAutoencoder(struct(  'visibleSize',28*28,... % input size
                                    'hiddenSize' ,196,...   % hidden size
                                    'drop'       ,.5,...    % drop level
                                    'momentum'   ,.6,...    % grad speed
                                    'alpha'      ,1,...     % learning rate
                                    'tiedWeights',true,...  % W2 = W1
                                    'x'          ,images(:,1:10000)));

% Start training our DA
DA3.train_model;

% Display resulting hidden layer features
Utility2.display_images(DA3.W1'); 

clear DA3

%% Extend previous model by using adadelta,

% Initialize sparse autoencoder object
DA3_2 = DenoisingAutoencoder(struct(  'visibleSize',28*28,... % input size
                                      'hiddenSize' ,196,...   % hidden size
                                      'drop'       ,.4,...    % drop level
                                      'momentum'   ,.4,...    % grad speed
                                      'adaDeltaRho',.99,...    % decay rate
                                      'useAdaDelta',true,...  % employ adadelta
                                      'tiedWeights',true,...  % W2 = W1
                                      'x'          ,images(:,1:10000)));

% Start training our DA
DA3_2.train_model;

% plot some stats
DA3_2.plot_training_error;
[h, Frobenius_norm, L1_dist_to_I] = DA3_2.plot_feature_corrs

% Display resulting hidden layer features
Utility2.display_images(DA3_2.W1'); 

clear DA3_2


%% And try cross entropy with tied weights 

% Initialize sparse autoencoder object
DA4 = DenoisingAutoencoder(struct(  'visibleSize',28*28,... % input size
                                    'hiddenSize' ,196,...   % hidden size
                                    'drop'       ,.5,...    % drop level
                                    'momentum'   ,.5,...    % grad speed
                                    'alpha'      ,1,...     % learning rate
                                    'tiedWeights',true,...  % W2 = W1
                                    'errFun'     ,1,...     % 1-cross entropy
                                    'x'          ,images(:,1:10000)));

% Start training our DA
DA4.train_model;

% Display resulting hidden layer features
Utility2.display_images(DA4.W1'); 

clear DA4


%% Then, train a linear autoencoder. A linear autoencoder can be obtained 
%   either by setting 'vActFun' option to 3 or 'isLinearCost' to true.

% Initialize sparse autoencoder object
DA5 = DenoisingAutoencoder(struct(  'visibleSize',28*28,... % input size
                                    'hiddenSize' ,196,...   % hidden size
                                    'drop'       ,.2,...    % drop level
                                    'momentum'   ,0,...     % grad speed
                                    'alpha'      ,1,...     % learning rate
                                    'tiedWeights',true,...  % W2 = W1
                                    'vActFun'    ,3,...     % linear output
                                    'x'          ,images(:,1:10000)));

% Start training our DA
DA5.train_model;

% Display resulting hidden layer features
Utility2.display_images(DA5.W1'); 

clear DA5

%% Next, let us analyse the Jacobian and its singular values to assess 
%   the contraction amount of the encoding

% Initialize sparse autoencoder object
DA6 = DenoisingAutoencoder(struct(    'visibleSize',28*28,... % input size
                                      'hiddenSize' ,196,...   % hidden size
                                      'drop'       ,.5,...    % drop level
                                      'momentum'   ,.9,...    % grad speed
                                      'adaDeltaRho',.95,...    % decay rate
                                      'useAdaDelta',true,...  % employ adadelta
                                      'tiedWeights',true,...  % W2 = W1
                                      'x'          ,images(:,1:5000)));

% Start training our DA
tic
DA6.train_model;
toc

% plot some stats
DA6.plot_training_error;
[h, Frobenius_norm, L1_dist_to_I] = DA6.plot_feature_corrs

% Display resulting hidden layer features
Utility2.display_images(DA6.W1'); 

% get jacobian and plot its singular values
Jac = DA6.get_jacobian;
[U S V] = svd(Jac);
figure,plot(diag(S));

clear DA6

%% Then, let us employ GPU on previous model

% Initialize sparse autoencoder object
DA7 = DenoisingAutoencoder(struct(    'visibleSize',28*28,... % input size
                                      'hiddenSize' ,196,...   % hidden size
                                      'drop'       ,.5,...    % drop level
                                      'momentum'   ,.9,...    % grad speed
                                      'adaDeltaRho',.95,...   % decay rate
                                      'useAdaDelta',true,...  % employ adadelta
                                      'tiedWeights',true,...  % W2 = W1
                                      'useGPU'     ,true,...  % as the name refers
                                      'x'          ,images(:,1:5000)));

                                  
% Start training our DA
tic
DA7.train_model;   
toc                               

% Display resulting hidden layer features
Utility2.display_images(DA7.W1');                                   
                                  
clear DA7  

%% Now let us add some sparsity term to the cost function

% Initialize sparse autoencoder object
DA8 = DenoisingAutoencoder(struct(    'visibleSize',28*28,... % input size
                                      'hiddenSize' ,196,...   % hidden size
                                      'hActFun'    ,0,...
                                      'drop'       ,.5,...    % drop level
                                      'momentum'   ,.9,...    % grad speed
                                      'adaDeltaRho',.95,...   % decay rate
                                      'useAdaDelta',true,...  % employ adadelta
                                      'tiedWeights',true,...  % W2 = W1
                                      'rho'        ,.03,...   % sparsity target
                                      'beta'       ,.25,...   % sparsity term effect   
                                      'x'          ,images(:,1:5000)));

                                  
% Start training our DA
tic
DA8.train_model;   
toc                               

% Display resulting hidden layer features
Utility2.display_images(DA8.W1');
          
clear DA8
