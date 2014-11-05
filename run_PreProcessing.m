clear all
close all
clc

addpath(genpath(pwd));
load data/sampleImagesRaw.mat
x = sampleImagesRaw;

%% Display some random images
figure('name','Raw images');
randsel = randi(size(x,2),200,1); 
Utility2.display_images(x(:,randsel));


%% Zero mean normalization for image patches
xZeroMean = PreProcessing.normalize_zero_mean(x);
figure('name','Zero mean images');
Utility2.display_images(xZeroMean(:,randsel));


%% Apply PCA and check covariance of rotated data

[U,S] = PreProcessing.apply_PCA(x);
xRot  = U' * x; % rotated version of the data. 
covar = PreProcessing.calculate_covariance_matrix(xRot);

% Visualise the covariance matrix. You should see a line across the
% diagonal against a blue background.
figure('name','Visualisation of covariance matrix');
imagesc(covar);


%% Reduce dimension by specifying retaining ratio

retainRatio   = 0.99;
[xRot1 xHat1] = PreProcessing.reduce_dimension_by_retaining(x,retainRatio);

retainRatio   = 0.9;
[xRot2 xHat2] = PreProcessing.reduce_dimension_by_retaining(x,retainRatio);

figure('name',['PCA processed images ',sprintf('(%d / %d dimensions)', size(xRot1,1), size(x, 1)),'']);
Utility2.display_images(xHat1(:,randsel));
figure('name',['PCA processed images ',sprintf('(%d / %d dimensions)', size(xRot2,1), size(x, 1)),'']);
Utility2.display_images(xHat2(:,randsel));


%% Reduce dimension by specifying number of features

nFeatures     = 50;
[xRot3 xHat3] = PreProcessing.reduce_dimension_by_nFeatures(x,nFeatures);

nFeatures     = 10;
[xRot4 xHat4] = PreProcessing.reduce_dimension_by_nFeatures(x,nFeatures);

figure('name',['PCA processed images ',sprintf('(%d / %d dimensions)', size(xRot3,1), size(x, 1)),'']);
Utility2.display_images(xHat3(:,randsel));
figure('name',['PCA processed images ',sprintf('(%d / %d dimensions)', size(xRot4,1), size(x, 1)),'']);
Utility2.display_images(xHat4(:,randsel));


%% PCA with whitening and regularisation

epsilon     = 0.1;
xPCAWhite1  = PreProcessing.apply_PCA_whitening(x,epsilon);

epsilon     = 0;
xPCAWhite2  = PreProcessing.apply_PCA_whitening(x,epsilon);

covar1      = PreProcessing.calculate_covariance_matrix(xPCAWhite1);
covar2      = PreProcessing.calculate_covariance_matrix(xPCAWhite2);

% Visualise the covariance matrix. You should see a red line across the
% diagonal against a blue background.
figure('name','Visualisation of covariance matrix with regularization');
imagesc(covar1);
figure('name','Visualisation of covariance matrix without regularization');
imagesc(covar2);


%% ZCA whitening with regularization
epsilon   = 0.1;
xZCAWhite = PreProcessing.apply_ZCA_whitening(x,epsilon);

% Visualise the data, and compare it to the raw data.
% You should observe that the whitened images have enhanced edges.
randsel = randi(size(xZCAWhite,2),200,1); 
figure('name','ZCA whitened images');
Utility2.display_images(xZCAWhite(:,randsel));
figure('name','Raw images');
Utility2.display_images(x(:,randsel));


%% ZCA whitening and dimension reduction 

retainRatio   = 0.99;
[xRot1 xHat1] = PreProcessing.reduce_dimension_by_retaining(xZCAWhite,retainRatio);

figure('name',['ZCA whitened images ',sprintf('(%d / %d dimensions)', size(xRot1,1), size(xZCAWhite, 1)),'']);
Utility2.display_images(xHat1(:,randsel));






