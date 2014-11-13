classdef SelfTaughtLearner < handle
%==========================================================================
% Class for Self-taught Learner
%   
%   - Has a Sparse Autoencoder object 
%   - Has a Softmax Regressor object
%
%   Self-taught learner has a sparse autoencoder which is used to learn
%   features in an unsupervised fashion and has a softmax regressor which
%   is used as a classifier over the learned features by sparse
%   autoencoder. 
%
% orhanf
%==========================================================================

    
    
    properties
        
        SA;             % Sparse Autoencoder object
        SM;             % Softmax Regressor object
        
        optionsSA;      % Initialization options for sparse autoencoder
        optionsSM;      % Initialization options for softmax regressor        
       
        silent;         % Ddisplay cost in each iteration etc.     
        
    end
    
    
    methods

        
        %==================================================================
        % constructor
        %==================================================================
        function obj = SelfTaughtLearner(options)            
                                   
            obj.silent   = false;       
            
            if nargin>0 && isstruct(options)
            
                if isfield(options,'optionsSA'), obj.optionsSA = options.optionsSA; end
                if isfield(options,'optionsSM'), obj.optionsSM = options.optionsSM; end                                
                                                           
                if isfield(options,'silent'), obj.silent = options.silent; end
            
                % Initialize sparse autoencoder object
                if isfield(options,'SA')
                    obj.SA = options.SA; 
                else            
                    obj.SA = SparseAutoencoder(obj.optionsSA);        
                end
                
                % Initialize softmaxRegressor object
                if isfield(options,'SM') 
                    obj.SM = options.SM; 
                else                                
                    obj.SM = SoftmaxRegressor(obj.optionsSM);        
                end     
                
                % Inýt softmax regressors training data
                if isfield(options,'trainData')
                    obj.SM.x = options.trainData;
                end
                
                % Inýt softmax regressors training labels
                if isfield(options,'trainLabels')
                    obj.SM.y = options.trainLabels;
                end
                
                % Inýt autoencoders unlabeled data
                if isfield(options,'unlabeledData')
                    obj.SA.x = options.unlabeledData;
                end                
            end
        end
        
        
        %==================================================================
        % Train model using given minimization function, in order to train
        % sparse autoencoder we use raw unlabeled data, after that step
        % raw input training data is fed to sparse autoencoder to extract
        % training features for softmax regressor and used to train softmax
        % regressor. 
        %==================================================================
        function [optthetaSA, optthetaSM] = train_model(obj)
                        
            optthetaSA = [];
            optthetaSM = [];
            
            if isModelValid(obj)                                

                [optthetaSA, costSA] = obj.SA.train_model;

                % Extract training features using raw training data
                trainFeatures = obj.feed_forward_autoencoder(optthetaSA, obj.SA.hiddenSize, obj.SA.visibleSize, ...
                                       obj.SM.x);
                
                % This is crucial, replace raw data with training features,
                % design may be complicated bu suitable for space issues
                obj.SM.x = trainFeatures;
                                   
                [optthetaSM, costSM] = obj.SM.train_model;
                
            end        
        end       
        
        
        %==================================================================
        %   Feed forward step for the sparse autoencoder using learned
        %   parameters.
        %==================================================================
        function activation = feed_forward_autoencoder(obj, theta, hiddenSize, visibleSize, data)
        
            sigmoid = inline('1.0 ./ (1.0 + exp(-z))'); % logistic function
            
            W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
            b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);

            nSamples = size(data,2);

            a1 = data;                                          % a1 is equal to inputs x
            z2 = W1 * a1 + repmat(b1,1,size(a1,2));             % z2 is weigted sum of a1
            a2 = reshape(sigmoid(z2(:)), size(W1,1), nSamples); % a2 is sigmoid output of z3
            activation = a2;
                        
        end
        
        
        %==================================================================
        %   Predict new samples with trained SA model, test data for this
        %   function must be raw input data hence it will be fed to sparse
        %   autoencoder.
        %==================================================================
        function pred = predict_samples(obj, testData)
            
            pred = [];
            
            if ~isempty(obj.SA.theta) && ~isempty(obj.SA.hiddenSize) && ...
                    ~isempty(obj.SA.visibleSize) && ~isempty(testData)
                
                testfeatures = obj.feed_forward_autoencoder(obj.SA.theta, obj.SA.hiddenSize, obj.SA.visibleSize, ...
                                           testData); 

                pred = obj.SM.predict_samples(testfeatures);
            end
        end
        
        
    end
    
    
    
    methods(Hidden)
        
        
        %==================================================================
        % checks model parameters for self-taught learner
        %==================================================================
        function isValid = isModelValid(obj)            
            isValid = false;
                        
            if isempty(obj.SA),           return; end
            if isempty(obj.SM),           return; end
            
            isValid = true;
        end
        
    end
    

end

