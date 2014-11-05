classdef StackedAutoencoder < handle
%==========================================================================
% Class for Stacked Sparse Autoencoders with Softmax in output layer
%   
%   - Has (n) number of Sparse Autoencoder object(s) 
%   - Has either a Softmax Regressor object or,
%                a Linear SVM object
%
%   Self-taught learner has a sparse autoencoder which is used to learn
%   features in an unsupervised fashion and has a softmax regressor which
%   is used as a classifier over the learned features by sparse
%   autoencoder. Note that using only 1 autoencoder in the stack is the
%   same as using a @SelfTaughtLearner, use this class if your aim is to
%   employ deep-learning. 
%
%   In case a Linear SVM object is selected as the output layer, libORF
%   does not add bias for SVM training by default, because input to SVM
%   layer is the output of the last layer autoencoder and autoencoder
%   implementation in libORF inherently adds bias to all hidden units.
%
% orhanf - (c) 2012 Orhan FIRAT <orhan.firat@ceng.metu.edu.tr>
%==========================================================================
    
    
    properties
        
        depth;       % Number of layers in the model excluding input and output layers
        nClasses;    % Number of classes for prediction
           
        SA;          % Array of Sparse Autoencoder object(s)
        SM;          % Softmax Regressor object  
        SVM;         % Linear SVM object
        
        trainData;   % Training data
        trainLabels; % Training labels
        
        lambda;      % Regularization parameter for fine-tuning
        minfunc;     % Optimizer function handle for fine-tuning
        netconfig;   % Stack configuration
        
        stackedAETheta;    % Initial model parameters
        stackedAEOptTheta; % Fine-tuned model parameters
                
        silent;      % Display cost in each iteration etc.     
        svmFlag;     % Indicator for using linear SVM as the last layer
    end
    
    
    methods

        
        %==================================================================
        % constructor
        %==================================================================
        function obj = StackedAutoencoder(options)            
                                   
            obj.silent   = false;       
            obj.lambda   = 1e-4;
            obj.svmFlag  = false;
            
            if nargin>0 && isstruct(options)
                          
                if isfield(options,'stackedAETheta'),    obj.stackedAETheta = options.stackedAETheta;       end
                if isfield(options,'stackedAEOptTheta'), obj.stackedAEOptTheta = options.stackedAEOptTheta; end
                                                
                if isfield(options,'trainData'),   obj.trainData = options.trainData;     end
                if isfield(options,'trainLabels')
                    obj.trainLabels = options.trainLabels;
                    obj.nClasses    = max(obj.trainLabels);
                end
                
                if isfield(options,'lambda'),     obj.lambda = options.lambda;       end
                if isfield(options,'minfunc'),    obj.minfunc = options.minfunc;     end
                if isfield(options,'netconfig'),  obj.netconfig = options.netconfig; end
                
                % Initialize sparse autoencoder objects
                if isfield(options,'SA')
                    obj.SA = options.SA; 
                elseif isfield(options,'optionsSA') % this part is tricky  
                    
                    obj.depth = numel(options.optionsSA);                    
                    obj.SA    = SparseAutoencoder.empty();
                    
                    for i=1:obj.depth
                        obj.SA(i) = SparseAutoencoder(options.optionsSA{i});
                        if i>1,	obj.SA(i).visibleSize = obj.SA(i-1).hiddenSize; end
                    end
                end
                
                % Initialize softmaxRegressor object
                if isfield(options,'SM') 
                    obj.SM = options.SM; 
                elseif isfield(options,'optionsSM')            
                    obj.SM = SoftmaxRegressor(options.optionsSM);        
                end                                                
                
                % Initialize linear SVM object if specified
                if isfield(options,'SVM') 
                    obj.SVM = options.SVM; 
                    obj.svmFlag = true;
                elseif isfield(options,'optionsSVM')            
                    obj.SVM = LinearSVM(options.optionsSVM);      
                    obj.SVM.addBias = false;
                    obj.svmFlag = true;
                end
                
                if obj.svmFlag 
                    fprintf('Ignoring softmax object and using linear SVM!\n');
                end
                
                if isfield(options,'silent'), obj.silent = options.silent; end
                
            end
        end
        
        
        %==================================================================
        % Train model, "Greedy Layer-wise Training"
        %==================================================================
        function opttheta = train_model(obj)
                   
            opttheta = [];
            
            if isModelValid(obj)                                
                
                trainFeatures = obj.trainData;
                
                for i=1:obj.depth
                    
                    % training data is changed in each iteration
                    obj.SA(i).x = trainFeatures;
                    
                    % train sparse autoencoder i^th in the  stack
                    obj.SA(i).train_model;

                    if ~isempty(obj.SA(i).theta)                         
                        % Extract training features using autoencoder activations
                        trainFeatures = obj.feed_forward_autoencoder(obj.SA(i).theta, obj.SA(i).hiddenSize, obj.SA(i).visibleSize, trainFeatures);
                    end                                        
                end
                                    
                % This is crucial, replace raw data with training features,
                % design may be complicated bu suitable for space issues
                if obj.svmFlag
                    obj.SVM.x = trainFeatures;
                    obj.SVM.y = obj.trainLabels;                
                    obj.SVM.train_model;
                else                    
                    obj.SM.x = trainFeatures;
                    obj.SM.y = obj.trainLabels;                
                    obj.SM.train_model;
                end
                
                % Convert parameters in a vectorized form and stack 
                % structure in a struct called netconfig
                [obj.stackedAETheta, obj.netconfig] = init_stack(obj);  
                
                opttheta = obj.stackedAETheta;                               
            end   
        end
        
        
        %==================================================================
        % Train model, "Fine-Tuned Training"
        %==================================================================
        function [opttheta, cost] = fine_tune_model(obj)
            opttheta = [];
            cost     = [];
            if isModelValid_fineTune(obj)    
                if obj.svmFlag
                    
                    % arrange label vector to a matrix and labels to [-1,1]
                    M = eye(obj.nClasses);
                    Y = M(:,obj.trainLabels)';
                    Y = sign(Y-0.5);                     
                    
                    [opttheta, cost] = obj.minfunc( @(p) obj.stackedAECostSVM(p, ...
                                                       obj.SA(obj.depth).hiddenSize, obj.nClasses, ...
                                                       obj.netconfig, obj.trainData, Y, obj.SVM.C), ...
                                                  obj.stackedAETheta);
                    
                else
                    [opttheta, cost] = obj.minfunc( @(p) obj.stackedAECostSM(p, ...
                                                       obj.SA(obj.depth).hiddenSize, obj.nClasses, obj.netconfig,...
                                                       obj.lambda, obj.trainData, obj.trainLabels), ...                                   
                                                  obj.stackedAETheta);
                end
                obj.stackedAEOptTheta = opttheta;
            end
        end
        
        
        %==================================================================
        %   Predict samples using feed-forward and softmax 
        %==================================================================
        function pred = predict_samples(obj, testData, theta)
            pred = [];
            if isModelValid_pred(obj) && ~isempty(testData)
                if nargin>2
                    pred = obj.stackedAEPredict(theta, obj.SA(obj.depth).hiddenSize, obj.nClasses, obj.netconfig, testData);
                elseif ~isempty(obj.stackedAETheta)
                	pred = obj.stackedAEPredict(obj.stackedAETheta, obj.SA(obj.depth).hiddenSize, obj.nClasses, obj.netconfig, testData);
                elseif ~isempty(obj.stackedAEOptTheta)
                	pred = obj.stackedAEPredict(obj.stackedAEOptTheta, obj.SA(obj.depth).hiddenSize, obj.nClasses, obj.netconfig, testData);                    
                end
            end
        end
        
        
    end    
    
    
    methods(Hidden)
                
        
        %==================================================================
        %   Feed forward step for the sparse autoencoders using learned
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
        % Parameters are need to be converted to a column vector for
        % optimizer and configuration of the network must be saved for
        % converting back stacked structure in advance
        %==================================================================
        function [stackedAETheta, netconfig] = init_stack(obj)
           
            % Initialize the stack using the parameters learned
            stack = cell(obj.depth,1);
            
            for i=1:obj.depth           
                sa = obj.SA(i);               
                stack{i}.w = reshape(sa.theta(1:sa.hiddenSize * sa.visibleSize), sa.hiddenSize, sa.visibleSize);                       
                stack{i}.b = sa.theta(2*sa.hiddenSize*sa.visibleSize+1:2*sa.hiddenSize*sa.visibleSize+sa.hiddenSize);                                 
            end            
            
            % Initialize the parameters for the deep model
            [stackparams, netconfig] = obj.stack2params(stack);
            if obj.svmFlag
                the = obj.SVM.theta';
                stackedAETheta = [ the(:) ; stackparams ];                
            else
                stackedAETheta = [ obj.SM.theta(:) ; stackparams ];
            end
            
        end
        
        
        %==================================================================
        % checks model parameters for softmax regressor
        %==================================================================
        function isValid = isModelValid(obj)            
            isValid = false;
                        
            if isempty(obj.trainData),   return; end
            if isempty(obj.trainLabels), return; end
            if isempty(obj.depth),       return; end
            if isempty(obj.SA),          return; end
            if isempty(obj.svmFlag),     return; end
            
            if obj.svmFlag            
                if isempty(obj.SVM), return; end
            else
                if isempty(obj.SM), return; end
            end
            
            isValid = true;
        end
        
        
        %==================================================================
        % checks model parameters for fine-tuning
        %==================================================================
        function isValid = isModelValid_fineTune(obj)            
            isValid = false;
                        
            if isempty(obj.trainData),      return; end
            if isempty(obj.trainLabels),    return; end
            if isempty(obj.lambda),         return; end
            if isempty(obj.depth),          return; end
            if isempty(obj.SA),             return; end                      
            if isempty(obj.minfunc),        return; end
            if isempty(obj.nClasses),       return; end
            if isempty(obj.stackedAETheta), return; end
            if isempty(obj.netconfig),      return; end                        
            
            if obj.svmFlag
                if isempty(obj.SVM), return; end
            else
                if isempty(obj.SM), return; end
            end
            
            for i=1:obj.depth
                if isempty(obj.SA(i).hiddenSize), return; end
            end
                                                                                                                                         
            isValid = true;
        end
        
        
        %==================================================================
        % checks model parameters for prediction
        %==================================================================
        function isValid = isModelValid_pred(obj)            
            isValid = false;
                        
            if isempty(obj.depth),          return; end
            if isempty(obj.SA),             return; end
            if isempty(obj.nClasses),       return; end
            if isempty(obj.netconfig),      return; end         
            if isempty(obj.SA(obj.depth).hiddenSize), return; end
                                                                                                                                         
            isValid = true;
        end
                         
    end            
    
end

