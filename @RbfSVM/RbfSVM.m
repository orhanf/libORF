classdef RbfSVM < handle
%==========================================================================
% Class for Kernel SVM with radial basis function 
%
%   This class is written only for self-educational purposes, strongly not
%   recommended for serious use. Most of the functions are adapted from
%   pmtk-toolbox and improved slightly.
%
%   Also, support vectors are not saved for prediction, rather entire
%   dataset is saved.
%
% orhanf - (c) 2014 Orhan FIRAT <orhan.firat@ceng.metu.edu.tr>
%==========================================================================
    
    properties
        % model parameters
        theta;         % parameter vector (vector) for gradient descent
        C;             % regularization parameter in terms of lambda (similar to 1/C parameter of libSVM)
        g;             % scaling parameter, gamma of kernel
        lambda;        % weight decay parameters, L2 regularization
        x;             % input training data < n x m > n: numFeatures
        y;             % input labels < m x 1 > m : numSamples
        silent;        % display cost in each iteration etc.
        minfunc;       % minimization function handle for optimization
        addBias;       % add a bias feature of ones
    end
    
    methods
        
        %==================================================================
        % constructor
        %==================================================================
        function obj = RbfSVM(options)
            
            obj.C        = 100;
            obj.g        = 1/256;
            obj.lambda   = 1e-3;
            obj.silent   = false;
            obj.addBias  = true;
            
            if nargin>0 && isstruct(options)
                if isfield(options,'theta'),  obj.theta    = options.theta;   end
                if isfield(options,'C'),      obj.C        = options.C;       end
                if isfield(options,'g'),      obj.g        = options.g;       end
                if isfield(options,'lambda'), obj.lambda   = options.lambda;  end
                if isfield(options,'x'),      obj.x        = options.x;       end
                if isfield(options,'y'),      obj.y        = options.y;       end
                if isfield(options,'silent'), obj.silent   = options.silent;  end
                if isfield(options,'minfunc'),obj.minfunc  = options.minfunc; end
                if isfield(options,'addBias'),obj.addBias  = options.addBias; end
            end
        end
        
        %==================================================================
        % Train model using given minimization function
        %==================================================================
        function [opttheta] = train_model(obj)
            
            opttheta = 0;
            
            if isModelValid(obj)                
                try
                    [opttheta] = train_model_l2rbf(obj);
                catch err
                    disp(['Linear SVM Minimization function terminated with error:' err.getReport]);
                end
            end
        end
        
        
        %==================================================================
        %   Predict new samples with trained model
        %==================================================================
        function pred = predict_samples(obj, teData)
        
            if ~isempty(obj.theta) && ~isempty(teData)

                % add bias to test data 
                trData = obj.x;
                if obj.addBias
                    teData= [teData; ones(1,size(teData,2))];
                    trData= [trData; ones(1,size(trData,2))];
                end
                
                K= obj.rbfKernel(teData, trData, obj.g);
                [~, pred]= max(K*obj.theta,[],2);
                               
            end
        end
       
    end
    
    
    methods(Hidden)
                
        %==================================================================
        % 
        %==================================================================        
        function [opttheta] = train_model_l2rbf(obj)
            
            if ~obj.silent, fprintf('training all classes at once...\n'); end

            % add bias to features
            if obj.addBias
                X = [obj.x; ones(1,size(obj.x,2))];
            else
                X = obj.x;
            end
            
            % get helpers
            nClasses  = numel(unique(obj.y));            
            nFeatures = size(X,2);
            
            % arrange label vector to a matrix and labels to [-1,1]
            M = eye(nClasses);
            Y = M(:,obj.y)';
            Y = sign(Y-0.5);                       
            
            % initialize parameters
            opttheta = randn(nFeatures*nClasses,1); % +1 features for bias
%             opttheta = zeros(nFeatures*nClasses,1); % +1 features for bias
                           
            % conduct optimization
            Krbf = obj.rbfKernel(X, X, obj.g);
            funObj = @(theta)obj.rbfSVMcost(theta, Krbf, Y, nClasses);
            opttheta= obj.minfunc(@obj.penalizedKernelL2_matrix, opttheta, Krbf, nClasses, funObj, obj.C);
            
            opttheta  = reshape(opttheta, [nFeatures, nClasses]);
            
            obj.theta = opttheta;       
            
        end        
                

        %==================================================================
        % checks model parameters for linear SVM
        %==================================================================
        function isValid = isModelValid(obj)
            isValid = false;
            
            if isempty(obj.x),       return; end
            if isempty(obj.y),       return; end            
            if isempty(obj.minfunc), return; end
            
            isValid = true;
        end
                              
    end
    
end

