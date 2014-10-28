classdef LinearSVM < handle
%==========================================================================
% Class for Linear SVM with L2 regularization
%
%  This class implements linear SVM classifier with square(L2)-hinge loss
%   for multi-class classification using one-vs-all framework.
%
%  The corresponding unconstrained optimization problem is (primal form):
%       min_w (1/2)w'w + C sum_i( max(1-w' x_iy_i,0)^2 )
%
%  We minimize the squared hinge loss for primal, note that square-hinge
%   loss is differentiable hence we can use numerical packages for
%   unconstrained optimization of differentiable real-valued multivariate
%   functions using line-search methods (ex:minFunc with lbfgs).
%
%  For multi-class extension, prediction is done similar to softmax (note
%   that softmax minimizes cross-entropy or maximizes the log-likelihood,
%   while SVM tries to find the maximum margin between different classes)
%   prediction as follows:
%       denoting the output of k-th SVM as, a_k(x) = w'x
%       the predicted class is, argmax_k a_k(x)
%  Hence over-parametrisation is again an issue.
%
%
% orhanf - (c) 2013 Orhan FIRAT <orhan.firat@ceng.metu.edu.tr>
%==========================================================================
    
    properties
        % model parameters
        theta;         % parameter vector (vector) for gradient descent
        C;             % regularization parameter (similar to C parameter of libSVM)
        x;             % input training data < n x m > n: numFeatures
        y;             % input labels < m x 1 > m : numSamples
        silent;        % display cost in each iteration etc.
        minfunc;       % minimization function handle for optimization
        J;             % cost function values (vector)
        method;        % method for cost function
        addBias;       % add a bias feature of ones
        lambdaL1;      % L1 regularization lambda
    end
    
    methods
        
        %==================================================================
        % constructor
        %==================================================================
        function obj = LinearSVM(options)
            
            obj.C        = 100;
            obj.silent   = false;
            obj.method   = 0;
            obj.addBias  = true;
            
            if nargin>0 && isstruct(options)
                if isfield(options,'theta'),   obj.theta    = options.theta;   end
                if isfield(options,'C'),       obj.C        = options.C;       end
                if isfield(options,'x'),       obj.x        = options.x;       end
                if isfield(options,'y'),       obj.y        = options.y;       end
                if isfield(options,'silent'),  obj.silent   = options.silent;  end
                if isfield(options,'minfunc'), obj.minfunc  = options.minfunc; end
                if isfield(options,'J'),       obj.J        = options.J;       end
                if isfield(options,'method'),  obj.method   = options.method;  end
                if isfield(options,'addBias'), obj.addBias  = options.addBias; end
                if isfield(options,'lambdaL1'),obj.lambdaL1 = options.lambdaL1;end
            end
        end
        
        %==================================================================
        % Train model using given minimization function
        %==================================================================
        function [opttheta, cost] = train_model(obj)
            
            opttheta = 0;
            cost     = 0;
            
            if isModelValid(obj)                
                try
                    if obj.method
                        [opttheta, cost] = train_model_l2hinge(obj);
                    else
                        [opttheta, cost] = train_model_l2hinge_all(obj);
                    end
                catch err
                    disp(['Linear SVM Minimization function terminated with error:' err.getReport]);
                end
            end
        end
        
        
        %==================================================================
        %   Predict new samples with trained model
        %==================================================================
        function pred = predict_samples(obj, data)
        
            if ~isempty(obj.theta) && ~isempty(data)

                % add bias to test data 
                if obj.addBias
                    data = [data; ones(1,size(data,2))];
                end
                
                % calculate a_k for all k
                M = data' * obj.theta;    
                
                [dummy, pred]= max(M,[],2);
            end
        end
        
        
        %==================================================================
        % Check correctness of the gradient function by calculating it
        % numerically - beware this may take several minutes, hours or days
        % according to your model, check your gradient in a restricted
        % model - eg. reduce hiddenSize and number of training samples
        %==================================================================
        function [numgrad, grad] = check_numerical_gradient(obj)            
                                         
            % arrange label vector to a matrix and labels to [-1,1]
            nClasses  = numel(unique(obj.y));            
            nFeatures = size(obj.x,1);
            M = eye(numel(unique(obj.y)));
            Y = M(:,obj.y)';
            Y = sign(Y-0.5);            
            
            % initialize parameters
            opttheta = rand(nFeatures,nClasses); 
            
            % get cost and gradient for training data
            [cost, grad] = obj.linearSVMcostL2( opttheta(:), obj.x', Y, obj.C);

            % Check cost function and derivative calculations
            % for the sparse autoencoder.  
            numgrad = obj.computeNumericalGradient( @(x)obj.linearSVMcostL2(x, obj.x', Y, obj.C), opttheta(:));

            diff = norm(numgrad-grad)/norm(numgrad+grad);

            % Use this to visually compare the gradients side by side
            % Compare numerically computed gradients with the ones obtained from backpropagation
            if ~obj.silent
                disp([numgrad grad]); 
                disp(diff); % Should be small. 
            end                                
            
        end
        
       
    end
    
    
    methods(Hidden)
        
        
        %==================================================================
        % Train model using L2 hinge loss - in general multiclass versions
        % of SVM is implemented similar to 1-vs-all softmax fashion, 
        % meaning that k number of binary classifiers are trained then
        % argmax applied to the results. Different from that approach this
        % function optimizes parameter vector at once, meaning that only 1
        % training phase applied.
        %==================================================================        
        function [opttheta, cost] = train_model_l2hinge_all(obj)
            
            if ~obj.silent, fprintf('training all classes at once...\n'); end

            % add bias to features
            if obj.addBias
                X = [obj.x; ones(1,size(obj.x,2))];
            else
                X = obj.x;
            end
            
            % get helpers
            nClasses  = numel(unique(obj.y));            
            nFeatures = size(X,1);
            
            % arrange label vector to a matrix and labels to [-1,1]
            M = eye(nClasses);
            Y = M(:,obj.y)';
            Y = sign(Y-0.5);                       
            
            % initialize parameters
            opttheta = zeros(nFeatures,nClasses); % +1 features for bias
                           
            % conduct optimization
            [opttheta, cost] = obj.minfunc( @(p) obj.linearSVMcostL2(p, X', Y, obj.C), ...
                opttheta(:)); % options must be passed to function handle as the 3rd parameter

            opttheta  = reshape(opttheta, [nFeatures, nClasses]);
            
            obj.theta = opttheta;
            obj.J     = cost;            
            
        end        
        
        
        %==================================================================
        % Train model using L2 hinge loss - own implementation
        %==================================================================        
        function [opttheta, cost] = train_model_l2hinge(obj)
            
            % add bias to features
            if obj.addBias
                X = [obj.x; ones(1,size(obj.x,2))];
            else
                X = obj.x;
            end
            
            % get helpers
            nClasses  = numel(unique(obj.y));            
            nFeatures = size(X,1);
            
            % arrange label vector to a matrix and labels to [-1,1]
            M = eye(nClasses);
            Y = M(:,obj.y)';
            Y = sign(Y-0.5);                       
            
            % initialize parameters
            obj.theta = zeros(nFeatures,nClasses); % +1 features for bias
            
            for i=1:nClasses
                
                if ~obj.silent
                    fprintf('training class [%d of %d] ...\n', i, nClasses);
                end
                
                % initialize parameters
                opttheta = zeros(nFeatures,1); % +1 features for bias
                
                % conduct optimization
                [opttheta, cost] = obj.minfunc( @(p) obj.linearSVMcostL2(p, X', Y(:,i), obj.C), ...
                    opttheta(:)); % options must be passed to function handle as the 3rd parameter
                
                obj.theta(:,i) = opttheta;
                obj.J     = obj.J + cost;
            end
            
            opttheta = obj.theta;
            cost     = obj.J;
            
        end
        
        
        %==================================================================
        % checks model parameters for linear SVM
        %==================================================================
        function isValid = isModelValid(obj)
            isValid = false;
            
            if isempty(obj.x),       return; end
            if isempty(obj.y),       return; end
            if isempty(obj.C),       return; end
            if isempty(obj.minfunc), return; end
            
            isValid = true;
        end
                              
    end
    
end

