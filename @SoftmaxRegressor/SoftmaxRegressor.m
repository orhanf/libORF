classdef SoftmaxRegressor < handle
%==========================================================================
% Class for Softmax regression
%
%   No iterative method supported for training only external optimizer
%   based optimization supported. The only cost function is L2 regularized 
%   in order to make cost function J strictly convex and Hessian 
%   invertible. Gradient descent, or approximate second order methods 
%   L-BFGS, CG guaranteed to converge onto the global optimum.
%   
%   Over-parametrization is not handled for vectorized implementation and
%   parameters for each class is kept in feature matrix.
%
% orhanf
%==========================================================================

    properties
        % model parameters
        theta;         % parameter matrix for gradient descent                
        lambda;        % regularization (weight decay) parameter 
        alpha;         % learning rate for gradient descent
        momentum;      % momentum of gradient descent
        nIters;        % number of iterations for gradient descent
        
        x;             % input training data < n x m > n: numFeatures
        y;             % input labels < m x 1 > m : numSamples
        silent;        % display cost in each iteration etc.     
        minfunc;       % minimization function handle for optimization
        J;             % cost function value
        initFlag;      % random initialization by default for theta
        useGPU;        % whether to use gpu in gradient descent scheme
    end

        
    methods
        
        %==================================================================
        % constructor
        %==================================================================
        function obj = SoftmaxRegressor(options)
                                    
            obj.lambda   = 1e-4;   
            obj.silent   = false;
            obj.initFlag = true;
            obj.nIters   = 100;
            obj.alpha    = 0.01;
            obj.momentum = 0.9;
            obj.useGPU   = false;
            
            if nargin>0 && isstruct(options)                                                                                                   
                if isfield(options,'useGPU'),       obj.useGPU   = options.useGPU;              end
                if isfield(options,'silent'),       obj.silent   = options.silent;              end
                if isfield(options,'lambda'),       obj.lambda   = options.lambda;              end
                if isfield(options,'nIters'),       obj.nIters   = options.nIters;              end
                if isfield(options,'alpha'),        obj.alpha    = options.alpha;               end
                if isfield(options,'momentum'),     obj.momentum = options.momentum;            end
                if isfield(options,'x'),            obj.x        = options.x;                   end
                if isfield(options,'y'),            obj.y        = options.y;                   end                
                if isfield(options,'initFlag'),     obj.initFlag = options.initFlag;            end
                if isfield(options,'minfunc'),      obj.minfunc  = options.minfunc;             end
                if isfield(options,'theta'),        obj.theta    = options.theta;               end
            end
        end
        
        
        %==================================================================
        % Train model using given minimization function
        %==================================================================
        function [opttheta, cost] = train_model(obj)
            
            opttheta = [];
            cost     = [];
            
            if isModelValid(obj)                
                
                nClasses  = max(obj.y);
                nFeatures = size(obj.x, 1);
                
                if obj.initFlag                
                    obj.theta = 0.005 * randn(nClasses * nFeatures, 1);   % initialize around zero                 
                end
                
                try
                    
                    % convert labels to one of k encoding
                    obj.y = Utility.convert_to_one_of_k_encoding(obj.y);
                    
                    if isempty(obj.minfunc) % apply gradient descent
                        
                        if obj.useGPU
                            cost       = gpuArray(zeros(obj.nIters,1));
                            opttheta   = gpuArray(obj.theta); 
                            trData     = gpuArray(obj.x);
                            trLabels   = gpuArray(obj.y);
                            curr_speed = gpuArray(zeros(size(opttheta)));
                        else
                            cost       = zeros(obj.nIters,1);
                            opttheta   = obj.theta;
                            trData     = obj.x;
                            trLabels   = obj.y;
                            curr_speed = zeros(size(opttheta));
                        end
                        
                        for i=1:obj.nIters
                            
                            % calculate cost and gradient
                            [cost(i),grad] = obj.softmaxCost(opttheta, nClasses, nFeatures, obj.lambda, trData, trLabels);
                            
                            % Update parameters
                            curr_speed = curr_speed * obj.momentum - grad;
                            opttheta = opttheta + curr_speed * obj.alpha;                            
                            
                            if ~obj.silent, fprintf('Iter:[%d/%d] - Training error:[%f] \n',i, obj.nIters, cost(i)); end
                        end                                                
                        
                        if obj.useGPU 
                            cost = gather(cost);
                            opttheta = gather(opttheta);
                        end
                        
                    else                    % apply external optimizer
                        
                        if obj.useGPU, fprintf('GPU cannot be used for external optimizers, use Gradient Descent instead!\n'); end
                        
                        [opttheta, cost] = obj.minfunc( @(p) obj.softmaxCost(p, nClasses, nFeatures, ...
                                                             obj.lambda, obj.x, obj.y), ...
                                                        obj.theta); % options must be passed to function handle as the 3rd parameter
                                                    
                    end
                                                
                    opttheta  = reshape(opttheta, nClasses, nFeatures);
                    obj.theta = opttheta;
                    obj.J     = cost;
                    
                    if nargout == 0, opttheta = true; end  
                    
                    if ~obj.silent & isempty(obj.minfunc)
                       figure('color','white');
                        hold on;
                        plot(cost, 'b');
                        legend('training');                        
                        ylabel('loss');
                        xlabel('iteration number');
                        hold off; 
                    end
                    
                catch err
                    if nargout == 0, opttheta = false; end  
                    disp(['Softmax Regressor Minimization function terminated with error:' err.getReport]);
                end                
            end        
        end        
        
        
        %==================================================================
        % Check correctness of the gradient function by calculating it
        % numerically - beware this may take several minutes, hours or days
        % according to your model, check your gradient in a restricted
        % model - eg. reduce nFeatures and number of training samples
        %==================================================================
        function [numgrad, grad] = check_numerical_gradient(obj)
            
            numgrad = [];
            grad    = [];
            
            if isModelValid_num(obj)                
                
                nClasses  = max(obj.y);
                nFeatures = size(obj.x, 1);
                
                thetaTMP = 0.005 * randn(nClasses * nFeatures, 1);   % initialize around zero                 
                                               
                [~, grad] = obj.softmaxCost(thetaTMP, nClasses, nFeatures, obj.lambda, obj.x, obj.y);
                
                
                % Check cost function and derivative calculations
                % for the sparse autoencoder.  
                numgrad = obj.computeNumericalGradient( @(p) obj.softmaxCost(p,  nClasses, nFeatures, ...
                                                             obj.lambda, obj.x, obj.y), ...
                                                        thetaTMP);

                diff = norm(numgrad-grad)/norm(numgrad+grad);
                                                              
                % Use this to visually compare the gradients side by side
                % Compare numerically computed gradients with the ones
                % obtained from softmaxCost
                if ~obj.silent
                    disp([numgrad grad]); 
                    disp(diff); % Should be small. 
                end                                
            else
                fprintf(2,'Model is not valid! Please check input options and arguments.\n');
            end        
        end  
          
        
        %==================================================================
        %   Predict new samples with trained model
        %==================================================================
        function pred = predict_samples(obj, data)        
            if ~isempty(obj.theta) && ~isempty(data)
                M = obj.theta * data;    
                [dummy, pred]= max(M);
            end
        end
        
        
        %==================================================================
        %   Predict probabilities of samples with trained model
        %==================================================================
        function probs = predict_probs(obj, data)        
            if ~isempty(obj.theta) && ~isempty(data)
                probs = obj.theta * data;                    
            end
        end
        
        
    end
    
    
    methods(Hidden)       
        
        %==================================================================
        % checks model parameters for softmax regressor
        %==================================================================
        function isValid = isModelValid(obj)            
            isValid = false;
                        
            if isempty(obj.x),           return; end
            if isempty(obj.y),           return; end
            if isempty(obj.lambda),      return; end
%             if isempty(obj.minfunc),     return; end
            
            isValid = true;
        end
           
        
        %==================================================================
        % checks model parameters for numerical checking initialization
        %==================================================================
        function isValid = isModelValid_num(obj)            
            isValid = false;
                        
            if isempty(obj.x),           return; end
            if isempty(obj.y),           return; end
            if isempty(obj.lambda),      return; end
            
            isValid = true;
        end
        
        
    end

end

