classdef LinearRegressor < handle
%
%
% orhanf

    properties

        % model parameters
        theta;        % parameter vector (vector)
        theta_normal; % parameter vector (vector) for normal equation
        alpha;        % step size for gradient descent (scaler)        
        nParams;      % length of parameter vector
        nIter;        % number of iterations for gradient descent   
        epsilon       % tolerance for consecutive iterations 
        h;            % hypothesis function        
        lambda;       % regularization parameter  
        
        % inputs
        x;           % input training data
        y;           % input training labels
        x_unscaled;  % original input training data (unnormalized)

    end         

    properties(Hidden)
        
        % flags 
        addBias;    % column of ones for x 
        scaleX;     % scale both types of inputs by their standard  
                    % deviations and set their means to zero  
                    
        m;          % number of samples
        J;          % cost function values (vector)              
    end
    
    
    methods        
        
        % constructor
        function obj = LinearRegressor(options)
                        
            obj.alpha   = 0.07;
            obj.nIter   = 1500;
            obj.epsilon = 1e-5;
            obj.addBias = true;     % libORF adds bias to features by default
            obj.scaleX  = false;    % libORF does not normalize by default                
            obj.h       = @linear_regression_hyp; 
            
            if nargin>0 && isstruct(options)                                   
                if isfield(options,'theta'),   obj.theta   = options.theta;   end
                if isfield(options,'alpha'),   obj.alpha   = options.alpha;   end
                if isfield(options,'J'),       obj.J       = options.J;       end
                if isfield(options,'nParams'), obj.nParams = options.nParams; end                
                if isfield(options,'nIter'),   obj.nIter   = options.nIter;   end 
                if isfield(options,'epsilon'), obj.epsilon = options.epsilon; end
                if isfield(options,'x'),       obj.x       = options.x;       end
                if isfield(options,'y'),       obj.y       = options.y;       end
                if isfield(options,'addBias'), obj.addBias = options.addBias; end
                if isfield(options,'h'),       obj.h       = options.h;       end
                if isfield(options,'scaleX'),  obj.scaleX  = options.scaleX;  end
                if isfield(options,'lambda'),  obj.lambda  = options.lambda;  end
                if isfield(options,'theta_normal'),   obj.theta_normal   = options.theta_normal;   end
                
                obj = set_model_parameters(obj);                                
            end
        end
        
        
        % apply gradient descent to estimate parameters 
        function [J theta] = gradient_descent(obj)
            
            J     = [];
            theta = [];
            
            if isModelValid(obj)
                
                M = obj.m;
                X = obj.x;                
                Y = obj.y;
                W = obj.theta;  % refering (W)eight
                C = obj.J;      % refering (C)ost
                
                for i=1:obj.nIter
                    
                    % Calculate cost functions J
                    C(i,1) = (0.5/M) .* (X * W - Y)' * ( X * W - Y);        

                    % Calculate gradient
                    g = (1/M) .* X' * ((X * W) - Y);

                    % Update parameters
                    W = W - obj.alpha .* g;                                                             
                    
                end
                obj.theta = W;
                obj.J     = C;
                J     = obj.J;
                theta = obj.theta;            
            end
            
            
        end


        % apply L2 regularized gradient descent to estimate parameters 
        function [J theta] = gradient_descent_L2(obj)
            
            J     = [];
            theta = [];
            
            if isModelValidL2(obj)
                
                X = obj.x;                
                Y = obj.y;
                W = obj.theta;  % refering (W)eight
                C = obj.J;      % refering (C)ost                
                
                for i=1:obj.nIter
                    
                    [C(i), g] = costFunctionLinRegL2(obj, W, X, Y, obj.lambda);                                        
                    
                    % Update parameters
                    W = W - obj.alpha .* g;                                                             
                    
                end
                obj.theta = W;
                obj.J     = C;
                J     = obj.J;
                theta = obj.theta;            
            end                        
        end
        
        
        % apply normal equation to estimate parameters
        function theta = normal_equation(obj)
            theta = [];
            if isModelValid(obj)                                  
                X = obj.x_unscaled;
                Y = obj.y;                
                obj.theta_normal  = pinv(X'*X)*X'*Y;
                theta = obj.theta_normal;
            end            
        end

        
        % apply normal equation to estimate parameters
        function theta = normal_equation_L2(obj)
            theta = [];
            if isModelValidL2(obj)                                  
                X = obj.x_unscaled;
                Y = obj.y;                   
                L = obj.lambda .* eye(obj.nParams);
                L(1) = 0;                
                obj.theta_normal = (X' * X + L )\ X' * Y;
                theta = obj.theta_normal;
            end            
        end

        
        % predict new samples
        function labels = predict_samples(obj, samples)
            if ~isempty(obj.theta)
                labels = samples * obj.theta;
            end
        end
        
        
        % predict new samples
        function labels = predict_samples_normal(obj, samples)
            if ~isempty(obj.theta_normal)
                labels = samples * obj.theta_normal;
            end
        end
               
    end

    
    methods(Hidden)
       
        % checks model parameters for linear regression
        function isValid = isModelValid(obj)            
            isValid = false;
            
            if isempty(obj.x),       return; end
            if isempty(obj.y),       return; end
            if isempty(obj.nIter),   return; end
            if isempty(obj.alpha),   return; end
            
            isValid = true;
        end

        
        % checks model parameters for L2 regularized linear regression
        function isValid = isModelValidL2(obj)
            isValid = false;
            
            if isempty(obj.x),       return; end
            if isempty(obj.y),       return; end
            if isempty(obj.nIter),   return; end
            if isempty(obj.alpha),   return; end
            if isempty(obj.lambda),  return; end
            
            isValid = true;
            
        end
        
        
        % setup parameters 
        function obj = set_model_parameters(obj)

            obj.x_unscaled = obj.x;
            if obj.scaleX                 
                obj.x = PreProcessing.normalize_zero_mean_unit_var(obj.x);
            end            
            
            if obj.addBias
               obj.x            = [ones(length(obj.y), 1), obj.x]; % Add a column of ones to x
               obj.x_unscaled   = [ones(length(obj.y), 1), obj.x_unscaled]; % Add a column of ones to x
            end

            if isempty(obj.nParams), obj.nParams = size(obj.x,2);        end
            if isempty(obj.theta),   obj.theta   = zeros(obj.nParams,1); end
            if isempty(obj.J),       obj.J       = zeros(obj.nIter,1);   end
            if isempty(obj.m),       obj.m       = size(obj.x,1);        end

        end
        
        
        % hypothesis function for linear regression
        function hyp = linear_regression_hyp(theta,x)
            hyp = theta' * x;
        end                  
        
    end
    
    
end

