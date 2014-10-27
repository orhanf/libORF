classdef LogisticRegressor < handle
%==========================================================================
%   Logistic regression class for two class classification. Following
%   utilities are provided:
%
%   Iterative methods for parameter estimation:
%       - Gradient descent
%       - Gradient descent with L2 regularization
%       - Stochastic gradient descent       
%       - Stochastic gradient descent with L2 regularization
%       - Gradient descent with provided minimization function 
%
%   Numerical methods for parameter estimation:
%       - Newton's method
%       - Newton's method with L2 regularization
%
% orhanf
%==========================================================================


    properties

        % model parameters
        theta_gd;     % parameter vector (vector) for gradient descent
        theta_newton; % parameter vector (vector) for normal equation
        alpha;        % step size for gradient descent (scaler)        
        nParams;      % length of parameter vector
        nIter;        % number of iterations for gradient descent   
        epsilon       % tolerance for consecutive iterations    
        lambda;       % regularization parameter   
        shuffle;      % shuffle flag training data for SGD  
        trackJ;       % use exponentially weighted averaging to keep track 
                      %     of cost function for SGD  
        
        % inputs
        x;           % input training data
        y;           % input training labels
        x_unscaled;  % original input training data (unnormalized)

        % meta parameters
        silent;      % display cost in each iteration etc.
        
    end         

    
    properties(Hidden)
        
        % flags         
        addBias;    % column of ones for x 
        scaleX;     % scale both types of inputs by their standard  
                    %       deviations and set their means to zero  
                    
        m;          % number of samples
        J;          % cost function values (vector)  
        J_hat;      % cost function values for exp.weighted averages        
    end
    
    
    methods        
        
        %==================================================================
        % constructor
        %==================================================================
        function obj = LogisticRegressor(options)
                        
            obj.alpha   = 0.07;     % libORF sets step-size for gradient descent by default
            obj.nIter   = 1500;     % libORF sets iteration number for GD and Newton by default 
            obj.epsilon = 1e-5;     % for future use
            obj.lambda  = 0;        % libORF does not use regularization by default
            obj.addBias = true;     % libORF adds bias to features by default
            obj.scaleX  = false;    % libORF does not normalize by default                            
            obj.silent  = false;    % libORF displays cost in each iteration
            obj.shuffle = false;    % libORF does not shuffle training data for SGD
            obj.trackJ  = false;    % libORF does not track J for SGD
            
            if nargin>0 && isstruct(options)                                                   
                if isfield(options,'alpha'),   obj.alpha   = options.alpha;   end
                if isfield(options,'J'),       obj.J       = options.J;       end
                if isfield(options,'nParams'), obj.nParams = options.nParams; end                
                if isfield(options,'nIter'),   obj.nIter   = options.nIter;   end 
                if isfield(options,'epsilon'), obj.epsilon = options.epsilon; end
                if isfield(options,'x'),       obj.x       = options.x;       end
                if isfield(options,'y'),       obj.y       = options.y;       end
                if isfield(options,'addBias'), obj.addBias = options.addBias; end
                if isfield(options,'scaleX'),  obj.scaleX  = options.scaleX;  end
                if isfield(options,'lambda'),  obj.lambda  = options.lambda;  end
                if isfield(options,'silent'),  obj.silent  = options.silent;  end
                if isfield(options,'shuffle'), obj.shuffle = options.shuffle; end
                if isfield(options,'trackJ'),  obj.trackJ  = options.trackJ; end
                if isfield(options,'theta_gd'),     obj.theta_gd     = options.theta_gd;     end
                if isfield(options,'theta_newton'), obj.theta_newton = options.theta_newton; end
                
                obj = set_model_parameters(obj);                                
            end
        end
        
        
        %==================================================================
        % apply gradient descent to estimate parameters 
        %==================================================================
        function [J theta_gd] = gradient_descent(obj)
            
            J        = [];
            theta_gd = [];
            
            if isModelValid_gd(obj)
                
                X = obj.x;                
                Y = obj.y;
                W = obj.theta_gd;       % refering (W)eight
                C = zeros(obj.nIter,1); % refering (C)ost                          
                
                for i=1:obj.nIter

                    % Calculate cost and gradient
                    [C(i), gradJ] = obj.costFunction(W, X, Y);

                    % Update parameters
                    W = W - (obj.alpha .* gradJ);
                    
                    if ~obj.silent
                        disp(['Iter:[' num2str(i) ']  J:[' num2str(C(i)) ']  Norm(theta):[' num2str(norm(W)) ']']);
                    end                    
                end
                obj.theta_gd = W;
                obj.J        = C;
                J            = obj.J;
                theta_gd     = obj.theta_gd;            
            end                        
        end


        %==================================================================
        % apply gradient descent to estimate parameters with L2
        % regularization
        %==================================================================
        function [J theta_gd] = gradient_descent_L2(obj)
            
            J        = [];
            theta_gd = [];
            
            if isModelValid_gd_L2(obj)
                
                X = obj.x;                
                Y = obj.y;
                W = obj.theta_gd;       % refering (W)eight
                C = zeros(obj.nIter,1); % refering (C)ost                          
                
                for i=1:obj.nIter

                    % Calculate cost and gradient
                    [C(i), gradJ] = obj.costFunctionLogRegL2(W, X, Y, obj.lambda);

                    % Update parameters
                    W = W - (obj.alpha .* gradJ);
                    
                    if ~obj.silent
                        disp(['Iter:[' num2str(i) ']  J:[' num2str(C(i)) ']  Norm(theta):[' num2str(norm(W)) ']']);
                    end
                end
                obj.theta_gd = W;
                obj.J        = C;
                J            = obj.J;
                theta_gd     = obj.theta_gd;            
            end                        
        end        

        
        %==================================================================
        % apply stochastic gradient descent to estimate parameters 
        %==================================================================
        function [J theta_gd] = stochastic_gradient_descent(obj)
            
            J        = [];            
            theta_gd = [];
            
            if isModelValid_gd(obj)
                
                X = obj.x;                
                Y = obj.y;
                M = obj.m;
                W = obj.theta_gd;               % refering (W)eight
                C = zeros(obj.nIter*M,1);       % refering (C)ost                          
                C_hat = zeros(obj.nIter*M+1,1); % refering (C)ost hat                
                
                % Shuffle training data
                if obj.shuffle
                    myperm = randperm(M);
                    X = X(myperm, :);
                    Y = Y(myperm);
                end
                                
                for i=1:obj.nIter    % Outer loop for SGD iterations
                    for j=1:M        % Inner loop for each sample   
                    
                        idx = (i-1) * M + j;
                        
                        % Calculate cost and gradient
                        [C(idx), gradJ] = obj.costFunction(W, X(j,:), Y(j));

                        % Track "running estimate" for J(theta)
                        if obj.trackJ
                            C_hat(idx+1) = (0.999 .* C_hat(idx)) + (0.001 .* C(i));
                        end                            
                        
                        % Update parameters
                        W = W - (obj.alpha .* gradJ);

                        if ~obj.silent
                            disp(['IterSGD:[' num2str(j) '/' num2str(i) ']  J:[' num2str(C(i)) ']  Norm(theta):[' num2str(norm(W)) ']']);
                        end                    
                    end
                end
                obj.theta_gd = W;
                obj.J        = C;
                obj.J_hat    = C_hat;
                J            = obj.J;
                theta_gd     = obj.theta_gd;            
            end                        
        end  
             
        
        %==================================================================
        % apply stochastic gradient descent to estimate parameters with L2
        % regularization
        %==================================================================
        function [J theta_gd] = stochastic_gradient_descent_L2(obj)
            
            J        = [];            
            theta_gd = [];
            
            if isModelValid_gd_L2(obj)
                
                X = obj.x;                
                Y = obj.y;
                M = obj.m;
                W = obj.theta_gd;               % refering (W)eight
                C = zeros(obj.nIter*M,1);       % refering (C)ost                          
                C_hat = zeros(obj.nIter*M+1,1); % refering (C)ost hat                
                
                % Shuffle training data
                if obj.shuffle
                    myperm = randperm(M);
                    X = X(myperm, :);
                    Y = Y(myperm);
                end
                                
                for i=1:obj.nIter    % Outer loop for SGD iterations
                    for j=1:M        % Inner loop for each sample   
                    
                        idx = (i-1) * M + j;
                        
                        % Calculate cost and gradient
                        [C(idx), gradJ] = obj.costFunctionLogRegL2(W, X(j,:), Y(j), obj.lambda);

                        % Track "running estimate" for J(theta)
                        if obj.trackJ
                            C_hat(idx+1) = (0.999 .* C_hat(idx)) + (0.001 .* C(i));
                        end                            
                        
                        % Update parameters
                        W = W - (obj.alpha .* gradJ);

                        if ~obj.silent
                            disp(['IterSGD:[' num2str(j) '/' num2str(i) ']  J:[' num2str(C(i)) ']  Norm(theta):[' num2str(norm(W)) ']']);
                        end                    
                    end
                end
                obj.theta_gd = W;
                obj.J        = C;
                obj.J_hat    = C_hat;
                J            = obj.J;
                theta_gd     = obj.theta_gd;            
            end                        
        end            
        
        
        %==================================================================
        % apply gradient descent using provided minimization function, this
        % method uses L2 regularization by default
        %==================================================================
        function [J theta_gd] = gradient_descent_minfunc(obj,minfunc)
            
            J        = [];
            theta_gd = [];
            
            if ~isempty(obj.x) && ~isempty(obj.y) && ~isempty(obj.lambda)
                
                [opttheta, cost] = minfunc( @(p) obj.costFunctionLogRegL2(p, obj.x, obj.y, obj.lambda),...
                                obj.theta); % options must be passed to function handle as the 3rd parameter
                
                obj.theta_gd = opttheta;
                obj.J        = cost;                
                J            = obj.J;
                theta_gd     = obj.theta_gd;            
            end                        
        end
        
        
        %==================================================================
        % apply newton's method to estimate parameters 
        %==================================================================
        function [J theta_newton] = newtons_method(obj)
            
            J            = [];
            theta_newton = [];
            
            if isModelValid_newton(obj)
                
                M = obj.m;
                X = obj.x;                
                Y = obj.y;
                W = obj.theta_newton;   % refering (W)eight
                C = zeros(obj.nIter,1); % refering (C)ost                          
                
                g = inline('1.0 ./ (1.0 + exp(-z))'); % logistic function
                
                for i=1:obj.nIter

                    % Calculate cost and gradient
                    [C(i), gradJ] = obj.costFunction(W, X, Y);       
                    
                    % Hessian                                       
                    h = g(X * W);
                    H = (1/M).* X' * diag(h) * diag(1-h) * X;

                    % Update parameters
                    W = W - (H\gradJ);
                    
                    if ~obj.silent
                        disp(['Iter:[' num2str(i) ']  J:[' num2str(C(i)) ']  Norm(theta):[' num2str(norm(W)) ']']);
                    end
                end
                
                obj.theta_newton = W;
                obj.J            = C;
                J                = obj.J;
                theta_newton     = obj.theta_newton;            
            end                        
        end                
        
 
        %==================================================================
        % apply newton's method to estimate parameters with L2
        % regularization
        %==================================================================
        function [J theta_newton] = newtons_method_L2(obj)
            
            J            = [];
            theta_newton = [];
            
            if isModelValid_newton_L2(obj)
                
                M = obj.m;
                X = obj.x;                
                Y = obj.y;
                W = obj.theta_newton;   % refering (W)eight
                C = zeros(obj.nIter,1); % refering (C)ost                          
                
                g = inline('1.0 ./ (1.0 + exp(-z))'); % logistic function
                
                for i=1:obj.nIter

                    % Calculate cost and gradient
                    [C(i), gradJ] = obj.costFunctionLogRegL2(W, X, Y, obj.lambda);                           
                    
                    % Hessian
                    L = obj.lambda / M * eye(obj.nParams);
                    L(1) = 0;
                    h = g(X * W);
                    H = (1/M).* X' * diag(h) * diag(1-h) * X + L;

                    % Update parameters
                    W = W - (H\gradJ);
                    
                    if ~obj.silent
                        disp(['Iter:[' num2str(i) ']  J:[' num2str(C(i)) ']  Norm(theta):[' num2str(norm(W)) ']']);
                    end
                end
                
                obj.theta_newton = W;
                obj.J            = C;
                J                = obj.J;
                theta_newton     = obj.theta_newton;            
            end                        
        end          
        
        
        %==================================================================
        % predict new samples
        %==================================================================
        function labels = predict_samples(obj, samples, theta)
            if ~isempty(theta)                
                g = inline('1.0 ./ (1.0 + exp(-z))'); % logistic function
                labels = g(theta' * samples);
            end
        end                
        
    end

    
    methods(Hidden)
       
        %==================================================================
        % checks model parameters for logistic regression using gradient
        % descent
        %==================================================================
        function isValid = isModelValid_gd(obj)            
            isValid = false;
            
            if isempty(obj.x),       return; end
            if isempty(obj.y),       return; end
            if isempty(obj.nIter),   return; end
            if isempty(obj.alpha),   return; end
            
            isValid = true;
        end

        
        %==================================================================
        % checks model parameters for logistic regression using gradient
        % descent with L2 regularization
        %==================================================================
        function isValid = isModelValid_gd_L2(obj)            
            isValid = false;
            
            if isempty(obj.x),       return; end
            if isempty(obj.y),       return; end
            if isempty(obj.nIter),   return; end
            if isempty(obj.alpha),   return; end
            if isempty(obj.lambda),  return; end
            
            isValid = true;
        end
        
        
        %==================================================================
        % checks model parameters for logistic regression using newtons
        % method
        %==================================================================
        function isValid = isModelValid_newton(obj)            
            isValid = false;
            
            if isempty(obj.x),       return; end
            if isempty(obj.y),       return; end
            if isempty(obj.nIter),   return; end
            if isempty(obj.lambda),  return; end
            
            isValid = true;
        end                       
        
        
        %==================================================================
        % checks model parameters for logistic regression using newtons
        % method with L2 regularization
        %==================================================================
        function isValid = isModelValid_newton_L2(obj)            
            isValid = false;
            
            if isempty(obj.x),       return; end
            if isempty(obj.y),       return; end
            if isempty(obj.nIter),   return; end
            if isempty(obj.lambda),  return; end
            
            isValid = true;
        end   
        
        
        %==================================================================
        % setup parameters 
        %==================================================================
        function obj = set_model_parameters(obj)

            obj.x_unscaled = obj.x;
            if obj.scaleX                 
                obj.x = PreProcessing.normalize_zero_mean_unit_var(obj.x);
            end            
            
            if obj.addBias
               obj.x            = [ones(length(obj.y), 1), obj.x]; % Add a column of ones to x
               obj.x_unscaled   = [ones(length(obj.y), 1), obj.x_unscaled]; % Add a column of ones to x
            end
            
            if isempty(obj.nParams),      obj.nParams      = size(obj.x,2);        end
            if isempty(obj.theta_gd),     obj.theta_gd     = zeros(obj.nParams,1); end
            if isempty(obj.theta_newton), obj.theta_newton = zeros(obj.nParams,1); end            
            if isempty(obj.m),            obj.m            = size(obj.x,1);        end

        end
                          
    end    
    
end

