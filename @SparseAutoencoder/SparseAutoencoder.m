classdef SparseAutoencoder < handle
%==========================================================================
%
%
%   May25-2014  : SGD, .encodeX added
%   June07-2014 : plot_feature_corrs function added for diagnosis
%   June08-2014 : Calculating Jacobian of hidden layer wrt input added
%
% orhanf - (c) 2012 Orhan FIRAT <orhan.firat@ceng.metu.edu.tr>
%==========================================================================
    
    properties
        % model parameters
        theta;         % parameter vector (vector) for gradient descent
        lambda;        % regularization (weight decay) parameter
        lambdaL1;      % regularization (L1) parameter
        sparsityParam; % desired average activation of the hidden units
        beta;          % weight of sparsity penalty term
        visibleSize;   % number of input units
        hiddenSize;    % number of hidden units
        x;             % input training data
        silent;        % display cost in each iteration etc.
        minfunc;       % minimization function handle for optimization
        J;             % cost function values (vector)
        initFlag;      % random initialization by default for theta
        isLinearCost;  % Use linear cost function

        hActFun;       % hidden-unit activation functions: 0-sigmoid, 1-tanh, 2-relu, 3-linear
        nEpochs;       % number of passes through the dataset
        batchSize;     % mini-batch size
        alpha;         % learning rate
        momentum;      % momentum parameter [0,1]
        useAdaDelta;   % use adaptive delta to estimate learning rate
        adaDeltaRho;   % decay rate rho, for adadelta
        trErr;         % training error vector
        tied;          % use tied weights, W1 == W2'
        
    end
    
    
    methods
        
        %==================================================================
        % constructor
        %==================================================================
        function obj = SparseAutoencoder(options)
            
            obj.lambda   = 0.0001;
            obj.lambdaL1 = 0.0;
            obj.beta     = 3;
            obj.sparsityParam =  0.01;
            obj.silent   = false;
            obj.initFlag = true;
            obj.isLinearCost = false;
            obj.tied = false;
            
            obj.hActFun      = 0;       % use sigmoid by default
            obj.nEpochs      = 100;
            obj.batchSize    = 100;
            obj.alpha        = .01;
            obj.momentum     = 0.9;
            obj.useAdaDelta  = false;
            obj.adaDeltaRho  = 0.95;
            
            
            if nargin>0 && isstruct(options)
                if isfield(options,'silent'),       obj.silent   = options.silent;              end
                if isfield(options,'lambda'),       obj.lambda   = options.lambda;              end
                if isfield(options,'lambdaL1'),     obj.lambdaL1 = options.lambdaL1;            end
                if isfield(options,'beta'),         obj.beta     = options.beta;                end
                if isfield(options,'sparsityParam'),obj.sparsityParam = options.sparsityParam;  end
                if isfield(options,'visibleSize'),  obj.visibleSize   = options.visibleSize;    end
                if isfield(options,'hiddenSize'),   obj.hiddenSize    = options.hiddenSize;     end
                if isfield(options,'x'),            obj.x        = options.x;                   end
                if isfield(options,'initFlag'),     obj.initFlag = options.initFlag;            end
                if isfield(options,'minfunc'),      obj.minfunc  = options.minfunc;             end
                if isfield(options,'theta'),        obj.theta    = options.theta;               end
                if isfield(options,'isLinearCost'), obj.isLinearCost = options.isLinearCost;    end
            
                if isfield(options,'hActFun'),      obj.hActFun      = options.hActFun;         end
                if isfield(options,'nEpochs'),      obj.nEpochs      = options.nEpochs;         end
                if isfield(options,'batchSize'),    obj.batchSize    = options.batchSize;       end
                if isfield(options,'alpha'),        obj.alpha        = options.alpha;           end
                if isfield(options,'momentum'),     obj.momentum     = options.momentum;        end
                if isfield(options,'useAdaDelta'),  obj.useAdaDelta  = options.useAdaDelta;     end
                if isfield(options,'adaDeltaRho'),  obj.adaDeltaRho  = options.adaDeltaRho;     end
                if isfield(options,'trErr'),        obj.trErr        = options.trErr;           end
            end
        end
        
        
        %==================================================================
        % Train model using given minimization function
        %==================================================================
        function [opttheta, cost] = train_model(obj)
            
            opttheta = 0;
            cost     = 0;
            
            if isModelValid(obj)
                
                if obj.initFlag
                    obj.theta = obj.initializeParameters(obj.hiddenSize, obj.visibleSize);
                end
                
                try
                    
                    if isempty(obj.minfunc)
                        [opttheta, cost] = trainSparseAutoencoderSGD(obj);
                    else
                        [opttheta, cost] = trainSparseAutoencoderBGD(obj);
                    end
                    
                    obj.theta = opttheta;
                    obj.J     = cost;
                catch err
                    fprintf(2,'Sparse Autoencoder Minimization function terminated with error:\n%s\n', err.getReport);
                end
            else
                fprintf(2,'Sparse Auto-encoder Model is not valid! Please check input options.\n');
            end
        end
        
        
        %==================================================================
        % Check correctness of the gradient function by calculating it
        % numerically - beware this may take several minutes, hours or days
        % according to your model, check your gradient in a restricted
        % model - eg. reduce hiddenSize and number of training samples
        %==================================================================
        function [numgrad, grad] = check_numerical_gradient(obj)
            
            numgrad = [];
            grad    = [];
            
            if isModelValid(obj)
                
                thetaTMP = obj.initializeParameters(obj.hiddenSize, obj.visibleSize);
                
                [~, grad] = obj.sparseAutoencoderCost(thetaTMP, obj.visibleSize, obj.hiddenSize, obj.lambda, ...
                    obj.lambdaL1, obj.sparsityParam, obj.beta, obj.x);
                
                
                % Check cost function and derivative calculations
                % for the sparse autoencoder.
                numgrad = obj.computeNumericalGradient( @(x) obj.sparseAutoencoderCost(x, obj.visibleSize, ...
                    obj.hiddenSize, obj.lambda, obj.lambdaL1, ...
                    obj.sparsityParam, obj.beta, ...
                    obj.x), thetaTMP);
                
                diff = norm(numgrad-grad)/norm(numgrad+grad);
                
                % Use this to visually compare the gradients side by side
                % Compare numerically computed gradients with the ones obtained from backpropagation
                if ~obj.silent
                    disp([numgrad grad]);
                    disp(diff); % Should be small.
                end
            end
        end
        
        
        %==================================================================
        % Returns parameters as a stack of 2 layers, if vectorize flag is
        % set, then parameters in stack are vectorized into column vectors
        %==================================================================
        function params = get_parameters_as_stack(obj,vectorizeFlag)
            
            if nargin<2, vectorizeFlag = false; end
            
            W1StartIdx = 1;
            W2StartIdx = obj.hiddenSize*obj.visibleSize+1;
            b1StartIdx = W2StartIdx + obj.hiddenSize*obj.visibleSize;
            b2StartIdx = b1StartIdx + obj.hiddenSize;
            
            params{1}.w = obj.theta(W1StartIdx:(W2StartIdx-1));
            params{2}.w = obj.theta(W2StartIdx:(b1StartIdx-1));
            
            if ~vectorizeFlag
                params{1}.w = reshape(params{1}.w,[obj.hiddenSize, obj.visibleSize]);
                params{2}.w = reshape(params{2}.w,[obj.visibleSize,obj.hiddenSize]);
            end
            
            params{1}.b = obj.theta(b1StartIdx:(b2StartIdx-1));
            params{2}.b = obj.theta(b2StartIdx:end);
            
        end
        
        
        %==================================================================
        % Encoder function of the autoencoder
        %==================================================================
        function mappedData = encodeX(obj, data)
            try
                
                W1 = reshape(obj.theta(1:obj.hiddenSize*obj.visibleSize), obj.hiddenSize, obj.visibleSize);
                b1 = obj.theta(2*obj.hiddenSize*obj.visibleSize+1:2*obj.hiddenSize*obj.visibleSize+obj.hiddenSize);
                                
                mappedData = nonLinearity(bsxfun(@plus, W1 * data, b1), obj.hActFun); 
                
            catch err
                fprintf(2,'Error in encodeX function!\n%s\n',err.getReport);
                mappedData = [];
            end
        end
        
        
        %==================================================================
        % Plots correlation matrix of learned hidden representations. Ideal
        % representations are decorrelated and correlation matrix can be
        % used to debug the quality of learned features crudely. Frobenius
        % norm of the representation correlation matrix and L1 distance
        % between diagonalized representation correlation matrix and
        % indentity matrix is provided to debug a numerical estimate of the
        % goodness.
        %==================================================================
        function [h, Frobenius_norm, L1_dist_to_identity]  = plot_feature_corrs(obj)            
            
            params = obj.get_parameters_as_stack;
            W1 = params{1}.w;
            corrMat = corr(W1');  % calculate correlation matrix
            corrMat(2,2) = -1;        % this is for color correction
            h = figure; imagesc(corrMat); colorbar;

            % calculate frobenius norm 
            Frobenius_norm = sqrt(sum(abs(corrMat(:)).^2));
            
            % L1 distance of diagonalized corrMat to identity matrix
            L1_dist_to_identity = sum(abs(ones(obj.hiddenSize,1)- svd(corrMat)));
                                    
        end
        
        
        %==================================================================
        % Calculates the Jacobian of the hidden layer with respect to data.  
        %   Jac_j = s'(b + x^T * W_j)*W_j
        %==================================================================
        function Jac = get_jacobian(obj,data)
            
            if nargin < 2
                data = obj.x;
            end
            
            nSamples = size(data,2);
            params   = obj.get_parameters_as_stack;
            W = params{1}.w;
            b = params{1}.b;            
                
            % iterative calculation for each sample-there should be a
            % vectorized way to make it faster
            Jac = zeros(obj.hiddenSize,obj.visibleSize);
            for i=1:nSamples
                h = nonLinearity(bsxfun(@plus, W * data(:,i), b),obj.hActFun);
                Jac = Jac + (repmat(dNonLinearity(h,obj.hActFun), 1, obj.visibleSize) .* W);
            end
            Jac = Jac ./ nSamples;
                
        end
            
        
    end
    
    
    methods(Hidden)                
        
        %==================================================================
        % Train sparse autoencoder with backprop using batch gradient
        % descent with given minimization function. (e.g. minFunc)
        %==================================================================
        function [theta, cost] = trainSparseAutoencoderBGD(obj)
            
            if obj.isLinearCost
                [theta, cost] = obj.minfunc( @(p) obj.sparseAutoencoderLinearCost(p, obj.visibleSize, obj.hiddenSize, ...
                    obj.lambda, obj.lambdaL1, obj.sparsityParam, obj.beta, obj.x), ...
                    obj.theta); % options must be passed to function handle as the 3rd parameter
                
            else
                [theta, cost] = obj.minfunc( @(p) obj.sparseAutoencoderCost(p, obj.visibleSize, obj.hiddenSize, ...
                    obj.lambda, obj.lambdaL1, obj.sparsityParam, obj.beta, obj.x), ...
                    obj.theta); % options must be passed to function handle as the 3rd parameter
            end
            
        end
        
        
        %==================================================================
        % Train sparse autoencoder with backprop using mini-batch
        % stochastic graident descent for given number of epochs.
        %==================================================================
        function [theta, trCosts] = trainSparseAutoencoderSGD(obj)
            
            % get helpers
            nSamples = size(obj.x, 2);
            if obj.batchSize > nSamples
                obj.batchSize = nSamples;
            end
            nBatches = floor(nSamples / obj.batchSize);
            
            params = obj.get_parameters_as_stack;
            
            % optimization
            opt.curr_speed_W1 = params{1,1}.w * 0;
            opt.curr_speed_W2 = params{1,2}.w * 0;
            opt.curr_speed_b1 = params{1,1}.b * 0;
            opt.curr_speed_b2 = params{1,2}.b * 0;
            if obj.useAdaDelta
                opt.expectedGrads_W1 = zeros(obj.hiddenSize, obj.visibleSize); % adadelta : E[g^2]_0
                opt.expectedDelta_W1 = zeros(obj.hiddenSize, obj.visibleSize); % adadelta : E[\delta x^2]_0
                opt.expectedGrads_W2 = zeros(obj.visibleSize, obj.hiddenSize);
                opt.expectedDelta_W2 = zeros(obj.visibleSize, obj.hiddenSize);
                opt.expectedGrads_b1 = zeros(obj.hiddenSize, 1);
                opt.expectedDelta_b1 = zeros(obj.hiddenSize, 1);
                opt.expectedGrads_b2 = zeros(obj.visibleSize, 1);
                opt.expectedDelta_b2 = zeros(obj.visibleSize, 1);
            end
            
            trCosts = zeros(obj.nEpochs*nBatches,1);
            iter    = 0;
            
            %--------------------------------------------------------------
            % conduct optimization using mini-batch gradient descent
            for ee = 1:obj.nEpochs
                
                % shuffle dataset
                sampleIdx = randperm(nSamples);
                
                % sweep over batches
                for ii = 1:nBatches
                    
                    % for adadelta
                    iter = iter + 1;
                    
                    % index calculation for current batch
                    batchIdx = sampleIdx((ii - 1) * obj.batchSize + 1 : ii * obj.batchSize);
                    
                    % get cost and gradient for training data
                    if obj.isLinearCost
                        [trCostThis, grad] = obj.sparseAutoencoderLinearCost( obj.theta, obj.visibleSize, obj.hiddenSize, ...
                            obj.lambda, obj.lambdaL1, obj.sparsityParam, obj.beta, obj.x(:, batchIdx));
                    else
                        [trCostThis, grad] = obj.sparseAutoencoderCost( obj.theta, obj.visibleSize, obj.hiddenSize, ...
                            obj.lambda, obj.lambdaL1, obj.sparsityParam, obj.beta, obj.x(:, batchIdx));
                    end
                    
                    % update weights with momentum and lrate options
                    opt = updateWeights(obj, opt, grad, iter);
                    
                    % keep record of costs
                    trCosts((ee-1) * nBatches + ii) = trCostThis;
                    
                end
                
                if ~obj.silent
                    curr_speed = [opt.curr_speed_W1(:); opt.curr_speed_W2(:); opt.curr_speed_b1(:); opt.curr_speed_b2(:)];
                    fprintf('Epoch:[%d/%d] - Training error:[%f] - NormSpeed:[%f]\n',...
                        ee, obj.nEpochs, trCosts((ee-1)*nBatches+ii), norm(curr_speed(:)));
                end
                
            end
            
            theta = obj.theta;
            obj.trErr = trCosts;
            
            if ~obj.silent
                figure('color','white');
                hold on;
                plot(obj.trErr, 'b'),legend('unsupervised training'),...
                    ylabel('reconstruction error'),xlabel('iteration number');
                hold off;
            end            
            
        end
        
        
        %==================================================================
        %   Updates weights of the model. Momentum or adadelta is employed.
        %==================================================================
        function opt = updateWeights(obj, opt, grad, iter)
            
            W1grad = reshape(grad(1:obj.hiddenSize*obj.visibleSize), obj.hiddenSize, obj.visibleSize);
            W2grad = reshape(grad(obj.hiddenSize*obj.visibleSize+1:2*obj.hiddenSize*obj.visibleSize), obj.visibleSize, obj.hiddenSize);
            b1grad = grad(2*obj.hiddenSize*obj.visibleSize+1:2*obj.hiddenSize*obj.visibleSize+obj.hiddenSize);
            b2grad = grad(2*obj.hiddenSize*obj.visibleSize+obj.hiddenSize+1:end);
            
            W1 = reshape(obj.theta(1:obj.hiddenSize*obj.visibleSize), obj.hiddenSize, obj.visibleSize);
            W2 = reshape(obj.theta(obj.hiddenSize*obj.visibleSize+1:2*obj.hiddenSize*obj.visibleSize), obj.visibleSize, obj.hiddenSize);
            b1 = obj.theta(2*obj.hiddenSize*obj.visibleSize+1:2*obj.hiddenSize*obj.visibleSize+obj.hiddenSize);
            b2 = obj.theta(2*obj.hiddenSize*obj.visibleSize+obj.hiddenSize+1:end);
            
            if obj.useAdaDelta
                
                % fancy way of setting decay_rate to zero if iter == 1
                decay_rate = ((iter-1) * obj.adaDeltaRho / (iter-1+eps));
                tiny = 1e-6;
                
                % first apply momentum on weights
                opt.curr_speed_b1 = ((1-obj.momentum) * b1grad) + (obj.momentum * opt.curr_speed_b1);
                opt.curr_speed_b2 = ((1-obj.momentum) * b2grad) + (obj.momentum * opt.curr_speed_b2);
                opt.curr_speed_W1 = ((1-obj.momentum) * W1grad) + (obj.momentum * opt.curr_speed_W1);
                
                % accumulate gradients
                opt.expectedGrads_b1 = (decay_rate * opt.expectedGrads_b1) +((1-decay_rate) * opt.curr_speed_b1.^2);
                opt.expectedGrads_b2 = (decay_rate * opt.expectedGrads_b2) +((1-decay_rate) * opt.curr_speed_b2.^2);
                opt.expectedGrads_W1 = (decay_rate * opt.expectedGrads_W1) +((1-decay_rate) * opt.curr_speed_W1.^2);
                
                % compute update
                delta_b1 = -opt.curr_speed_b1 .* (sqrt(opt.expectedDelta_b1 + tiny) ./ sqrt(opt.expectedGrads_b1 + tiny));
                delta_b2 = -opt.curr_speed_b2 .* (sqrt(opt.expectedDelta_b2 + tiny) ./ sqrt(opt.expectedGrads_b2 + tiny));
                delta_W1 = -opt.curr_speed_W1 .* (sqrt(opt.expectedDelta_W1 + tiny) ./ sqrt(opt.expectedGrads_W1 + tiny));
                
                % accumulate updates
                opt.expectedDelta_b1 = (decay_rate * opt.expectedDelta_b1) + ((1-decay_rate) * delta_b1.^2);
                opt.expectedDelta_b2 = (decay_rate * opt.expectedDelta_b2) + ((1-decay_rate) * delta_b2.^2);
                opt.expectedDelta_W1 = (decay_rate * opt.expectedDelta_W1) + ((1-decay_rate) * delta_W1.^2);
                
                % apply update
                b1 = b1 + delta_b1;
                b2 = b2 + delta_b2;
                W1 = W1 + delta_W1;

                if ~obj.tied
                    opt.curr_speed_W2 = ((1-obj.momentum) * W2grad) + (obj.momentum * opt.curr_speed_W2);
                    opt.expectedGrads_W2 = (decay_rate * opt.expectedGrads_W2) +((1-decay_rate) * opt.curr_speed_W2.^2);
                    delta_W2 = -opt.curr_speed_W2 .* (sqrt(opt.expectedDelta_W2 + tiny) ./ sqrt(opt.expectedGrads_W2 + tiny));
                    opt.expectedDelta_W2 = (decay_rate * opt.expectedDelta_W2) + ((1-decay_rate) * delta_W2.^2);
                    W2 = W2 + delta_W2;
                end
                
            else
                % calculate momentum speed and update weights
                opt.curr_speed_W1 = opt.curr_speed_W1 * obj.momentum - W1grad;
                opt.curr_speed_b1 = opt.curr_speed_b1 * obj.momentum - b1grad;
                opt.curr_speed_b2 = opt.curr_speed_b2 * obj.momentum - b2grad;
                W1 = W1 + opt.curr_speed_W1 * obj.alpha;
                b1 = b1 + opt.curr_speed_b1 * obj.alpha;
                b2 = b2 + opt.curr_speed_b2 * obj.alpha;

                if ~obj.tied
                    opt.curr_speed_W2 = opt.curr_speed_W2 * obj.momentum - W2grad;
                    W2 = W2 + opt.curr_speed_W2 * obj.alpha;
                end
            end
            
            % actual update here
            obj.theta = [W1(:); W2(:); b1(:); b2(:)];
            
        end
        
        
        %==================================================================
        % checks model parameters for sparse encoder using gradient
        % descent
        %==================================================================
        function isValid = isModelValid(obj)
            isValid = false;
            
            if isempty(obj.x),           return; end
            if isempty(obj.hiddenSize),  return; end
            if isempty(obj.visibleSize), return; end
            %             if isempty(obj.minfunc),     return; end
            
            isValid = true;
        end
        
        
    end
    
    
    
end

