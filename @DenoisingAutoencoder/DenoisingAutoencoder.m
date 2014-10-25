classdef DenoisingAutoencoder < handle
%==========================================================================
% Class for standard denoising autoencoders
%
%  This class implements denoising autoencoders in Vincent, Pascal, et al.
%  "Extracting and composing robust features with denoising autoencoders."
%  Proceedings of the 25th international conference on Machine learning. 
%  ACM, 2008. An autoencoder is a neural network having one hidden layer
%  and aims to reconstruct its input in the output layer, trained by
%  back-propagation. Mean Squared cost and Cross entopy cost functions are
%  available to measure error. L2 regularization is applied to weights.
%
%  Momentum method is provided in order to improve optimization. External 
%  optimizer is allowed by setting minfunc parameter to a minimization 
%  function handle. Adadelta method is also provided for tuning learning
%  rate. Tied/untied weights, linear cost for output layer, sigmoid, tanh,
%  rectified linear and linear activation for hidden neurons are provided.
%
% orhanf - (c) 2014 Orhan FIRAT <orhan.firat@ceng.metu.edu.tr>
%==========================================================================
    
    properties
        % model parameters
        W1;            % transition parameters (matrix) - visible to hidden
        W2;            % transition parameters (matrix) - hidden to reconstruction
        b1;            % bias parameters (vector) - visible to hidden
        b2;            % bias parameters (vector) - hidden to reconstruction
        lambda;        % regularization (weight decay) parameter
        nu;            % input denoising parameter in range [0,1)
        drop;          % dropping ratio to apply a dropping mask
        nEpochs;       % number of passes through the dataset
        batchSize;     % mini-batch size
        alpha;         % learning rate
        momentum;      % momentum parameter [0,1]
        hActFun;       % hidden-unit activation functions: 0-sigmoid, 1-tanh, 2-relu, 3-linear
        vActFun;       % visible-unit activation functions: 0-sigmoid, 1-tanh, 2-relu, 3-linear
        errFun;        % error function: 0-mean squared error, 1-cross entropy
        adaDeltaRho;   % decay rate rho, for adadelta 
        trErr;         % training error vector
        
        rho;           % desired average activation of the hidden units
        beta;          % weight of sparsity penalty term
        
        visibleSize;   % number of input units
        hiddenSize;    % number of hidden units
        x;             % input training data
        
        verbose;       % display cost in each iteration etc.
        minfunc;       % minimization function handle for optimization
        J;             % cost function values (vector)
        initFlag;      % random initialization by default for W
        isLinearCost;  % Use linear cost function
        addBias;       % adding bias vector or not
        tiedWeights;   % sets W2 = W1' if tied, W2~=W1 for untied
        useAdaDelta;   % use adaptive delta to estimate learning rate
        useGPU;        % Use gpu to speed-up if supported
    end
    
    
    methods
        
        %==================================================================
        % constructor
        %==================================================================
        function obj = DenoisingAutoencoder(options)
            
            obj.lambda       = 1e-3;
            obj.nu           = 0;       % input corruption level
            obj.drop         = 0.5;     % input dropping mask
            obj.verbose      = false;   % verbose
            obj.initFlag     = true;
            obj.isLinearCost = false;   % non-linear by default
            obj.addBias      = true;    % add bias to input and hidden layers
            obj.nEpochs      = 100;
            obj.batchSize    = 100;
            obj.alpha        = .01;
            obj.momentum     = 0.9;
            obj.hActFun      = 0;       % use sigmoid by default
            obj.vActFun      = 0;       % use sigmoid by default
            obj.errFun       = 0;       % use mean squared error
            obj.tiedWeights  = false;
            obj.useAdaDelta  = false;
            obj.adaDeltaRho  = 0.95;
            obj.useGPU       = false;
            obj.rho          = .01;
            obj.beta         = 0;       % meaning no sparsity term by default
            
            if nargin>0 && isstruct(options)
                if isfield(options,'verbose'),      obj.verbose      = options.verbose;      end
                if isfield(options,'lambda'),       obj.lambda       = options.lambda;       end
                if isfield(options,'nu'),           obj.nu           = options.nu;           end
                if isfield(options,'drop'),         obj.drop         = options.drop;         end
                if isfield(options,'visibleSize'),  obj.visibleSize  = options.visibleSize;  end
                if isfield(options,'hiddenSize'),   obj.hiddenSize   = options.hiddenSize;   end
                if isfield(options,'x'),            obj.x            = options.x;            end
                if isfield(options,'initFlag'),     obj.initFlag     = options.initFlag;     end
                if isfield(options,'minfunc'),      obj.minfunc      = options.minfunc;      end
                if isfield(options,'W1'),           obj.W1           = options.W1;           end
                if isfield(options,'b1'),           obj.b1           = options.b1;           end
                if isfield(options,'W2'),           obj.W2           = options.W2;           end
                if isfield(options,'b2'),           obj.b2           = options.b2;           end
                if isfield(options,'isLinearCost'), obj.isLinearCost = options.isLinearCost; end
                if isfield(options,'addBias'),      obj.addBias      = options.addBias;      end
                if isfield(options,'nEpochs'),      obj.nEpochs      = options.nEpochs;      end
                if isfield(options,'batchSize'),    obj.batchSize    = options.batchSize;    end
                if isfield(options,'alpha'),        obj.alpha        = options.alpha;        end
                if isfield(options,'momentum'),     obj.momentum     = options.momentum;     end
                if isfield(options,'hActFun'),      obj.hActFun      = options.hActFun;      end
                if isfield(options,'vActFun'),      obj.vActFun      = options.vActFun;      end
                if isfield(options,'errFun'),       obj.errFun       = options.errFun;       end
                if isfield(options,'tiedWeights'),  obj.tiedWeights  = options.tiedWeights;  end
                if isfield(options,'useAdaDelta'),  obj.useAdaDelta  = options.useAdaDelta;  end
                if isfield(options,'adaDeltaRho'),  obj.adaDeltaRho  = options.adaDeltaRho;  end                
                if isfield(options,'trErr'),        obj.trErr        = options.trErr;        end
                if isfield(options,'useGPU'),       obj.useGPU       = options.useGPU;       end
                if isfield(options,'rho'),          obj.rho          = options.rho;          end
                if isfield(options,'beta'),         obj.beta         = options.beta;         end
            end
            
            % check if matlab supports GPU operations
            if obj.useGPU
                v = version('-release');
                if str2double(v(1:4))<2012, obj.useGPU = false; end
                try 
                    obj.useGPU = (0<gpuDeviceCount);
                catch err
                    obj.useGPU = false;                        
                end
            end 
            
            if obj.isLinearCost, obj.vActFun = 3;           end
            if obj.initFlag,     obj.initializeParameters;  end
            
        end
        
        
        %==================================================================
        % Train model using stochastic gradient descent (SGD) on mini 
        % batches or batch gradient descent(BGD) using external
        % minimization function.
        %==================================================================
        function [theta, cost] = train_model(obj)
            
            theta = 0;
            cost  = 0;
            
            if isModelValid(obj)
                try
                    if obj.useGPU
                        [theta, cost] = trainDenoisingAutoencoderGPU(obj);
                    elseif isempty(obj.minfunc)               
                        [theta, cost] = trainDenoisingAutoencoderSGD(obj);                        
                    else
                        [theta, cost] = trainDenoisingAutoencoderBGD(obj);
                    end
                catch err
                    fprintf(2,'Denoising Autoencoder Optimization terminated with error:\n%s\n', err.getReport);
                end
            else
                fprintf(2,'Denoising Auto-encoder Model is not valid! Please check input options.');
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
                
                r  = sqrt(6) / sqrt(obj.hiddenSize+obj.visibleSize+1);   % choose weights uniformly from the interval [-r, r]
                W1tmp = (rand(obj.hiddenSize, obj.visibleSize) - .5) * 2 * 4 * r;
                W2tmp = (rand(obj.visibleSize, obj.hiddenSize) - .5) * 2 * 4 * r;
                b1tmp = zeros(obj.hiddenSize, 1);
                b2tmp = zeros(obj.visibleSize,1);
                
                thetaTMP = [W1tmp(:); W2tmp(:); b1tmp(:); b2tmp(:)];
                
                [~, grad] = obj.denoisingAutoencoderCostBGD(thetaTMP, obj.x);                
                
                % Check cost function and derivative calculations
                % for the sparse autoencoder.  
                numgrad = obj.computeNumericalGradient( @(x)obj.denoisingAutoencoderCostBGD(x, obj.x), thetaTMP);

                diff = norm(numgrad-grad)/norm(numgrad+grad);
                                                              
                % Use this to visually compare the gradients side by side
                % Compare numerically computed gradients with the ones obtained from backpropagation
                if ~obj.verbose
                    disp([numgrad grad]); 
                    disp(diff); % Should be small. 
                end                                
            end
            
        end
        
        
        %==================================================================
        % Re-initialize weights of the model
        %==================================================================
        function isSuccess = reset_model(obj)
            isSuccess = true;
            try
                obj.initializeParameters;
            catch err
                fprintf(2,'Reset terminated with error:\n%s\n', err.getReport);
                isSuccess = false;
            end
        end
        
        
        %==================================================================
        % Plots training error vs iteration number. Call after training. 
        %==================================================================
        function h = plot_training_error(obj)
            h = [];
            if ~isempty(obj.trErr)
                h = figure('color','white');
                hold on;
                plot(obj.trErr, 'b');
                legend('unsupervised training');
                ylabel('reconstruction error');
                xlabel('iteration number');
                hold off;
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
            
            corrMat = corr(obj.W1');  % calculate correlation matrix
            corrMat(2,2) = -1;        % this is for color correction
            h = figure; imagesc(corrMat); colorbar;

            % calculate frobenius norm 
            Frobenius_norm = sqrt(sum(abs(corrMat(:)).^2));
            
            % L1 distance of diagonalized corrMat to identity matrix
            L1_dist_to_identity = sum(abs(ones(obj.hiddenSize,1)- svd(corrMat)));
            
        end
        
        
        %==================================================================
        % Encoder function of the autoencoder
        %==================================================================
        function mappedData = encodeX(obj, data)
            try                                
                mappedData = nonLinearity(bsxfun(@plus, obj.W1 * data, obj.b1), obj.hActFun);                 
            catch err
                fprintf(2,'Error in encodeX function!\n%s\n',err.getReport);
                mappedData = [];
            end
        end
        
        
        %==================================================================
        % Returns parameters as a stack of 2 layers
        %==================================================================
        function params = get_parameters_as_stack(obj)
                                               
            params{1}.w = obj.W1;
            params{2}.w = obj.W2;            
            
            params{1}.b = obj.b1;
            params{2}.b = obj.b2;            
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
            W = obj.W1;
            b = obj.b1;            
                
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
        % checks model parameters for sparse encoder using gradient
        % descent
        %==================================================================
        function isValid = isModelValid(obj)
            isValid = false;
            
            if isempty(obj.x),           return; end
            if isempty(obj.hiddenSize),  return; end
            if isempty(obj.visibleSize), return; end
            
            isValid = true;
        end
        
        
        %==================================================================
        % Initialize parameters randomly based on layer sizes. Transition
        % weights W1 and W2 are initialized by sampling a uniform
        % distribution from range -4*r to 4*r, where r is a function of
        % fan_in and fan_out of hidden layer.
        %==================================================================
        function initializeParameters(obj)
            
            if obj.useGPU                
                r  = sqrt(6) / sqrt(obj.hiddenSize+obj.visibleSize+1);   % choose weights uniformly from the interval [-r, r]
                obj.W1 = gpuArray((rand(obj.hiddenSize, obj.visibleSize) - .5) * 2 * 4 * r);
                obj.W2 = gpuArray((rand(obj.visibleSize, obj.hiddenSize) - .5) * 2 * 4 * r);
                obj.b1 = gpuArray(zeros(obj.hiddenSize, 1));
                obj.b2 = gpuArray(zeros(obj.visibleSize,1));

                if obj.tiedWeights
                    obj.W2 = gpuArray(zeros(obj.visibleSize, obj.hiddenSize));
                end                        
            
            else
                r  = sqrt(6) / sqrt(obj.hiddenSize+obj.visibleSize+1);   % choose weights uniformly from the interval [-r, r]
                obj.W1 = (rand(obj.hiddenSize, obj.visibleSize) - .5) * 2 * 4 * r;
                obj.W2 = (rand(obj.visibleSize, obj.hiddenSize) - .5) * 2 * 4 * r;
                obj.b1 = zeros(obj.hiddenSize, 1);
                obj.b2 = zeros(obj.visibleSize,1);

                if obj.tiedWeights
                    obj.W2 = zeros(obj.visibleSize, obj.hiddenSize);
                end
            end
            
        end
        
        
        %==================================================================
        % 
        %==================================================================
        function opt = initializeOptimizationHelpers(obj)
            
            opt.curr_speed_W1 = obj.W1 * 0;
            opt.curr_speed_W2 = obj.W2 * 0;
            opt.curr_speed_b1 = obj.b1 * 0;
            opt.curr_speed_b2 = obj.b2 * 0;
            
            if obj.useGPU                
                if obj.useAdaDelta
                    opt.expectedGrads_W1 = gpuArray(zeros(obj.hiddenSize, obj.visibleSize)); % adadelta : E[g^2]_0
                    opt.expectedDelta_W1 = gpuArray(zeros(obj.hiddenSize, obj.visibleSize)); % adadelta : E[\delta x^2]_0
                    opt.expectedGrads_W2 = gpuArray(zeros(obj.visibleSize, obj.hiddenSize)); 
                    opt.expectedDelta_W2 = gpuArray(zeros(obj.visibleSize, obj.hiddenSize)); 
                    opt.expectedGrads_b1 = gpuArray(zeros(obj.hiddenSize, 1));
                    opt.expectedDelta_b1 = gpuArray(zeros(obj.hiddenSize, 1));
                    opt.expectedGrads_b2 = gpuArray(zeros(obj.visibleSize, 1));
                    opt.expectedDelta_b2 = gpuArray(zeros(obj.visibleSize, 1));
                end                             
            else
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
            end
            
        end
        
        
        %==================================================================
        % Train denosing autoencoder with backprop using mini-batch
        % stochastic graident descent for given number of epochs.
        %==================================================================
        function [theta, trCosts] = trainDenoisingAutoencoderSGD(obj)
            
            % get helpers
            nSamples = size(obj.x, 2);
            if obj.batchSize > nSamples
                obj.batchSize = nSamples;
            end
            nBatches = floor(nSamples / obj.batchSize);
            
            % optimization
            opt = obj.initializeOptimizationHelpers;
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
                    [trCostThis, gradients] = obj.denoisingAutoencoderCostSGD( obj.x(:, batchIdx));
                    
                    % update weights with momentum and lrate options
                    opt = updateWeights(obj, opt, gradients, iter);
                                                            
                    % keep record of costs
                    trCosts((ee-1) * nBatches + ii) = trCostThis;
                    
                end
                
                if ~obj.verbose
                    curr_speed = [opt.curr_speed_W1(:); opt.curr_speed_W2(:); opt.curr_speed_b1(:); opt.curr_speed_b2(:)];
                    fprintf('Epoch:[%d/%d] - Training error:[%f] - NormSpeed:[%f]\n',...
                        ee, obj.nEpochs, trCosts((ee-1)*nBatches+ii), norm(curr_speed(:)));
                end
                
            end
            
            theta = [obj.W1(:); obj.W2(:); obj.b1(:); obj.b2(:)];
            obj.trErr = trCosts;
            
            if ~obj.verbose
                h = obj.plot_training_error;
            end
        end
        
        
        %==================================================================
        % Train denosing autoencoder with backprop using mini-batch
        % stochastic graident descent for given number of epochs on GPU
        %==================================================================
        function [theta, trCosts] = trainDenoisingAutoencoderGPU(obj)
            
            % get helpers
            nSamples = size(obj.x, 2);
            if obj.batchSize > nSamples
                obj.batchSize = nSamples;
            end
            nBatches = floor(nSamples / obj.batchSize);
            
            % optimization
            opt = obj.initializeOptimizationHelpers;
            
            trCosts = gpuArray(zeros(obj.nEpochs*nBatches,1));
            iter    = 0;            
            trData  = gpuArray(obj.x);
            
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
                    [trCostThis, gradients] = obj.denoisingAutoencoderCostSGD( trData(:, batchIdx));
                    
                    % update weights with momentum and lrate options
                    opt = updateWeights(obj, opt, gradients, iter);
                                                            
                    % keep record of costs
                    trCosts((ee-1) * nBatches + ii) = trCostThis;
                    
                end
                
                if ~obj.verbose
                    curr_speed = [opt.curr_speed_W1(:); opt.curr_speed_W2(:); opt.curr_speed_b1(:); opt.curr_speed_b2(:)];
                    fprintf('Epoch:[%d/%d] - Training error:[%f] - NormSpeed:[%f]\n',...
                        ee, obj.nEpochs, gather(trCosts((ee-1)*nBatches+ii)), gather(norm(curr_speed(:))));
                end
                
            end
            
            theta = gather([obj.W1(:); obj.W2(:); obj.b1(:); obj.b2(:)]);
            obj.trErr = gather(trCosts);
            
            % move parameters from GPU
            obj.W1 = gather(obj.W1);
            obj.W2 = gather(obj.W2);
            obj.b1 = gather(obj.b1);
            obj.b2 = gather(obj.b2);
            
            if ~obj.verbose
                h = obj.plot_training_error;
            end
        end
        
        
        %==================================================================
        % Train denosing autoencoder with backprop using batch gradient 
        % descent with given minimization function. (e.g. minFunc)
        %==================================================================
        function [theta, cost] = trainDenoisingAutoencoderBGD(obj)
            
            theta = [obj.W1(:); obj.W2(:); obj.b1(:); obj.b2(:)];
            [theta, cost] = obj.minfunc( @(p) obj.denoisingAutoencoderCostBGD(p, obj.x), theta); 

            % extract parameters
            obj.W1 = reshape(theta(1:obj.hiddenSize*obj.visibleSize), obj.hiddenSize, obj.visibleSize);
            obj.W2 = reshape(theta(obj.hiddenSize*obj.visibleSize+1:2*obj.hiddenSize*obj.visibleSize), obj.visibleSize, obj.hiddenSize);
            obj.b1 = theta(2*obj.hiddenSize*obj.visibleSize+1:2*obj.hiddenSize*obj.visibleSize+obj.hiddenSize);
            obj.b2 = theta(2*obj.hiddenSize*obj.visibleSize+obj.hiddenSize+1:end);      
            
        end
        
        
        %==================================================================
        %   Updates weights of the model. Momentum or adadelta is employed.
        %==================================================================
        function opt = updateWeights(obj, opt, gradients, iter)        
            
            if obj.useAdaDelta

                % fancy way of setting decay_rate to zero if iter == 1
                decay_rate = ((iter-1) * obj.adaDeltaRho / (iter-1+eps));                                
                tiny = 1e-6;
                
                % first apply momentum on weights
                opt.curr_speed_b1 = ((1-obj.momentum) * gradients.b1grad) + (obj.momentum * opt.curr_speed_b1);
                opt.curr_speed_b2 = ((1-obj.momentum) * gradients.b2grad) + (obj.momentum * opt.curr_speed_b2);
                opt.curr_speed_W1 = ((1-obj.momentum) * gradients.W1grad) + (obj.momentum * opt.curr_speed_W1);                
                
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
                obj.b1 = obj.b1 + delta_b1;
                obj.b2 = obj.b2 + delta_b2;
                obj.W1 = obj.W1 + delta_W1;
                                                
                if ~obj.tiedWeights 
                    opt.curr_speed_W2    = ((1-obj.momentum) * gradients.W2grad) + (obj.momentum * opt.curr_speed_W2);
                    opt.expectedGrads_W2 = (decay_rate * opt.expectedGrads_W2) +((1-decay_rate) * opt.curr_speed_W2.^2); 
                    delta_W2 = -opt.curr_speed_W2 .* (sqrt(opt.expectedDelta_W2 + tiny) ./ sqrt(opt.expectedGrads_W2 + tiny));
                    opt.expectedDelta_W2 = (decay_rate * opt.expectedDelta_W2) + ((1-decay_rate) * delta_W2.^2);
                    obj.W2 = obj.W2 + delta_W2;
                end
                
            else
                % calculate momentum speed and update weights
                opt.curr_speed_W1 = opt.curr_speed_W1 * obj.momentum - gradients.W1grad;                
                opt.curr_speed_b1 = opt.curr_speed_b1 * obj.momentum - gradients.b1grad;
                opt.curr_speed_b2 = opt.curr_speed_b2 * obj.momentum - gradients.b2grad;                            
                obj.W1 = obj.W1 + opt.curr_speed_W1 * obj.alpha;                
                obj.b1 = obj.b1 + opt.curr_speed_b1 * obj.alpha;
                obj.b2 = obj.b2 + opt.curr_speed_b2 * obj.alpha;
                
                if ~obj.tiedWeights
                    opt.curr_speed_W2 = opt.curr_speed_W2 * obj.momentum - gradients.W2grad;
                    obj.W2 = obj.W2 + opt.curr_speed_W2 * obj.alpha;
                end
            end
        end
    end
end

