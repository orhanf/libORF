classdef NeuralNet < handle
%==========================================================================
% Class for Feed-Forward Neural Network
%
%  This class implements feed-forward Neural Network with sigmoid neurons
%  at the hidden layers and softmax at the output layer. Neural network is
%  trained by backpropagation. Cross entopy cost function is used to
%  measure error and its derivative wrt activation of penultimate layer is
%  backpropagated during training. L2 regularization is applied to weights.
%
%  For hidden unit activation functions; sigmoid, hyperbolic-tangent,
%  scaled hyperbolic-tangent, rectified linear units, linear activation 
%  and maxout are provided.
%
%  Momentum method is provided in order to improve optimization, early
%  stopping criterion is also provided for regularization. Arbitrary
%  network structure is allowed. External optimizer is allowed by setting
%  trainingRegime parameter. 4 options allowed for training regime. (1) for
%  training mini-batch gradient descent upto a predefined number of 
%  iterations. (2) optimization using mini-batch gradient descent with
%  epochs. (3) using an external optimizer in batch gradient descent and
%  (4) using an external optimizer in stochastic gradient descent.
%  
%  Common tricks such as Dropout, momentum, adadelta are implemented along
%  with early stopping when a cross validation set is supplied.
%  Pre-training using autoencoders is also supported internally.
%
%  Computing backprop at GPU is supported for releases staring Matlab2012a
%  
%   TODO : fix addBias issues
%   TODO : fix transpose in pre-training
%   TODO : add RBM pre-training
%   TODO : add L1 regularization to GPU cost function
%
% orhanf - (c) 2013 Orhan FIRAT <orhan.firat@ceng.metu.edu.tr>
%==========================================================================
    
    properties
        
        nnLayers;    % Cell array of structs defining properties of each 
                     % layer, including input and output layers.
                     % Each struct defines a layer, should have following 
                     %  fields:
                     %  'type'     : Type of layer
                     %                  'input'  - input layer
                     %                  'fc'     - fully connected layer
                     %                  'output' - output layer
                     %  'nNeurons' : Number of neurons in the layer <int>
                     %  'act'      : Activation function 
                     %                  'sigmoid'- sigmod
                     %                  'relu'   - rectified linear unit
                     %                  'tanh'   - hyperbolic tangent
                     %                  'stanh'  - scaled tanh
                     %                  'linear' - linear activation
                     %                  'maxout' - maxout
                     %                  'softmax'- softmax for output
                     %  'poolSize' : Pooling size for maxout                     
                
        trainData;   % Training data <nFeatures,nSamples> matrix
        trainLabels; % Labels for training data <nClass, nSamples> matrix in one of k encoding
        cvData;      % Cross-validation data
        cvLabels;    % Labels for cross-validation data
        trErrors;    % Error vector for training phase
        cvErrors;    % Error vector for validation phase
        
        nEpochs;     % Number of passes through the dataset
        nIters;      % Number of iterations in each epoch
        batchSize;   % Mini-batch size.
        alpha;       % Learning rate
        momentum;    % Momentum parameter [0,1]
        lambdaL2;    % Weight decay parameter (for L2 regularization)
        lambdaL1;    % Weight decay parameter (for L1 regularization)
        stdInitW;    % Standard deviation of the normal distribution which is sampled to get the initial weights
        addBias;     % Add bias to input and hidden layers
        dropOutRatio;% Employ drop-out method for hidden units
        useAdaDelta; % use adaptive delta to estimate learning rate
        adaDeltaRho; % decay rate rho, for adadelta
        
        
        nnTheta;     % Initial model parameters
        nnOptTheta;  % Optimum model parameters
        
        silent;          % Display cost in each iteration etc (verbose)
        isEarlyStopping; % Apply early stopping criterion with best cv error
        trainingRegime;  % 0='epochs', 1='iters', 3='external-BGD', 4='external-SGD'
        minfunc;         % External minimization function handle
        initFanInOut;    % Initialize weights wrt fan-in and fan-out
        useGPU;          % Use gpu to speed-up if supported
    end
    
    properties(Hidden)
        layers;         % Data structure for neural net config and parameters
        isCV;           % Flag for cross validation set is supplied
        grads;          % Gradient matrices and vectors that are pre-allocated on GPU if useGPU=true
        hiddenSizes;    % Array for hidden unit sizes, excluding input and output layers
        outputSize;     % Number of units in the output layer as it is softmax
        inputSize;      % Number of units in one of the input layers
        oActFun;        % Output layer function, either 'softmax', 'sigmoid' or 'linear' (default 'softmax')
        hActFuns;       % Hidden layer activation functions, (cell) array with the same length as hiddenSizes                        
    end
    
    methods
        
        %==================================================================
        % constructor
        %==================================================================
        function obj = NeuralNet(options)
            
            % defaults
            obj.nEpochs      = 100;
            obj.nIters       = 1000;
            obj.batchSize    = 100;
            obj.alpha        = 0.1;
            obj.momentum     = 0.9;
            obj.lambdaL2     = 1e-3;
            obj.lambdaL1     = 0;
            obj.stdInitW     = 0.01;
            obj.addBias      = true;
            obj.dropOutRatio = 0;
            obj.useAdaDelta  = false;
            obj.adaDeltaRho  = 0.95;
            obj.oActFun      = 'softmax';
            
            % defaults
            obj.trainingRegime  = 2;
            obj.isEarlyStopping = false;
            obj.silent          = false;
            obj.isCV            = false;
            obj.initFanInOut    = false;
            obj.useGPU          = false; 
            
            if nargin>0 && isstruct(options)
                
                if isfield(options,'trainData'),       obj.trainData        = options.trainData;        end
                if isfield(options,'trainLabels'),     obj.trainLabels      = options.trainLabels;      end
                if isfield(options,'cvData'),          obj.cvData           = options.cvData;           end
                if isfield(options,'cvLabels'),        obj.cvLabels         = options.cvLabels;         end
                
                if isfield(options,'nnLayers'),        obj.nnLayers         = options.nnLayers;         end
                if isfield(options,'nnTheta'),         obj.nnTheta          = options.nnTheta;          end
                if isfield(options,'nnOptTheta'),      obj.nnOptTheta       = options.nnOptTheta;       end
                if isfield(options,'batchSize'),       obj.batchSize        = options.batchSize;        end
                if isfield(options,'alpha'),           obj.alpha            = options.alpha;            end
                if isfield(options,'momentum'),        obj.momentum         = options.momentum;         end
                if isfield(options,'lambdaL2'),        obj.lambdaL2         = options.lambdaL2;         end
                if isfield(options,'lambdaL1'),        obj.lambdaL1         = options.lambdaL1;         end
                if isfield(options,'stdInitW'),        obj.stdInitW         = options.stdInitW;         end
                if isfield(options,'addBias'),         obj.addBias          = options.addBias;          end
                if isfield(options,'dropOutRatio'),    obj.dropOutRatio     = options.dropOutRatio;     end
                if isfield(options,'useAdaDelta'),     obj.useAdaDelta      = options.useAdaDelta;      end
                if isfield(options,'adaDeltaRho'),     obj.adaDeltaRho      = options.adaDeltaRho;      end
                
                if isfield(options,'silent'),          obj.silent           = options.silent;           end
                if isfield(options,'isEarlyStopping'), obj.isEarlyStopping  = options.isEarlyStopping;  end
                if isfield(options,'initFanInOut'),    obj.initFanInOut     = options.initFanInOut;     end                
                if isfield(options,'useGPU'),          obj.useGPU           = options.useGPU;           end
                
                % This is crucial for using Adadelta
                if obj.useAdaDelta, obj.initFanInOut = true;    end                                
                
                % at worst case, choose 'epochs'
                if isfield(options,'minfunc'),obj.minfunc = options.minfunc; obj.trainingRegime = 3; end
                if isfield(options,'nIters'), obj.nIters  = options.nIters;  obj.trainingRegime = 1; end
                if isfield(options,'nEpochs'),obj.nEpochs = options.nEpochs; obj.trainingRegime = 2; end
                if isfield(options,'nEpochs') && isfield(options,'minfunc'), obj.trainingRegime = 4; end
                if ~isempty(obj.cvData) && ~isempty(obj.cvLabels),   obj.isCV = true;                end
                
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
                                                
                % Initialize layers
                obj.init_layers;
                
                % Initialize parameters 
                obj.init_layer_params;
                
            end
        end
        
        
        %==================================================================
        % Train model given optimization regime
        %==================================================================
        function [theta] = train_model(obj)
            try
                if isModelValid(obj)
                    if obj.useGPU
                        [theta] = obj.train_model_onGPU;
                    elseif obj.trainingRegime == 1      % using 'nIters'
                        [theta] = obj.train_model_regime1;
                    elseif obj.trainingRegime == 2      % using 'nEpochs'
                        [theta] = obj.train_model_regime2;
                    elseif obj.trainingRegime == 3      % using minfunc in BGD
                        [theta] = obj.train_model_regime3;
                    elseif obj.trainingRegime == 4      % using minfunc in SGD
                        [theta] = obj.train_model_regime4;
                    else
                        error('Undefined training regime indicator!');
                    end
                    
                    obj.nnOptTheta = theta;
                    
                    if ~obj.silent && obj.trainingRegime ~= 3
                        figure('color','white');
                        hold on;
                        plot(obj.trErrors, 'b');
                        if obj.isCV
                            plot(obj.cvErrors, 'r');
                            legend('training', 'validation');
                        else
                            legend('training');
                        end
                        
                        ylabel('loss');
                        xlabel('iteration number');
                        hold off;
                    end
                else
                    error('Model is not valid, please check your input parameters!');
                end
            catch err
                theta = [];
                disp(['NeuralNet Optimization terminated with error:' err.getReport]);
            end
        end
        
                      
        %==================================================================
        %   Predict new samples with trained model
        %==================================================================
        function pred = predict_samples(obj, data)
            
            if ~isempty(obj.nnOptTheta) && ~isempty(data)
                
                act = obj.feedForwardNN(data);
                
                [dummy, pred]= max(act);
            else
                fprintf(2,'Prediction not possible!\nYour model may not yet been trained or data is empty.\n')                ;
            end
        end
        
        
        %==================================================================
        % Sets neural network parameters given the stack of parameters for
        % each layer including transition weight matrices W and bias
        % vectors b.
        %==================================================================
        function nnTheta = init_params_by_stack(obj,SAstack)
            
            nnTheta = obj.stack2params(SAstack);
            
        end
            
            
        %==================================================================
        % Resets parameter vectors and information about previous runs
        %==================================================================
        function reset(obj)
            obj.nnOptTheta = [];
            obj.trErrors   = [];
            obj.cvErrors   = [];
            
            % Initialize parameter vectors with zeros
            obj.init_layer_params;
        end
        
        
        %==================================================================
        % Resets parameter vectors and information about previous runs
        %==================================================================
        function params = pre_train_model(obj, data, opts)
            
            params = struct('W',[],'b',[]);
            
            if nargin<3 % use default parameters
                opts = struct('aeType',[],'aeOpt',[]);
                for i=1:numel(obj.hiddenSizes)
                    opts(i).aeType = @DenoisingAutoencoder;
                    opts(i).aeOpt  = struct( 'alpha',1,'momentum',.7,'useAdaDelta',true, 'hActFun', obj.hActFuns{i});
                end
            end                                    
            
            sizes = [obj.inputSize; obj.hiddenSizes(:)];
            dataThis = data;
            
            for i=1:numel(obj.hiddenSizes)
                
                % set options of autoencoder
                opts(i).aeOpt.visibleSize = sizes(i);
                opts(i).aeOpt.hiddenSize  = sizes(i+1);
                opts(i).aeOpt.x           = dataThis;
                
                % initialize autoencoder
                AE = opts(i).aeType(opts(i).aeOpt);

                % train autoencoder
                AE.train_model;    
                
                % encode current data as an input to the next autoencoder
                dataThis = AE.encodeX(dataThis);
                
                % acquire encoding weights to initialize neural net
                paramsThis  = AE.get_parameters_as_stack();
                params(i).W = paramsThis{1}.w;
                params(i).b = paramsThis{1}.b;
                
                % transpose W if necessary - TODO : this should be fixed
                params(i).W = params(i).W';
                
                % further set layers 
                obj.layers{i}.w = params(i).W';
                obj.layers{i}.b = params(i).b;
                
                clear('paramsThis','AE');
            end            
                
%             save('params.mat','params');
            
            % set neural net weights to the autoencoder weights
            obj.nnTheta(1:end-(obj.hiddenSizes(end)*obj.outputSize+obj.outputSize)) = ...
                cell2mat(arrayfun(@(x)[x.W(:);x.b(:)], params, 'UniformOutput',false)');
            
        end
            
        
        %==================================================================
        % Maps the input to the penultimate (the layer before top) layer 
        % and returns the penultimate layer activations
        %==================================================================
        function act = get_preActivation(obj, data)
            try
                act = obj.feedForwardNN(data, true);
            catch err
                act = [];
               fprintf(2,'%s\n',err.getReport); 
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
                
                % get cost and gradient for training data
                [cost, grad] = obj.neuralNetCost( obj.nnTheta, ...
                        obj.trainData,...
                        obj.trainLabels,...
                        obj.lambdaL2,obj.lambdaL1);
                
                % Check cost function and derivative calculations
                % for the sparse autoencoder.  
                numgrad = obj.computeNumericalGradient( @(x)obj.neuralNetCost(x, obj.trainData,obj.trainLabels,...
                                                                obj.lambdaL2,obj.lambdaL1), obj.nnTheta);

                diff = norm(numgrad-grad)/norm(numgrad+grad);
                                                              
                % Use this to visually compare the gradients side by side
                % Compare numerically computed gradients with the ones obtained from backpropagation
                if ~obj.silent
                    disp([numgrad grad]); 
                    disp(diff); % Should be small. 
                end                                
            end
            
        end
        
    end
    
    methods(Hidden)
        
        
        %==================================================================
        % Train model using regime1 - pass (nIters*(nSamples/batchSize))
        % times over the given dataset
        %==================================================================
        function [theta] = train_model_regime1(obj)
            
            % get helpers
            nSamples = size(obj.trainData, 2);
                        
            % optimization
            opt.theta = obj.nnTheta;
            opt.curr_speed = opt.theta * 0;
            if obj.useAdaDelta
                opt.expectedGrads = zeros(size(opt.theta)); % adadelta : E[g^2]_0
                opt.expectedDelta = zeros(size(opt.theta)); % adadelta : E[\delta x^2]_0
            end               
            
            % defaults - for early stopping options and others
            trCosts     = [];
            cvCosts     = [];
            cvCost      = Inf;
            bestTheta   = opt.theta;
            bestCVerror = inf;                     
            iter        = 0;
            
            %--------------------------------------------------------------
            % conduct optimization using mini-batch gradient descent, in
            % each iteration a small chunk of training data (batchSize) is
            % used for model adjustments, note that if iteration number is
            % big enough, we end up wrapping around the data set meaning
            % that we actualy passed to another epoch. If you want to
            % control number of epochs through the dataset, use regime2.
            for ii = 1:obj.nIters
                
                % for adadelta
                iter = iter + 1;
                
                % index calculation for current batch
                batchStartIdx = mod((ii-1) * obj.batchSize, nSamples)+1;
                batchEndIdx   = batchStartIdx + obj.batchSize - 1;
                
                % get cost and gradient for training data
                [trCostThis, gradient] = obj.neuralNetCost( opt.theta, ...
                    obj.trainData(:, batchStartIdx:batchEndIdx),...
                    obj.trainLabels(:, batchStartIdx:batchEndIdx),...
                    obj.lambdaL2,obj.lambdaL1);
                
                % update weights with momentum and lrate options
                opt = updateWeights(obj, opt, gradient, iter);
                
                % get cost cross validation data
                if obj.isCV
                    [cvCost] = obj.neuralNetCost( opt.theta, ...
                        obj.cvData, obj.cvLabels,obj.lambdaL2,obj.lambdaL1);
                end
                
                % keep record of costs
                trCosts = [trCosts, obj.neuralNetCost( opt.theta, ...
                    obj.trainData, obj.trainLabels, obj.lambdaL2,obj.lambdaL1)];
                cvCosts = [cvCosts, cvCost];
                
                % apply early stopping if applicable
                if obj.isEarlyStopping && cvCosts(end) < bestCVerror
                    bestTheta   = opt.theta;
                    bestCVerror = cvCosts(end);
                end
                
                if ~obj.silent &&  mod(ii, round(obj.nIters/10)) == 0
                    fprintf('Iter:[%d/%d] - Training error:[%f] - Validation error:[%f] - NormSpeed:[%f]\n',...
                        ii, obj.nIters, trCosts(end), cvCosts(end),norm(opt.curr_speed));
                end
                
            end
            
            if obj.isEarlyStopping,  opt.theta = bestTheta;  end
            
            theta = opt.theta;
            obj.trErrors = trCosts;
            obj.cvErrors = cvCosts;
            
        end
        
        
        %==================================================================
        % Train model using regime2 - pass nEpochs times over the dataset
        %==================================================================
        function [theta] = train_model_regime2(obj)
            
            % get helpers
            nSamples = size(obj.trainData, 2);
            if obj.batchSize > nSamples
                obj.batchSize = nSamples;
            end
            nBatches = nSamples / obj.batchSize;
            
            % optimization
            opt.theta = obj.nnTheta;
            opt.curr_speed = opt.theta * 0;
            if obj.useAdaDelta
                opt.expectedGrads = zeros(size(opt.theta)); % adadelta : E[g^2]_0
                opt.expectedDelta = zeros(size(opt.theta)); % adadelta : E[\delta x^2]_0
            end
                                    
            % defaults - for early stopping options and others
            trCosts     = [];
            cvCosts     = [];
            cvCost      = Inf;
            bestTheta   = opt.theta;
            bestCVerror = inf;
            iter        = 0;
            
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
                    [trCostThis, gradient] = obj.neuralNetCost( opt.theta, ...
                        obj.trainData(:, batchIdx),...
                        obj.trainLabels(:, batchIdx),...
                        obj.lambdaL2,obj.lambdaL1);
                                 
                    % update weights with momentum and lrate options
                    opt = updateWeights(obj, opt, gradient, iter);
                    
                    % get cost cross validation data
                    if obj.isCV
                        [cvCost] = obj.neuralNetCost( opt.theta, ...
                            obj.cvData, obj.cvLabels,obj.lambdaL2,obj.lambdaL1);
                    end
                    
                    % keep record of costs
                    trCosts = [trCosts, obj.neuralNetCost( opt.theta, ...
                        obj.trainData, obj.trainLabels, obj.lambdaL2,obj.lambdaL1)];
                    cvCosts = [cvCosts, cvCost];
                    
                    % apply early stopping if applicable
                    if obj.isEarlyStopping && cvCosts(end) < bestCVerror
                        bestTheta   = opt.theta;
                        bestCVerror = cvCosts(end);
                    end
                    
                end
                
                if ~obj.silent
                    fprintf('Epoch:[%d/%d] - Training error:[%f] - Validation error:[%f] - NormSpeed:[%f]\n',...
                        ee, obj.nEpochs, trCosts(end), cvCosts(end),norm(opt.curr_speed));
                end
                
            end
            if obj.isEarlyStopping,  opt.theta = bestTheta;  end
            
            theta = opt.theta;            
            obj.trErrors = trCosts;
            obj.cvErrors = cvCosts;
            
        end
        
        
        %==================================================================
        % Train model using regime3 - use given minimization function
        % handle for optimization. Arguments of the minimization function
        % should be passed beforehand.
        %==================================================================
        function [theta] = train_model_regime3(obj)
            [theta, cost] = obj.minfunc( @(p) obj.neuralNetCost(p, ...
                                              obj.trainData, obj.trainLabels, obj.lambdaL2, obj.lambdaL1), ...
                                        obj.nnTheta); % options must be passed to function handle as the 3rd parameter                                   
        end
        
        
        %==================================================================
        % Train model using regime3 - pass nEpochs times over the dataset
        %==================================================================
        function [theta] = train_model_regime4(obj)
            
            % get helpers
            nSamples = size(obj.trainData, 2);
            if obj.batchSize > nSamples
                obj.batchSize = nSamples;
            end
            nBatches = floor(nSamples / obj.batchSize);
            
            % optimization
            opt.theta = obj.nnTheta;
            opt.curr_speed = opt.theta * 0;
            
            % defaults - for early stopping options and others
            trCosts     = [];
            cvCosts     = [];
            cvCost      = Inf;
            iter        = 0;
            
            %--------------------------------------------------------------
            % conduct optimization using mini-batch SGD
            for ee = 1:obj.nEpochs
                
                % shuffle dataset
                sampleIdx = randperm(nSamples);
                trCostThisEpoch = 0;
                
                % sweep over batches
                for ii = 1:nBatches
                    
                    % for adadelta
                    iter = iter + 1;
                    
                    % index calculation for current batch
                    batchIdx = sampleIdx((ii - 1) * obj.batchSize + 1 : ii * obj.batchSize);
                    
                    [opt.theta, trCost] = obj.minfunc( @(p) obj.neuralNetCost(p, ...
                        obj.trainData(:, batchIdx), obj.trainLabels(:, batchIdx), obj.lambdaL2, obj.lambdaL1), ...
                        opt.theta); % options must be passed to function handle as the 3rd parameter                                     
                    
                    % this accumulates the average cost in this epoch
                    trCostThisEpoch = trCostThisEpoch + (trCost - trCostThisEpoch) / ii;
                                                            
                    trCosts = [trCosts, trCost];
                    cvCosts = [cvCosts, cvCost];
                                                            
                end
                
                if ~obj.silent
                    fprintf('Epoch:[%d/%d] - Training error:[%f] - Validation error:[%f] - NormSpeed:[%f] - Avg TrErr:[%f]\n',...
                        ee, obj.nEpochs, trCosts(end), cvCosts(end),norm(opt.curr_speed),trCostThisEpoch);
                end
                
            end            
            
            theta = opt.theta;
            obj.trErrors = trCosts;
            obj.cvErrors = cvCosts;
            
        end
        
        
        %==================================================================
        % Train model using regime2 - pass nEpochs times over the dataset
        %==================================================================
        function [theta] = train_model_onGPU(obj)
            
            % get helpers
            nSamples = size(obj.trainData, 2);
            if obj.batchSize > nSamples
                obj.batchSize = nSamples;
            end
            nBatches = nSamples / obj.batchSize;
            
            % optimization
            for i=1:numel(obj.layers)
                opt.curr_speed{i}.w = gpuArray(zeros(size(obj.layers{i}.w)));
                opt.curr_speed{i}.b = gpuArray(zeros(size(obj.layers{i}.b)));
                if obj.useAdaDelta
                    opt.expectedGrads{i}.w = gpuArray(zeros(size(obj.layers{i}.w))); % adadelta : E[g^2]_0
                    opt.expectedGrads{i}.b = gpuArray(zeros(size(obj.layers{i}.b))); % adadelta : E[g^2]_0
                    opt.expectedDelta{i}.w = gpuArray(zeros(size(obj.layers{i}.w))); % adadelta : E[\delta x^2]_0                    
                    opt.expectedDelta{i}.b = gpuArray(zeros(size(obj.layers{i}.b))); % adadelta : E[\delta x^2]_0                    
                end
            end                        
            
            % defaults - for early stopping options and others
            trCosts     = [];
            cvCosts     = [];
            cvCost      = Inf;
            bestTheta   = obj.layers;
            bestCVerror = inf;
            iter        = 0;
            
            % move training data and labels to gpu
            trainData_g   = gpuArray(obj.trainData);
            trainLabels_g = gpuArray(obj.trainLabels);
            if obj.isCV
                cvData_g   = gpuArray(obj.cvData);
                cvLabels_g = gpuArray(obj.cvLabels);
            end
                
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
                    [trCostThis, gradient] = obj.neuralNetCostGPU( obj.layers, obj.grads,...
                        trainData_g(:, batchIdx),...
                        trainLabels_g(:, batchIdx),...
                        obj.lambdaL2);%, obj.lambdaL1);
                                 
                    % update weights with momentum and lrate options
                    opt = updateWeightsGPU(obj, opt, gradient, iter);
                    
                    % get cost cross validation data
                    if obj.isCV
                        [cvCost] = obj.neuralNetCostGPU( obj.layers, obj.grads,...
                            cvData_g, cvLabels_g, obj.lambdaL2);%, obj.lambdaL1);
                    end
                    
                    % keep record of costs
                    trCosts = [trCosts, obj.neuralNetCostGPU( obj.layers, obj.grads,...
                        trainData_g, trainLabels_g, obj.lambdaL2)];%, obj.lambdaL1)];
                    cvCosts = [cvCosts, cvCost];
                    
                    % apply early stopping if applicable
                    if obj.isEarlyStopping && cvCosts(end) < bestCVerror
                        bestTheta   = obj.layers;
                        bestCVerror = cvCosts(end);
                    end
                    
                end
                
                if ~obj.silent
                    fprintf('Epoch:[%d/%d] - Training error:[%f] - Validation error:[%f] - NormSpeed:[%f]\n',...
                        ee, obj.nEpochs, trCosts(end), cvCosts(end),Inf);
%                         ee, obj.nEpochs, trCosts(end), cvCosts(end),norm(opt.curr_speed));
                end
                
            end
            if obj.isEarlyStopping,  obj.layers = bestTheta;  end
            
            % convert layers to params            
            theta = gather(obj.stack2params(obj.layers,true));
            obj.trErrors = trCosts;
            obj.cvErrors = cvCosts;
            
        end
        
        
        %==================================================================
        %   Updates weights of the model. Momentum or adadelta is employed.
        %==================================================================
        function opt = updateWeights(obj, opt, grad, iter)
                       
            theta = opt.theta;
            
            if obj.useAdaDelta
                
                % fancy way of setting decay_rate to zero if iter == 1
                decay_rate = ((iter-1) * obj.adaDeltaRho / (iter-1+eps));
                tiny = 1e-6;
                
                % first apply momentum on weights
                opt.curr_speed = ((1-obj.momentum) * grad) + (obj.momentum * opt.curr_speed);
                                
                % accumulate gradients
                opt.expectedGrads = (decay_rate * opt.expectedGrads) +((1-decay_rate) * opt.curr_speed.^2);
                
                % compute update
                delta = -opt.curr_speed .* (sqrt(opt.expectedDelta + tiny) ./ sqrt(opt.expectedGrads + tiny));
                
                % accumulate updates
                opt.expectedDelta = (decay_rate * opt.expectedDelta) + ((1-decay_rate) * delta.^2);
                
                % apply update
                theta = theta + delta;
                
            else
                
                % calculate momentum speed and update weights
                opt.curr_speed = opt.curr_speed * obj.momentum - grad;
                theta = theta + opt.curr_speed * obj.alpha;
                
            end
            
            % actual update here
            opt.theta = theta;
            
        end
        
        
        %==================================================================
        %   Updates weights of the model. Momentum or adadelta is employed.
        %==================================================================
        function opt = updateWeightsGPU(obj, opt, grads, iter)

            for i=1:numel(obj.layers)
            
                if obj.useAdaDelta
                
                    % fancy way of setting decay_rate to zero if iter == 1
                    decay_rate = ((iter-1) * obj.adaDeltaRho / (iter-1+eps));
                    tiny = 1e-6;
                
                    % first apply momentum on weights
                    opt.curr_speed{i}.w = ((1-obj.momentum) * grads{i}.w) + (obj.momentum * opt.curr_speed{i}.w);
                    opt.curr_speed{i}.b = ((1-obj.momentum) * grads{i}.b) + (obj.momentum * opt.curr_speed{i}.b);

                    % accumulate gradients
                    opt.expectedGrads{i}.w = (decay_rate * opt.expectedGrads{i}.w) +((1-decay_rate) * opt.curr_speed{i}.w .^2);
                    opt.expectedGrads{i}.b = (decay_rate * opt.expectedGrads{i}.b) +((1-decay_rate) * opt.curr_speed{i}.b .^2);

                    % compute update
                    deltaW = -opt.curr_speed{i}.w .* (sqrt(opt.expectedDelta{i}.w + tiny) ./ sqrt(opt.expectedGrads{i}.w + tiny));
                    deltab = -opt.curr_speed{i}.b .* (sqrt(opt.expectedDelta{i}.b + tiny) ./ sqrt(opt.expectedGrads{i}.b + tiny));

                    % accumulate updates
                    opt.expectedDelta{i}.w = (decay_rate * opt.expectedDelta{i}.w) + ((1-decay_rate) * deltaW.^2);
                    opt.expectedDelta{i}.b = (decay_rate * opt.expectedDelta{i}.b) + ((1-decay_rate) * deltab.^2);

                    % apply update
                    obj.layers{i}.w = obj.layers{i}.w + deltaW;
                    obj.layers{i}.b = obj.layers{i}.b + deltab;
                    
                else

                    % calculate momentum speed and update weights
                    opt.curr_speed{i}.w = opt.curr_speed{i}.w * obj.momentum - grads{i}.w;
                    opt.curr_speed{i}.b = opt.curr_speed{i}.b * obj.momentum - grads{i}.b;

                    % apply update
                    obj.layers{i}.w = obj.layers{i}.w + opt.curr_speed{i}.w * obj.alpha;
                    obj.layers{i}.b = obj.layers{i}.b + opt.curr_speed{i}.b * obj.alpha;
                end

            end            
            
        end
        
        
        %==================================================================
        % Parameters are need to be converted to a column vector for
        % optimizer and configuration of the network must be saved for
        % converting back stacked structure in advance
        %==================================================================
        function [] = init_layer_params(obj)
            
            % Initialize the layers using the parameters learned
            sizes  = [obj.inputSize, obj.hiddenSizes(:)', obj.outputSize ];
            depth  = numel(sizes)-1;
            obj.layers = cell(depth,1);
            obj.grads  = [];            
            
            % randomly initialize weights
            if obj.useGPU                
                obj.grads = cell(depth,1);               
                for i=2:depth+1
                    if obj.initFanInOut
                        r  = sqrt(6) / sqrt(sizes(i)+sizes(i-1)+1);   % choose weights uniformly from the interval [-r, r]
                        obj.layers{i-1}.w = (gpuArray(rand(sizes(i), sizes(i-1))) - .5) * 2 * 4 * r;
                        obj.grads{i-1}.w  = gpuArray(zeros(sizes(i),sizes(i-1)));
                    else
                        obj.layers{i-1}.w = obj.stdInitW * gpuArray(randn(sizes(i), sizes(i-1)));
                        obj.grads{i-1}.w  = gpuArray(zeros(sizes(i),sizes(i-1)));
                    end
                    obj.layers{i-1}.b = gpuArray(zeros(sizes(i), 1));
                    obj.grads{i-1}.b  = gpuArray(zeros(sizes(i), 1));
                end                               
            else                
                for i=2:depth+1
                    if obj.initFanInOut
                        r  = sqrt(6) / sqrt(obj.nnLayers{i}.nNeuron+ (obj.nnLayers{i-1}.outSize)+1);   % choose weights uniformly from the interval [-r, r]
                        obj.layers{i-1}.w = (rand(obj.nnLayers{i}.nNeuron, obj.nnLayers{i-1}.outSize) - .5) * 2 * 4 * r;
                    else
                        obj.layers{i-1}.w = obj.stdInitW * randn(obj.nnLayers{i}.nNeuron, obj.nnLayers{i-1}.outSize);
                    end
                    obj.layers{i-1}.b = zeros(obj.nnLayers{i}.nNeuron, 1);
                end
            end
            obj.nnTheta = obj.stack2params(obj.layers,true);
        end
        
        
        %==================================================================
        % Initialize hidden attributes of the object for activation 
        % functions and input-hidden-output sizes.
        %==================================================================
        function [] = init_layers(obj)
            
            % set default activation of hidden units as sigmoid            
            obj.hActFuns    = cellfun(@(y)y.act,obj.nnLayers(...
                                            cellfun(@(x)strcmp(x.type,'fc'),obj.nnLayers)...
                                                            ),'UniformOutput',false);
            obj.oActFun     = obj.nnLayers{cellfun(@(x)strcmp(x.type,'output'),obj.nnLayers)}.act;                                                        
            
            
            obj.hiddenSizes = cellfun(@(y)y.nNeuron,obj.nnLayers(...
                                            cellfun(@(x)strcmp(x.type,'fc'),obj.nnLayers)));

            % for maxout layers reduce the size to hiddenSize/poolSize
            maxoutLayerInd = cellfun(@(x)strcmp(x,'maxout'),obj.hActFuns);
            poolingSizes   = cellfun(@(x)x.poolSize,obj.nnLayers(logical([0 maxoutLayerInd 0])));
            if ~isempty(poolingSizes)
                assert(all( (obj.hiddenSizes(maxoutLayerInd)./poolingSizes) == ...
                        round(obj.hiddenSizes(maxoutLayerInd)./poolingSizes)),...
                    'Number of features(neurons/layerSize) should be exactly divisible by poolSize!!')                        
            end
            
            % again for maxout layers arrange the input struct for mex
            maxoutLayerIdx = find(maxoutLayerInd) + 1;
            for i=maxoutLayerIdx
                assert(isfield(obj.nnLayers{i},'poolSize'),...
                    'poolSize must be specified for maxout layers!')                
                if ~isfield(obj.nnLayers{i},'stride')
                    obj.nnLayers{i}.stride = 1;     % stride is 1 by default
                end
                if ~isfield(obj.nnLayers{i},'isRandom')
                    obj.nnLayers{i}.isRandom = 0;   % no random pooling by default
                end
                if ~isfield(obj.nnLayers{i},'isDebug')
                    obj.nnLayers{i}.isDebug = 0;    % no verbose by default
                end
            end
               
            % set outSize for all layers including input/output
            for i=1:numel(obj.nnLayers)
                if isfield(obj.nnLayers{i},'act') && strcmp(obj.nnLayers{i}.act,'maxout')
                    obj.nnLayers{i}.outSize = obj.nnLayers{i}.nNeuron / obj.nnLayers{i}.poolSize; 
                else
                    obj.nnLayers{i}.outSize = obj.nnLayers{i}.nNeuron;
                end
                
            end
            
            obj.outputSize  = obj.nnLayers{end}.outSize;
                
            obj.inputSize   = cellfun(@(y)y.nNeuron,obj.nnLayers(...
                                            cellfun(@(x)strcmp(x.type,'input'),obj.nnLayers)));
            
            if isempty(obj.oActFun)
                obj.oActFun = 'softmax';
            end
                                        
            if isempty(obj.inputSize)
                obj.inputSize = size(obj.trainData,1);
            end
                                                    
        end
        
        
        %==================================================================
        % checks model parameters for neural net for stochastic gradient
        % descent with epochs
        %==================================================================
        function isValid = isModelValid(obj)
            isValid = false;
            
            if isempty(obj.trainData),      return; end
            if isempty(obj.trainLabels),    return; end
            if isempty(obj.lambdaL2),       return; end
            if isempty(obj.lambdaL1),       return; end
            if isempty(obj.nnTheta),        return; end            
            if isempty(obj.trainingRegime), return; end                                    
            if isempty(obj.inputSize),      return; end
            if isempty(obj.hiddenSizes),    return; end
            if isempty(obj.outputSize),     return; end
            if isempty(obj.batchSize),      return; end
            if isempty(obj.stdInitW),       return; end
            if obj.trainingRegime == 2 % 'epochs'
                if isempty(obj.nEpochs),    return; end
            elseif obj.trainingRegime == 1 % 'nIters'
                if isempty(obj.nIters),     return; end
            elseif obj.trainingRegime == 3 % 'minfunc'
                if isempty(obj.minfunc),    return; end
            end            
            
            isValid = true;
        end
        
    end
    
end

