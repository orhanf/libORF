classdef EmbeddingNeuralNet < handle
%==========================================================================
%
%   TODO : add comments here
%   TODO : fix addBias issues
%
% orhanf - (c) 2013 Orhan FIRAT <orhan.firat@ceng.metu.edu.tr>
%==========================================================================
    
    properties
        
        embedSize;   % Array for embedding layer sizes
        nEmbeds;     % Number of embedding layers (lateral layers that are sharing weights)
        hiddenSizes; % Array for hidden unit sizes, excluding input, embedding and output layers
        outputSize;  % Number of units in the output layer as it is softmax
        inputSize;   % Number of units in one of the input layers
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
        lambda;      % Weight decay parameter (for L2 regularization)
        stdInitW;    % Standard deviation of the normal distribution which is sampled to get the initial weights
        addBias;     % Add bias to input and hidden layers
        dropOutRatio;% Employ drop-out method for hidden units
        useAdaDelta; % use adaptive delta to estimate learning rate
        adaDeltaRho; % decay rate rho, for adadelta
        oActFun;     % Output layer function, either 'softmax', 'sigmoid' or 'linear' (default 'softmax')
        hActFuns;    % Hidden layer activation functions, (cell) array with the same length as hiddenSizes
                     %   Activation functions should a cell array composed of elements either one of
                     %          'sigmoid', 'tanh', 'relu' or 'linear'
                     %   or it can be a vector containing
                     %          0-'sigmoid', 1-'tanh', 2-'relu' or 3-'linear'
        
        nnTheta;     % Initial model parameters with random initialization
        nnOptTheta;  % Optimum model parameters after training
        
        silent;          % Display cost in each iteration etc (verbose)
        isEarlyStopping; % Apply early stopping criterion with best cv error
        keepRecords;     % Keeping records per iteration (1), per epoch (2) or not (0)
        trainingRegime;  % 1='epochs':SGD, 2='external':BGD
        minfunc;         % External minimization function handle
        initFanInOut;    % Initialize weights wrt fan-in and fan-out
        vocabulary;      % Word dictionary
    end
    
    properties(Hidden)
        layers;         % Data structure for neural net config and parameters
        netconfig;      % Data structure for neural net config
        isCV;           % Flag for cross validation set is supplied        
    end
    
    methods
        
        %==================================================================
        % constructor
        %==================================================================
        function obj = EmbeddingNeuralNet(options)
            
            % defaults
            obj.nEpochs      = 100;
            obj.nIters       = 1000;
            obj.batchSize    = 100;
            obj.alpha        = 0.1;
            obj.momentum     = 0.9;
            obj.lambda       = 0.01;
            obj.stdInitW     = 0.01;
            obj.addBias      = true;
            obj.dropOutRatio = 0;
            obj.useAdaDelta  = false;
            obj.adaDeltaRho  = 0.95;
            obj.oActFun      = 'softmax';
            
            % defaults
            obj.trainingRegime  = 1;
            obj.isEarlyStopping = false;
            obj.keepRecords     = 2;
            obj.silent          = false;
            obj.isCV            = false;
            obj.initFanInOut    = false;
            
            if nargin>0 && isstruct(options)
                
                if isfield(options,'trainData'),       obj.trainData        = options.trainData;        end
                if isfield(options,'trainLabels'),     obj.trainLabels      = options.trainLabels;      end
                if isfield(options,'cvData'),          obj.cvData           = options.cvData;           end
                if isfield(options,'cvLabels'),        obj.cvLabels         = options.cvLabels;         end
                if isfield(options,'inputSize'),       obj.inputSize        = options.inputSize;        end
                if isfield(options,'embedSize'),       obj.embedSize        = options.embedSize;        end
                if isfield(options,'hiddenSizes'),     obj.hiddenSizes      = options.hiddenSizes;      end
                if isfield(options,'outputSize'),      obj.outputSize       = options.outputSize;       end
                if isfield(options,'nnTheta'),         obj.nnTheta          = options.nnTheta;          end
                if isfield(options,'nnOptTheta'),      obj.nnOptTheta       = options.nnOptTheta;       end
                if isfield(options,'batchSize'),       obj.batchSize        = options.batchSize;        end
                if isfield(options,'alpha'),           obj.alpha            = options.alpha;            end
                if isfield(options,'momentum'),        obj.momentum         = options.momentum;         end
                if isfield(options,'lambda'),          obj.lambda           = options.lambda;           end
                if isfield(options,'stdInitW'),        obj.stdInitW         = options.stdInitW;         end
                if isfield(options,'addBias'),         obj.addBias          = options.addBias;          end
                if isfield(options,'dropOutRatio'),    obj.dropOutRatio     = options.dropOutRatio;     end
                if isfield(options,'useAdaDelta'),     obj.useAdaDelta      = options.useAdaDelta;      end
                if isfield(options,'adaDeltaRho'),     obj.adaDeltaRho      = options.adaDeltaRho;      end
                if isfield(options,'oActFun'),         obj.oActFun          = options.oActFun;          end
                if isfield(options,'silent'),          obj.silent           = options.silent;           end
                if isfield(options,'isEarlyStopping'), obj.isEarlyStopping  = options.isEarlyStopping;  end
                if isfield(options,'keepRecords'),     obj.keepRecords      = options.keepRecords;      end
                if isfield(options,'initFanInOut'),    obj.initFanInOut     = options.initFanInOut;     end
                if isfield(options,'vocabulary'),      obj.vocabulary       = options.vocabulary;       end
                if isfield(options,'trainingRegime'),  obj.trainingRegime   = options.trainingRegime;   end
                if isfield(options,'minfunc'),         obj.minfunc          = options.minfunc;          end
                if isfield(options,'nEpochs'),         obj.nEpochs          = options.nEpochs;          end
                
                % Set embedding size
                if isfield(options,'nEmbeds')
                    obj.nEmbeds = options.nEmbeds;
                else
                    obj.nEmbeds = size(options.trainData,1);
                end
                
                % This is crucial for using Adadelta
                if obj.useAdaDelta, obj.initFanInOut = true;    end
                
                % set default activation of hidden units as sigmoid
                if isfield(options,'hActFuns')
                    obj.hActFuns = options.hActFuns;
                else
                    obj.hActFuns = zeros(numel(obj.hiddenSizes),1);
                end
                
                if ~isempty(obj.cvData) && ~isempty(obj.cvLabels),   obj.isCV = true; end
                               
                % Initialize parameter vectors with zeros
                [obj.nnTheta, obj.layers, obj.netconfig] = obj.init_layers;
                
            end
        end
        
        
        %==================================================================
        % Train model given optimization regime
        %==================================================================
        function [theta] = train_model(obj)
            try
                if isModelValid(obj)
                    % first arrange labels, if in vector form, convert it
                    % into a matrix form in 1-of-k encoding
                    if isvector(obj.trainLabels)
                        M = eye(max(obj.trainLabels));
                        obj.trainLabels = M(:,obj.trainLabels);
                    end
                    if obj.isCV && isvector(obj.cvLabels)
                        M = eye(max(obj.cvLabels));
                        obj.cvLabels = M(:,obj.cvLabels);
                    end
                    
                    if obj.trainingRegime == 1          % using 'nEpochs' SGD
                        fprintf('Starting training following regime 1:using nEpochs SGD...\n');
                        [theta] = obj.train_model_regime1;                        
                    elseif obj.trainingRegime == 2      % using minfunc
                        fprintf('Starting training following regime 2:using external optimizer BGD...\n');
                        [theta] = obj.train_model_regime2;
                    elseif obj.trainingRegime == 3      % using minfunc
                        fprintf('Starting training following regime 3:using external optimizer SGD...\n');
                        [theta] = obj.train_model_regime3;
                    else
                        error('Undefined training regime indicator!');
                    end
                    
                    obj.nnOptTheta = theta;
                    
                    if ~obj.silent && obj.trainingRegime ~= 2
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
                
                act = obj.feedForwardENN(data);
                
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
            [obj.nnTheta, obj.layers, obj.netconfig] = obj.init_layers;
        end
        
        
        %==================================================================
        % Maps the input to the penultimate (the layer before top) layer
        % and returns the penultimate layer activations
        %==================================================================
        function act = get_preActivation(obj, data)
            try
                act = obj.feedForwardENN(data, true);
            catch err
                act = [];
                fprintf(2,'%s\n',err.getReport);
            end
        end
        
        
        %==================================================================
        % Returns the first level embedding weight matrix (look-up table)
        %==================================================================
        function embeddingWeights = get_embedding_weights(obj)
            try
                embeddingWeights = reshape(obj.nnOptTheta(1:obj.inputSize*obj.embedSize),...
                    [obj.embedSize, obj.inputSize]);
            catch err
                embeddingWeights = [];
                fprintf(2,'%s\n',err.getReport);
            end
        end
        
        
        %==================================================================
        % Retrieve k-nearest neighbor of a query word in the embedding
        % space using euclidean distance, use after training the model
        %==================================================================
        function [neighWords, distances] = get_closest_words(obj,seedWord,k)
            try
                % retrieve look-up table
                embeddingWeights = obj.get_embedding_weights;
                
                if nargin<3,    k=5;    end
                if ischar(seedWord)
                    if isempty(obj.vocabulary), error('Provide dictionary for string inputs!'); end
                    
                    wordIdx = find(cellfun(@(x)strcmp(x,seedWord),obj.vocabulary),1);
                    if isempty(wordIdx),        error('Query word is not in the dictionary!');  end
                    
                    [distances, indices] = pdist2(embeddingWeights',embeddingWeights(:,wordIdx)',...
                        'euclidean','smallest',k+1);
                    neighWords = obj.vocabulary(indices(2:end))';
                    
                elseif isnumeric(seedWord)
                    
                    [distances, indices] = pdist2(embeddingWeights',embeddingWeights(:,seedWord)',...
                        'euclidean','smallest',k+1);
                    if isempty(obj.vocabulary)
                        neighWords = indices;
                    else
                        neighWords = obj.vocabulary(indices(2:end));
                    end
                else
                    error('Query word should be either a string or an index to dictionary!');
                end
            catch err
                neighWords = [];
                fprintf(2,'%s\n',err.getReport);
            end
        end
        
        
        %==================================================================
        % Retrieve k-nearest neighbor of a query word in the embedding
        % space using euclidean distance, use after training the model
        %==================================================================
        function distance = get_word_distances(obj, word1, word2)
            try
                % retrieve look-up table
                embeddingWeights = obj.get_embedding_weights;
                if isempty(obj.vocabulary) && any([ischar(word1), ischar(word2)]) 
                    error('Provide dictionary for string inputs!'); 
                end
                if ischar(word1), word1Idx = find(cellfun(@(x)strcmp(x,word1),obj.vocabulary),1); end
                if ischar(word2), word2Idx = find(cellfun(@(x)strcmp(x,word2),obj.vocabulary),1); end                                
                if any([isempty(word1Idx), isempty(word2Idx)]), error('Query word is not in the dictionary!'); end
                
                % calculate distance
                distance = pdist2(embeddingWeights(:,word1Idx)',embeddingWeights(:,word2Idx)');
                
            catch err
                distance = [];
                fprintf(2,'%s\n',err.getReport);
            end
        end
        
    end
    
    methods(Hidden)
        
        %==================================================================
        % Train model using regime1 - pass nEpochs times over the dataset
        %==================================================================
        function [theta] = train_model_regime1(obj)
            
            % get helpers
            nSamples = size(obj.trainData, 2);
            if obj.batchSize > nSamples
                obj.batchSize = nSamples;
            end
            nBatches = floor(nSamples / obj.batchSize);
            
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
                    
                    % get cost and gradient for training data
                    [trCostThis, gradient] = obj.embeddingNeuralNetCost( opt.theta, obj.netconfig,...
                        obj.trainData(:, batchIdx),...
                        obj.trainLabels(:, batchIdx),...
                        obj.lambda);
                    
                    % update weights with momentum and lrate options
                    opt = updateWeights(obj, opt, gradient, iter);                    
                    
                    % this accumulates the average cost in this epoch
                    trCostThisEpoch = trCostThisEpoch + (trCostThis - trCostThisEpoch) / ii;
                                                            
                    if obj.keepRecords == 1 % this will be slow if your dataset is big                                                
                        
                        % keep record of costs
                        trCosts = [trCosts, obj.embeddingNeuralNetCost( opt.theta, obj.netconfig,...
                            obj.trainData, obj.trainLabels, obj.lambda)];
                        
                        % get cost cross validation data
                        if obj.isCV
                            [cvCost] = obj.embeddingNeuralNetCost( opt.theta, obj.netconfig,...
                                obj.cvData, obj.cvLabels,obj.lambda);
                        end                        
                        cvCosts = [cvCosts, cvCost];
                        
                        % apply early stopping if applicable
                        if obj.isEarlyStopping && cvCosts(end) < bestCVerror
                            bestTheta   = opt.theta;
                            bestCVerror = cvCosts(end);
                        end
                    elseif obj.keepRecords == 0 
                        trCosts = [trCosts, trCostThis];
                        % get cost cross validation data
                        if obj.isCV
                            [cvCost] = obj.embeddingNeuralNetCost( opt.theta, obj.netconfig,...
                                obj.cvData, obj.cvLabels,obj.lambda);
                        end                        
                        cvCosts = [cvCosts, cvCost];
                    end
                                                            
                end
                
                if obj.keepRecords == 2
                    
                    % get cost for training data
                    trCosts = [trCosts, obj.embeddingNeuralNetCost( opt.theta, obj.netconfig,...
                        obj.trainData, obj.trainLabels, obj.lambda)];
                    
                    % get cost for validation data
                    if obj.isCV
                        cvCosts = [cvCosts, obj.embeddingNeuralNetCost( opt.theta, obj.netconfig,...
                                obj.cvData, obj.cvLabels,obj.lambda)];
                    else
                        cvCosts = [cvCosts, cvCost];
                    end 
                    
                    % apply early stopping if applicable
                    if obj.isEarlyStopping && cvCosts(end) < bestCVerror
                        bestTheta   = opt.theta;
                        bestCVerror = cvCosts(end);
                    end                    
                end
                
                if ~obj.silent
                    fprintf('Epoch:[%d/%d] - Training error:[%f] - Validation error:[%f] - NormSpeed:[%f] - Avg TrErr:[%f]\n',...
                        ee, obj.nEpochs, trCosts(end), cvCosts(end),norm(opt.curr_speed),trCostThisEpoch);
                end
                
            end
            if obj.isEarlyStopping,  opt.theta = bestTheta;  end
            
            theta = opt.theta;
            obj.trErrors = trCosts;
            obj.cvErrors = cvCosts;
            
        end
                      
        
        %==================================================================
        % Train model using regime2 - use given minimization function
        % handle for optimization. Arguments of the minimization function
        % should be passed beforehand.
        %==================================================================
        function [theta] = train_model_regime2(obj)
            [theta, cost] = obj.minfunc( @(p) obj.embeddingNeuralNetCost(p, obj.netconfig, ...
                obj.trainData, obj.trainLabels, obj.lambda), ...
                obj.nnTheta); % options must be passed to function handle as the 3rd parameter
        end                                       
        
        
        %==================================================================
        % Train model using regime3 - pass nEpochs times over the dataset
        %==================================================================
        function [theta] = train_model_regime3(obj)
            
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
                    
                    [opt.theta, trCost] = obj.minfunc( @(p) obj.embeddingNeuralNetCost(p, obj.netconfig, ...
                        obj.trainData(:, batchIdx), obj.trainLabels(:, batchIdx), obj.lambda), ...
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
        % Parameters are need to be converted to a column vector for
        % optimizer and configuration of the network must be saved for
        % converting back stacked structure in advance
        %==================================================================
        function [params, layers, netconfig] = init_layers(obj)
            
            % Initialize the layers using the parameters learned
            sizes  = [obj.inputSize, obj.embedSize*obj.nEmbeds, obj.hiddenSizes(:)', obj.outputSize ];
            depth  = numel(sizes)-1;
            layers = cell(depth,1);
            
            % randomly initialize weights
            for i=2:depth+1
                inSz  = sizes(i-1);
                outSz = sizes(i);

                if i==2, outSz = outSz/obj.nEmbeds;  end
                if obj.initFanInOut
                    r  = sqrt(6) / sqrt(outSz + inSz +1);   % choose weights uniformly from the interval [-r, r]
                    layers{i-1}.w = (rand(outSz, inSz) - .5) * 2 * 4 * r;
                else
                    layers{i-1}.w = obj.stdInitW * randn(outSz, inSz);
                end
                layers{i-1}.b = zeros(outSz, 1);
            end

            save('layers.mat','layers')
            [params, netconfig] = obj.stack2params(layers);
        end
        
        
        %==================================================================
        % checks model parameters for neural net for stochastic gradient
        % descent with epochs
        %==================================================================
        function isValid = isModelValid(obj)
            isValid = false;
            
            if isempty(obj.trainData),      return; end
            if isempty(obj.trainLabels),    return; end
            if isempty(obj.lambda),         return; end
            if isempty(obj.nnTheta),        return; end
            if isempty(obj.trainingRegime), return; end
            if isempty(obj.inputSize),      return; end
            if isempty(obj.hiddenSizes),    return; end
            if isempty(obj.embedSize),      return; end
            if isempty(obj.outputSize),     return; end
            if isempty(obj.batchSize),      return; end
            if isempty(obj.stdInitW),       return; end
            if obj.trainingRegime == 1 && isempty(obj.nEpochs)     % 'epochs'
                error('Specify nEpochs for training regime 1');
            elseif obj.trainingRegime == 2 && isempty(obj.minfunc) % 'external'
                error('Specify a minimizer function handle for training regime 2');
            end
            
            isValid = true;
        end
        
    end
    
end

