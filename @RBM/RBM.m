classdef RBM < handle
%==========================================================================
%   This class implements Restricted Boltzman Machine.
%
% orhanf - (c) 2013 Orhan FIRAT <orhan.firat@ceng.metu.edu.tr>
%==========================================================================
    
    properties
        % model parameters
        rbm_W;       % parameter matrix for weights
        rbm_b;       % parameter vector for biases from visible to hidden units
        rbm_c;       % parameter vector for biases from hidden to visible units
        visibleSize; % number of input units
        hiddenSize;  % number of hidden units
        x;           % input training data <nFeatures,nSamples> (samples are at columns)
        y;           % input target labels
        top;         % top layer activations (for DBN)
        
        % hyper-parameters
        learningRate;  % Learning rate for optimization
        weightDecay;   % Weight decay parameter (for regularization)
        nEpochs;       % Number of epochs (full sweeps through dataset)
        momentum;      % Momentum parameter for optimization btwn [0,1]
        batchSize;     % Batch size to split dataset
        reduceNoise;   % Reducte sampling noise by taking probabilities rather states
        useCondProb;   % Use conditional probabilities for improved training
        
        minfunc;       % Minimization function handle 
        randSource;    % Randomness source for deterministic outputs
        addBias;       % add bias unit for hidden and visible units
        verbose;       % display cost in each iteration etc.
        labelFitMode;  % RBM is to be used for fitting labels
        statesMode;    % State classes for visible and binary units,
        % 'BB' : Binary visible and hidden units
        % 'GB' : Real visible and binary hidden units
        % 'BG' : Binary visible and gaussian hidden units
        
        J;             % cost function values (vector)
    end
    
    
    methods
        
        %==================================================================
        % constructor
        %==================================================================
        function obj = RBM(options)
            
            % defaults
            obj.learningRate = 0.001;
            obj.weightDecay  = 0;
            obj.nEpochs      = 10;
            obj.momentum     = 0.9;
            obj.batchSize    = 100;
            obj.addBias      = true;
            obj.verbose      = true;
            obj.labelFitMode = false;
            obj.statesMode   = 'BB';
            obj.reduceNoise  = false;
            obj.useCondProb  = false;
            
            if nargin>0 && isstruct(options)
                if isfield(options,'rbm_W'),        obj.rbm_W        = options.rbm_W;       end
                if isfield(options,'rbm_b'),        obj.rbm_b        = options.rbm_b;       end
                if isfield(options,'rbm_c'),        obj.rbm_c        = options.rbm_c;       end
                if isfield(options,'visibleSize'),  obj.visibleSize  = options.visibleSize; end
                if isfield(options,'hiddenSize'),   obj.hiddenSize   = options.hiddenSize;  end
                if isfield(options,'x'),            obj.x            = options.x;           end
                if isfield(options,'y'),            obj.y            = options.y;           end
                if isfield(options,'top'),          obj.top          = options.top;         end
                if isfield(options,'learningRate'), obj.learningRate = options.learningRate;end
                if isfield(options,'weightDecay'),  obj.weightDecay  = options.weightDecay; end
                if isfield(options,'nEpochs'),      obj.nEpochs      = options.nEpochs;     end
                if isfield(options,'momentum'),     obj.momentum     = options.momentum;    end
                if isfield(options,'batchSize'),    obj.batchSize    = options.batchSize;   end
                if isfield(options,'momentum'),     obj.momentum     = options.momentum;    end
                if isfield(options,'addBias'),      obj.addBias      = options.addBias;     end
                if isfield(options,'verbose'),      obj.verbose      = options.verbose;     end
                if isfield(options,'labelFitMode'), obj.labelFitMode = options.labelFitMode;end
                if isfield(options,'statesMode'),   obj.statesMode   = options.statesMode;  end
                if isfield(options,'reduceNoise'),  obj.reduceNoise  = options.reduceNoise; end
                if isfield(options,'J'),            obj.J            = options.J;           end
                if isfield(options,'minfunc'),      obj.minfunc      = options.minfunc;     end
                if isfield(options,'randSource'),   obj.randSource   = options.randSource;  end
                if isfield(options,'useCondProb'),  obj.useCondProb  = options.useCondProb; end                
            end
        end
        
        
        %==================================================================
        % Train model using contrastive divergence
        %==================================================================
        function [model, costs] = train_model(obj)
            
            model = 0;
            costs = 0;
            
            if isModelValid(obj)
                try
                    if obj.labelFitMode % using RBM for classification
                        switch obj.statesMode
                            case 'BB'
                                [model, costs] = obj.train_FitBB;
                            case 'RB'
                                [model, costs] = obj.train_FitRB;
                            case 'BG'
                                [model, costs] = obj.train_FitBG;
                            case 'GB'
                                [model, costs] = obj.train_FitGB;
                        end
                    else  % using RBM for unsupervised feature learning
                        switch obj.statesMode
                            case 'BB'
                                [model, costs] = obj.train_BB;
                            case 'RB'
                                [model, costs] = obj.train_RB;
                            case 'BG'
                                [model, costs] = obj.train_BG;
                            case 'GB'
                                [model, costs] = obj.train_GB;
                        end
                    end
                    
                    obj.rbm_W = model.W;
                    obj.rbm_b = model.b;
                    obj.rbm_c = model.c;
                    
                catch err
                    disp(['RBM Minimization function terminated with error:' err.getReport]);
                end
            end
        end
        
        
        %==================================================================
        % Samples hidden states according to given visible data
        %==================================================================
        function [hStates, hProbs, E] = push_visible_data(obj, data)
            
            hStates = [];
            hProbs  = [];
            E = [];
            
            if ~(isempty(obj.rbm_W) || isempty(obj.rbm_b) || isempty(obj.rbm_c))
                [hStates,hProbs,E] = obj.vis_to_hid(data);
            end
        end
        
        
        %==================================================================
        % Samples visible states according to given hidden data
        %==================================================================
        function [vStates, vProbs, E] = push_hidden_data(obj, data)
            
            vStates = [];
            vProbs  = [];
            E = [];
            
            if ~(isempty(obj.rbm_W) || isempty(obj.rbm_b) || isempty(obj.rbm_c))
                [vStates,vProbs,E] = obj.hid_to_vis(data);
            end
        end
        
        
        %==================================================================
        % Samples visible states according to trained model, randomly
        % initialize visible units and follow alternating Gibbs sampling.
        % Samples are collected at consecutive steps of Gibbs sampling.
        % Because of the initialization regime (once at the beginning)
        % samples are correlated.
        %==================================================================
        function [vStates, vProbs, E] = sample_data(obj, nSamples)
            
            vStates = [];
            vProbs  = [];
            E = [];
            
            if ~(isempty(obj.rbm_W) || isempty(obj.rbm_b) || isempty(obj.rbm_c))
                [vStates,vProbs,E] = obj.sample_visible_data(nSamples);
            end
        end
        
        
        %==================================================================
        % Returns the model parameters as a struct
        %==================================================================
        function model = getModel(obj)
            model.statesMode = obj.statesMode;
            model.W    = obj.rbm_W;
            model.b    = obj.rbm_b;
            model.c    = obj.rbm_c;
            model.top  = obj.top;
        end
        
        
    end
    
    
    methods(Hidden)
        
        %==================================================================
        % checks model parameters for RBM - TODO : revise
        %==================================================================
        function isValid = isModelValid(obj)
            isValid = false;
            
            if isempty(obj.x),           return; end
            if isempty(obj.hiddenSize),  return; end
            if isempty(obj.visibleSize), return; end
            
            isValid = true;
        end
               
    end
        
end

