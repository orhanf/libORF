classdef NaiveBayesGM < handle
%==========================================================================
%
% This class is for naive bayes implementation of
%   - (G)aussian naive bayes for real valued features X and discrete 
%       class labels Y (eg. fMRI sentiment classification)
%   - (M)ultinomial naive bayes for discrete features X and discrete 
%       class labels Y (eg. document classification)    
%   - (K)ernel smoothed (density estimated) naive bayes for real valued 
%       features X and discrete class labels Y (eg. fMRI sentiment 
%       classification)
%
%
% orhanf - (c) 2013 Orhan FIRAT <orhan.firat@ceng.metu.edu.tr>
%==========================================================================
    
    properties
        % model parameters
        nClasses;       % parameter vector (vector) for gradient descent
        nFeatures;      % regularization (weight decay) parameter
        nSamples;       % number of samples in training phase
        train_matrix;   % training data matrix < nSamples , nFeatures>
        train_labels;   % training labels
        p_of_y;         % estimates of the probability of labels
        p_of_x_given_c; % estimates of the probability of features given labels
        classNames;     % vector for arbitrary class numbering
        isGaussianNB;   % use gaussian naive bayes for continuous features
        isKernelNB;     % use kernel naive bates for continuous features
        mu_of_x;        % expected of mean value of X
        sigma_of_x;     % standard deviation of X
        silent;         % output visibility option
    end
    
    
    methods
        
        %==================================================================
        % constructor
        %==================================================================
        function obj = NaiveBayesGM(options)
            
            obj.isGaussianNB = true;  % by default libORF applies gaussian naive bayes
            obj.isKernelNB   = false;
            obj.silent       = true;  % by default silent
            
            if nargin>0 && isstruct(options)                
                if isfield(options,'train_matrix'),   obj.train_matrix = options.train_matrix;     end
                if isfield(options,'train_labels'),   obj.train_labels = options.train_labels;     end
                if isfield(options,'p_of_y'),         obj.p_of_y = options.p_of_y;                 end
                if isfield(options,'p_of_x_given_c'), obj.p_of_x_given_c = options.p_of_x_given_c; end
                if isfield(options,'mu_of_x'),        obj.mu_of_x = options.mu_of_x;               end
                if isfield(options,'sigma_of_x'),     obj.sigma_of_x = options.sigma_of_x;         end
                if isfield(options,'nClasses'),       obj.nClasses = options.nClasses;             end
                if isfield(options,'nFeatures'),      obj.nFeatures = options.nFeatures;           end
                if isfield(options,'nSamples'),       obj.nSamples = options.nSamples;             end
                if isfield(options,'classNames'),     obj.classNames = options.classNames;         end
                if isfield(options,'isGaussianNB'),   obj.isGaussianNB = options.isGaussianNB;     end                
                if isfield(options,'isKernelNB'),     obj.isKernelNB = options.isKernelNB;         end                                
                if isfield(options,'silent'),          obj.silent = options.silent;                end
                
                if obj.isKernelNB
                    obj.isGaussianNB = false;
                end                     
            end
        end
        
        
        %==================================================================
        % Fit a model using training data
        %==================================================================
        function train_model(obj)
            
            if isModelValid(obj)
                
                try
                    
                    % compute helpers and update object properties
                    if isempty(obj.classNames), obj.classNames = unique(obj.train_labels); end
                    if isempty(obj.nClasses),   obj.nClasses   = numel(obj.classNames);    end
                    if isempty(obj.nSamples),   obj.nSamples   = numel(obj.train_labels);  end
                    if isempty(obj.nFeatures),  obj.nFeatures  = size(obj.train_matrix,2); end
                    
                    % calculate class-priors
                    obj.p_of_y = zeros(obj.nClasses,1);
                    for i=1:obj.nClasses
                        obj.p_of_y(i) = sum(obj.train_labels == obj.classNames(i)) / obj.nSamples;
                    end
                    
                    % calculate class-conditionals
                    if obj.isGaussianNB % continuous features X
                        
                        obj.mu_of_x    = zeros(obj.nClasses, obj.nFeatures);
                        obj.sigma_of_x = zeros(obj.nClasses, obj.nFeatures);
                        
                        for i=1:obj.nClasses
                            obj.mu_of_x(i,:)    = mean(obj.train_matrix(obj.train_labels == obj.classNames(i),:)); 
                            obj.sigma_of_x(i,:) = std(obj.train_matrix(obj.train_labels == obj.classNames(i),:)); 
                        end    
                        
                    elseif obj.isKernelNB % continuous features X
                        
                        % this step is skipped in order to save memory and
                        % time, thus implemented in prediction phase
                        if ~obj.silent
                            disp('Densities will be estimated during prediction!');
                        end
                        
                    else % discrete features X
                        
                        for i=1:obj.nClasses
                            currClassData = obj.train_matrix(obj.train_labels == obj.classNames(i),:);
                            denominator   = sum(currClassData(:)) + obj.nFeatures;
                            obj.p_of_x_given_c (i,:) =  (sum(currClassData) + 1 ) / denominator;
                        end
                        
                    end
                    
                catch err
                    disp(['Error in the training phase:' err.getReport]);
                end
            end
        end
        
        
        %==================================================================
        % Predict labels of test matrix using trained Naive Bayes model
        %==================================================================
        function predicted_labels = predict_labels(obj, test_matrix)
            
            if isTestValid(obj,test_matrix)
                
                % calculate helpers
                nSamplesTE = size(test_matrix,1);
                
                % Calculate log p(x|y=1) + log p(y=1)
                log_probs = zeros(nSamplesTE,obj.nClasses);
                
                if obj.isGaussianNB
                    for i=1:obj.nClasses
                        exp_term  = exp(-0.5 .* (bsxfun(@rdivide,bsxfun(@minus,test_matrix,obj.mu_of_x(i,:)),obj.sigma_of_x(i,:)) .^ 2 ));
                        norm_term = sqrt(2*pi .* (obj.sigma_of_x(i,:) .^ 2) );
                        log_probs(:,i) = sum(log(bsxfun(@rdivide,exp_term,norm_term)),2) + log(obj.p_of_y(i));
                    end
                elseif obj.isKernelNB
                    log_condPdf = get_kernel_log_condPDF(obj,test_matrix);
                    log_probs   = squeeze(sum(log_condPdf,2))+ repmat(log(obj.p_of_y)', size(log_condPdf,1) ,1 );
                else
                    for i=1:obj.nClasses
                        log_probs(:,i) = test_matrix*log(obj.p_of_x_given_c(i,:))' + log(obj.p_of_y(i));
                    end
                end
                
                [dummy, max_prob_label ]= max(log_probs,[],2);
                
                predicted_labels = zeros(nSamplesTE,1);
                
                for i=1:obj.nClasses
                    predicted_labels(max_prob_label == i) = obj.classNames(i);
                end                
                
            end
        end
        
        
        
    end
    
    
    methods(Hidden)
        
        %==================================================================
        % checks model parameters for naive bayes
        %==================================================================
        function isValid = isModelValid(obj)
            isValid = false;
            
            if isempty(obj.train_labels), return; end
            if isempty(obj.train_matrix), return; end
            if length(obj.train_labels) ~= size(obj.train_matrix,1), return; end
            
            isValid = true;
        end
        
        
        %==================================================================
        % checks test data with model parameters for naive bayes
        %==================================================================
        function isValid = isTestValid(obj,test_matrix)
            isValid = false;
            
            if isempty(test_matrix),   return; end
            if isempty(obj.p_of_y),    return; end
            if isempty(obj.nClasses),  return; end
            if isempty(obj.nFeatures), return; end
            
            if size(test_matrix,2) ~= obj.nFeatures, return; end
            if length(obj.p_of_y)  ~= obj.nClasses,  return; end
            
            if obj.isGaussianNB
                
                if isempty(obj.mu_of_x),    return; end
                if isempty(obj.sigma_of_x), return; end
                
                if size(obj.mu_of_x,1)    ~= obj.nClasses,  return; end
                if size(obj.mu_of_x,2)    ~= obj.nFeatures, return; end
                if size(obj.sigma_of_x,1) ~= obj.nClasses,  return; end
                if size(obj.sigma_of_x,2) ~= obj.nFeatures, return; end
                
            elseif obj.isKernelNB
                % all set
            else
                
                if isempty(obj.p_of_x_given_c),    return; end
                
                if size(obj.p_of_x_given_c,1) ~= obj.nClasses,  return; end
                if size(obj.p_of_x_given_c,2) ~= obj.nFeatures, return; end
                
            end
            
            isValid = true;
        end
        
        
        %==================================================================
        % computes log conditional probability density function for each
        % feature and for each class using kernel density estimation and
        % returns conditional propabilities of each test sample for each
        % class. Kernel width is estimated automatically and type of kernel
        % smoother is selected as normal.
        %==================================================================
        function log_condPdf = get_kernel_log_condPDF(obj, test_matrix)
            
            nTestSamples = size(test_matrix,1);
            log_condPdf  = zeros(nTestSamples, obj.nFeatures, obj.nClasses);                        
            
            for i=1:nTestSamples
                for j=1:obj.nFeatures
                    for k=1:obj.nClasses
                        log_condPdf(i,j,k) = ksdensity(obj.train_matrix(obj.train_labels == k,j), test_matrix(i,j)); 
                    end
                end
            end
            log_condPdf = log(log_condPdf);
        end
    end
    
    
    
end

