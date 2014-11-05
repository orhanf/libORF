classdef PreProcessing
%==========================================================================
%
% Static class for preprocessing, following utilities are provided:
%   - zero mean, unit variance normalization
%   - zero mean normalization (Use this for PCA)
%   - principal component analysis (PCA)
%   - PCA whitening (sphering)
%   - ZCA whitening (zero component analysis)
%   - Regreessing out conditions from data (linear trend and bias removal)
%   - GLM analysis to obtain betamaps and beta coefficients
%   - F-test and corresponding p values for GLM results for multiple
%       regressors
%   - normalize data between zero and one, direct mapping
%
% orhanf - (c) 2012 Orhan FIRAT <orhan.firat@ceng.metu.edu.tr>
%==========================================================================


    methods(Static)

        
        %==================================================================
        % Normalizes given data by subtracting mean of each sample from
        % itself. Each sample is asumend to be in a column, data
        % matrix is <nxm> where n is number of features and m is number of
        % samples. It is suitable using this normalization for patches
        % sampled from images independent of one another.
        %==================================================================
        function data = normalize_zero_mean(data)
            if ~isempty(data)
                meanPatch = mean(data, 2);  
                data = bsxfun(@minus, data, meanPatch);                
            end
        end
        
        
        %==================================================================
        % Normalizes given data by subtracting mean of each sample from
        % each band itself. Each sample is assumed to be in a column, data
        % matrix is <nxm> where n is number of features and m is number of
        % samples. It is suitable using this normalization for patches
        % sampled from images independent of one another.
        %==================================================================
        function data = normalize_zero_mean_eachBand(data,nDim)
            if ~isempty(data)
                len = size(data,1);
                itr = floor(len / nDim);
                
                for i=1:itr
                    startInd = nDim*(i-1)+1;
                    endInd   = startInd + nDim - 1;
                    meanPatch = mean(data(startInd:endInd,:), 2);  
                    data(startInd:endInd,:) = bsxfun(@minus, data(startInd:endInd,:), meanPatch);                
                end
            end
        end
        
     
        %==================================================================
        % Normalizes given data by subtracting mean of each sample from
        % itself and dividing by its standard deviation. Each sample is 
        % asumend to be in a column, data matrix is <nxm> where n is number 
        % of features and m is number of samples. 
        %==================================================================
        function data = normalize_zero_mean_unit_var(data)           
            if ~isempty(data)
                data = bsxfun(@rdivide, ...
                                bsxfun(@minus, data, repmat(mean(data),size(data,1),1)),...
                                repmat(std(data),size(data,1),1));
            end
        end
        
        
        %==================================================================
        % Normalizes given data by subtracting mean of each sample from
        % itself and dividing by its standard deviation. Each sample is 
        % asumend to be in a column, data matrix is <nxm> where n is number 
        % of features and m is number of samples. 
        % Trailing *_safe indicates that with this member function division
        % by zero cases are eliminated (when std==0). If you are sure that
        % none of the samples have zero std then you may think using 
        % 'normalize_zero_mean_unit_var' which may be faster.
        %==================================================================
        function data = normalize_zero_mean_unit_var_safe(data)
            if ~isempty(data)
                std0 = std(data);
                std0(std0==0) = 1; % this does not change the result 
                data = bsxfun(@rdivide, ...
                                bsxfun(@minus, data, repmat(mean(data),size(data,1),1)),...
                                repmat(std0,size(data,1),1));
            end
        end
        
        
        %==================================================================
        % Applies principal component analysis to given data and returns
        % eigenvectors in each column of U , along with corresponding eigen 
        % values in the diagonals of S. This implementation uses singular
        % value decomposition because sigma(covariance matrix) is a
        % symmetric positive semi-definite and it is more numerically
        % reliable to do this using svd function. Data matrix is assumed to
        % be an < n x m > matrix where n is number of features and m is 
        % number of samples. 
        %==================================================================
        function [U,S] = apply_PCA(data)
            if ~isempty(data)
                sigma = data * data' / size(data, 2); % Covariance matrix
                try
                    [U,S,V] = svd(sigma); % Singular value decomposition
                catch err
                    disp(err.getReport);
                    disp('Calculating with eig()...');
                    [U,S] = eig(sigma);
                    U = fliplr(U);
                    S = diag(sort(diag(S),'descend'));
                end
            end
        end
        
        
        %==================================================================
        % Calculates the covariance matrix of given data. Data matrix is 
        % assumed to be an < n x m > matrix where n is number of features 
        % and m is  number of samples. 
        %==================================================================
        function sigma = calculate_covariance_matrix(data)
            sigma = [];
            if ~isempty(data)
                sigma = data * data' / size(data, 2); % Covariance matrix
            end
        end
        
        
        %==================================================================
        % Calculates the covariance matrix of given data. Data matrix is 
        % assumed to be an < n x m > matrix where n is number of features 
        % and m is  number of samples. Same function on GPU.
        %==================================================================
        function sigma_d = calculate_covariance_matrix_onGPU(data_d)
            sigma_d = [];
            if ~isempty(data_d)
                sigma_d = data_d * data_d' / size(data_d, 2); % Covariance matrix                
            end
        end
        
        
        %==================================================================
        % This function reduces the dimension using principal component
        % analysis and speicified 'percentage of variance retained'
        % (retainRatio).
        %==================================================================
        function [xRot xHat]= reduce_dimension_by_retaining(data, retainRatio)
            xHat = [];
            if ~isempty(data)
                if nargin<2 ||isempty(retainRatio)
                    retainRatio = .99; % retain %99 of variation in data by default
                end
                
                [U,S] = PreProcessing.apply_PCA(data); 
                
                lambdas = diag(S);      % eigenvalues
                variancePercentage = cumsum(lambdas) ./ sum(lambdas);   % percentage of variance retained
                idx = find( variancePercentage > retainRatio );
                k   = idx(1);           % number of features that will be reduced to
                                
                % Compute xRot, the projection on to the eigenbasis
                xRot = U(:,1:k)' * data;
                
                %  Following the dimension reduction, invert the PCA transformation to produce 
                %  the matrix xHat, the dimension-reduced data with respect to the original basis.
                xHat = U(:,1:k)  * xRot;                
            end
        end
        
        
        %==================================================================
        % This function reduces the dimension using principal component
        % analysis and speicified number of features to be reduced.
        % (nFeatures). In any error return zeros as dimension reduced
        % matrix xRot, and original matrix as processed data xHat
        %==================================================================
        function [xRot xHat] = reduce_dimension_by_nFeatures(data, nFeatures)
            xHat = [];
            xRot = [];
            if ~isempty(data) && ~(nargin<2 ||isempty(nFeatures))
                try
                    
                    [U,dummy] = PreProcessing.apply_PCA(data); 

                    % Compute xRot, the projection on to the eigenbasis
                    xRot = U(:,1:nFeatures)' * data;

                    %  Following the dimension reduction, invert the PCA transformation to produce 
                    %  the matrix xHat, the dimension-reduced data with respect to the original basis.
                    xHat = U(:,1:nFeatures)  * xRot;                
                    
                catch err
                    xHat = data;
                    xRot = zeros(nFeatures,size(data,2));
                    disp(err.getReport);
                end
            end
        end
        
        
        %==================================================================
        % This function reduces the dimension using principal component
        % analysis and speicified number of features to be reduced.
        % (nFeatures). Same function on GPU
        %==================================================================
        function [xRot, xHat] = reduce_dimension_by_nFeatures_onGPU(data, nFeatures)
            xHat = [];
            xRot = [];
            if ~isempty(data) && ~(nargin<2 ||isempty(nFeatures))
                gpu = gpuDevice(1);
                try                                                            
                    data_d = gpuArray(data);                    
                    
                    % Calculate covariance matrix on GPU
                    sigma_d = PreProcessing.calculate_covariance_matrix_onGPU(data_d);
                    
                    [U_d, ~, ~] = svd(sigma_d); % Singular value decomposition
                    U   = gather(U_d);
                    
                    % Compute xRot, the projection on to the eigenbasis
                    xRot = U(:,1:nFeatures)' * data;

                    %  Following the dimension reduction, invert the PCA transformation to produce 
                    %  the matrix xHat, the dimension-reduced data with respect to the original basis.
                    xHat = U(:,1:nFeatures)  * xRot;                     
                    
                    clear('sigma_d','data_d','U_d','S_d','V_d');
                    wait(gpu);                    
                    
                catch err
                   disp(err.getReport);
                   disp('Resetting GPU...')
                   reset(gpu);
                end                
            end
        end
        
        
        %==================================================================
        % Reduce redundancy in the input and make features all have the 
        % same variance with whitening. Unlike PCA alone, whitening 
        % additionally ensures that the diagonal entries are equal to 1, 
        % i.e. that the covariance matrix is the identity matrix when you
        % are doing whitening along with no regularization (epsilon = 0).
        % When whitening with regularization (to avoid numerical issues for
        % small eigenvalues), some of the diagonal entries of the
        % covariance will be smaller than 1.
        %==================================================================
        function [PCAWhite xPCAWhite] = apply_PCA_whitening(data, epsilon)
            xPCAWhite = data;
            if ~isempty(data)                 
                if (nargin<2 ||isempty(epsilon))
                    epsilon = 0; % do not regularize by default 
                end
                
                [U,S] = PreProcessing.apply_PCA(data); 
                                    
                PCAWhite = diag(1./sqrt(diag(S) + epsilon)) * U' ;
                xPCAWhite = PCAWhite * data;
                
            end
        end
        
        
        %==================================================================
        % Reduce redundancy in the input and make features all have the 
        % same variance and mean normalized with whitening. Rest is same
        % with PCA whitening.
        %==================================================================
        function [ZCAWhite xZCAWhite] = apply_ZCA_whitening(data, epsilon)
            xZCAWhite = data;
            if ~isempty(data)                 
                if (nargin<2 ||isempty(epsilon))
                    epsilon = 0; % do not regularize by default 
                end
                
                [U,S] = PreProcessing.apply_PCA(data); 
                                    
                ZCAWhite  = U * diag(1./sqrt(diag(S) + epsilon)) * U' ;
                xZCAWhite = ZCAWhite * data;

            end
        end
        
        
        %==================================================================
        % Use this function for [0 255] multiple-band images (e.g. RGB)
        % Mean / Std normalization is done implicitly. Useful for image
        % classification pipelines 
        % NOTE THAT different from other functions data must be < m x n > 
        % matrix where m is number of samples and n is number of features. 
        %==================================================================
        function [ZCAWhite xZCAWhite xMean] = apply_ZCA_whitening_multiBand(data, epsilon)
            xZCAWhite = data;
            if ~isempty(data)                 
                if (nargin<2 ||isempty(epsilon))
                    epsilon = 0; % do not regularize by default 
                end
                
                % Normalize for contrast
                data = bsxfun(@rdivide, bsxfun(@minus, data, mean(data,2)), sqrt(var(data,[],2)+10));
                
                % ZCA whitening
                C = cov(data);
                xMean = mean(data);
                [V,D] = eig(C);
                ZCAWhite = V * diag(sqrt(1./(diag(D) + epsilon))) * V';
                xZCAWhite = bsxfun(@minus, data, xMean) * ZCAWhite; 
                
                
            end
        end
        
                
        %==================================================================
        % Use this function for [0 255] single-band images (gray-level)
        % Mean / Std normalization is done implicitly. 
        %==================================================================
        function patches = remove_DC_and_squash(patches)            
            
            % Remove DC (mean of images). 
            patches = bsxfun(@minus, patches, mean(patches));

            % Truncate to +/-3 standard deviations and scale to -1 to 1
            pstd = 3 * std(patches(:));
            patches = max(min(patches, pstd), -pstd) / pstd;

            % Rescale from [-1,1] to [0.1,0.9]
            patches = (patches + 1) * 0.4 + 0.1;
            
        end
        
        
        %==================================================================
        % This function removes (regresses-out) conditions from data which
        % are assumed to be linear using general linear model. Conditions
        % can be linear trends (scanner drift for fMRI data), bias
        % components (offset for fMRI data) and etc. Cleaned data and
        % regressor weights are both returned. For linear trend removal
        % process add a regularly increasing column vector to conditions 
        % matrix (eg.[1:nSamples]). For bias removal process add a column 
        % vector to conditions matrix that is composed of all ones 
        % (eg. ones(nSamples,1)).
        %
        %   Inputs : 
        %       data         : matrix of size <nFeatures x nSamples>.
        %       conditions   : matrix of size <nSamples x nConds>.
        %       discardConds : vector of size <nConds,1>. Conditions that
        %           will be discarded from removal.
        %   Output :
        %       cleanedData  : cleaned data, same size with input data.
        %       betaMatrix   : beta coefficients for each feature and for
        %           each condition.
        % 
        %==================================================================
        function [cleanedData, betaMatrix] = regress_out_conditions(data, conditions, discardConds)
            
            nFeatures = size(data,1);
            nSamples  = size(data,2);
            nConds    = size(conditions,2);
            betaMatrix  = zeros(nFeatures, nConds); 
            cleanedData = zeros(nFeatures, nSamples); 
            
            % if not specified regress-out both conditions
            if nargin < 3 
               discardConds = zeros(nConds,1); 
            end            
            
            % linear regression on the samples (time series for fmri data)
            % y = X*beta + e ; which has a closed form solutions as follows
            % by simple algebra and assuming error is zero, simply :)
            % beta = inv(X'*X)*X'*y' and we know pinv(X) = inv(X'*X)*X'
            % for non-invertible situations
            pInvX = pinv(conditions);
            
            % calculate each beta for each feature and for each sample and
            % regress out the conditions as y_clean = y - X*beta; 
            for i=1:nFeatures       
                
                % calculate weights
                betaMatrix(i,:) = reshape(pInvX * data(i,:)', [], 1);
                
                % regress out the conditions where y_hat = X*beta
                cleanedData(i,:) = data(i,:)' - reshape(sum(bsxfun(@times,conditions(:,~discardConds),betaMatrix(i,~discardConds)),2),[],1);
            end
            
        end
        
                
        %==================================================================
        % Standard General Linear Model (GLM) analysis given regressors.
        % Both betaMap (reconstructed data) and linear regression weights
        % (beta weights) are returned. For t-statistics beta weights should
        % be further used in order to obtain p values.
        %
        %   Inputs : 
        %       data         : matrix of size <nFeatures x nSamples>.
        %       regressors   : matrix of size <nSamples x nRegressors>.
        %       discardRegs  : vector of size <nRegressors,1>. Regressors 
        %           that will be discarded from betaMap.
        %   Output :
        %       reconstData  : reconstructed data,same size with input data
        %       betaMatrix   : beta coefficients, for each feature and for
        %           each condition.
        % 
        %==================================================================
        function [reconstData, betaMatrix] = apply_GLM(data, regressors, discardRegs)
            
            nFeatures   = size(data,1);
            nSamples    = size(data,2);
            nRegressors = size(regressors,2);
            betaMatrix  = zeros(nFeatures, nRegressors); 
            reconstData = zeros(nFeatures, nSamples); 
            
            % if not specified regress-out both conditions
            if nargin < 3 
               discardRegs = zeros(nRegressors,1); 
            end     
            
            % linear regression on the samples (time series for fmri data)
            % y = X*beta + e ; which has a closed form solutions as:
            % beta = inv(X'*X)*X'*y' and we know pinv(X) = inv(X'*X)*X'
            pInvX = pinv(regressors);
            
            % calculate each beta for each feature and for each sample and
            % regress out the conditions as y_clean = y - X*beta; 
            % Perhaps there are faster/vectorized ways to do the following.
            for i=1:nFeatures       
                
                % calculate weights
                betaMatrix(i,:) = reshape(pInvX * data(i,:)', [], 1);
                                
                % reconstruct the cleaned signal
                beta_clean = betaMatrix(i,:);
                if numel(discardRegs)~=1
                    beta_clean(discardRegs) = 0; % exclude regressor                     
                end
                y_clean = regressors*beta_clean';
                reconstData(i,:) = y_clean(:)';
                
            end
            
        end                      
        
        
        %==================================================================
        % This function performs F-test given original data y and
        % recosntructed data yHat using nRegressor number of regressors.
        % F-test resulting F values are converted into probability values
        % further and function returns both F values and p values.
        %
        %   Inputs :
        %       y           : Original signal/observation,
        %                       matrix of size <nFeatures x nSamples>.
        %       yHat        : Reconstructed signal/observation,
        %                       matrix of size <nFeatures x nSamples>.
        %       nRegressors : Number of regressors/conditions used in
        %                       reconstruction
        %   Output :
        %       F_values    : Resulting F-test values,
        %                       vector of size nFeatures
        %       p_values    : Converted probabilty values for F-test result
        %                       vector of size nFeatures
        %
        %==================================================================
        function [F_values, p_values] = perform_f_test(y, yHat, nRegressors)
            F_values = [];
            p_values = [];
            if ~(isempty(y) || isempty(yHat) || ~isequal(size(y),size(yHat)) || nargin<3 )
                
                nSamples = size(y,2); % as the number of time points
                
                numerator_dof   = (nRegressors-1);        % degrees of freedom 
                denominator_dof = (nSamples-nRegressors); % degrees of freedom 
                
                %----------------------------------------------------------
                % compute R^2 = var(yHat)/var(y)
                % note that var(y) = var(yHat) + var(error)                
                var_yHat = var(yHat,0,2);
                var_e    = var(y-yHat,0,2);                
                RR       = var_yHat ./ (var_yHat + var_e);
                
                %----------------------------------------------------------
                % compute F statistics
                F_values = (RR .* denominator_dof) ./ ...
                            ( (1-RR) .* numerator_dof );
                
                %----------------------------------------------------------
                % convert it into p-values
                p_values = 1-fcdf(F_values,numerator_dof,denominator_dof);
                
            end
        end
        
        
        %==================================================================
        % Maps given data into the interval [0,1], minimum value is set to
        % zeros and maximum value is set to 1 considering entire data
        % matrix. If your data have some noise or outliers (that may cause
        % blow offs or shrinkage of rest of the data), you have to
        % eliminate those elements before calling this function.
        %==================================================================
        function data = normalize_zero_one(data)
            if ~isempty(data)
                data = mat2gray(data);
            end
        end
        
    end
    
end

