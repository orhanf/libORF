classdef PatchSampler 
%==========================================================================
% Class for Patch Sampler with static methods, following utilities are 
%   provided:
%   
%   - Random patch sampling within binary mask
%   - GT patch extraction given images and gt
%   - GT patch extraction with rotation given images and gt
%   - Apply a function handle to components or bounding boxes
%   - Rotate given patches matrix elements by 4 times, 90 degrees 
%   - Split image to tiles with two buffering options
%   - Cropping window tile computations
%   - GT patch extraion within components
%   - Sampling random 1d temporal windows for time-series
%   - Sampling random 2d spatial patches for 3d volumes(from slices)
%
% orhanf - (c) 2012 Orhan FIRAT <orhan.firat@ceng.metu.edu.tr>
%==========================================================================
    
    methods(Static)
                 
        %==================================================================
        % Discrete uniform sampling within given binary mask
        %   Inputs : 
        %       images     : cell of images (nImages,1)
        %       masks      : cell of masks (nImages,1)
        %       patchsize  : scaler
        %       numpatches : scaler
        %   Output :
        %       patches(patchsize*patchsize, numpatches)
        %==================================================================
        function patches = sample_patches_within_mask(images, masks, patchsize, numpatches)        
                                
            nImages = numel(images);            
            
            if ~nImages
                patches = [];
                return;
            end
            
            % allocate outputs and helpers
            nBands     = size(images{1},3);
            patches    = zeros(patchsize*patchsize*nBands, numpatches);                        
            masks_map0 = cell(nImages,1); 
            nSampled   = numpatches;
            
            for i=1:nImages
                masks_map0{i} = find(masks{i}==0);
            end
            i = 1;
            while(i<=nSampled)

                if (mod(i,1000) == 0), fprintf('Extracting patch: %d / %d\n', i, numpatches); end

                m = random('unid', nImages);  
                
                IMAGE_DIM = size(images{m});
                
                r = random('unid', IMAGE_DIM(1) - patchsize + 1);
                c = random('unid', IMAGE_DIM(2) - patchsize + 1);  
                try 
                if PatchSampler.check_boudary_condition_linear(r, c, IMAGE_DIM(1), IMAGE_DIM(2), patchsize, masks_map0{m})
                    patches(:,i) = reshape(images{m}(r:r+patchsize-1,c:c+patchsize-1,:), [patchsize*patchsize*nBands, 1]);
                    i = i+1;                                        
                end 
                catch err
                    m
                    r
                    c
                end
            end            
            
        end
                       

        %==================================================================
        % Collect window samples over skeleton with specified interval
        %   Inputs : 
        %       images     : cell of images (nImages,1)
        %       masks      : cell of masks (nImages,1)
        %       patchsize  : scaler
        %   Output :
        %       gtPatches  : cell of patches (1, nImages)
        %==================================================================
        function patchCell = sample_patches_over_skeleton(images, masks, patchsize, interval)
            
            nImages = numel(images);            
            
            if ~nImages
                patchCell = [];
                return;
            end
            
            % allocate outputs and helpers
            patchCell = cell(1, nImages);                        
               
            % obtain patches
            for i=1:nImages
                
                fprintf('Processing %d/%d\n', i, nImages);
                
                IMAGE_DIM  = size(images{i});                
                components = bwlabel(logical(masks{i}));                
                nComps     = max(components(:));
                nBands     = size(images{i},3);
                  
                % allocate patches for this image
                patchesPerImg = cell(1,nComps);
                
                for j=1:nComps

                    [skg, rad] = Skel.compute_skeleton(components == j);                    
                    linearInds = find(skg>50); % pruning a little
                    
                    pointInds  = linearInds(1:interval:end);                                        
                    nSamples   = numel(pointInds);
                                        
                    [x y] = ind2sub( IMAGE_DIM(1:2), pointInds);
                    
                    % allocate patches for samples
                    patches  = zeros(patchsize*patchsize*nBands ,nSamples);
                    
                    errorIdx = [];
                    
                    for k=1:nSamples
                    
                        centerx = x(k);
                        centery = y(k);

                        idxStart = max([1, floor(centerx - (patchsize/2) )]);
                        idyStart = max([1, floor(centery - (patchsize/2) )]);
                        idxEnd   = min([IMAGE_DIM(1), idxStart + patchsize - 1]);
                        idyEnd   = min([IMAGE_DIM(2), idyStart + patchsize - 1]);

                        imgCrop = images{i}(idxStart:idxEnd,idyStart:idyEnd,:);
                    
                        try 
                            patches(:,k) = imgCrop(:);
                        catch err                            
                            errorIdx = [errorIdx k];
                            disp(['Cropped image is smaller than window size!...']); 
                            disp(err.message);
                        end                                                
                    end
                    
                    patches(:,errorIdx) = [];
                    patchesPerImg{1,j} = patches;
                    
                end 
                patchCell{1,i} = cell2mat(patchesPerImg); 
            end            
        end
        
        
        %==================================================================
        % Obtain gt patches without rotating, centering gt within patchdim
        % sized square window, all bands of original image is taken.
        %   Inputs : 
        %       images       : cell of images (nImages,1)
        %       gtMasks      : cell of masks (nImages,1)
        %       imageDim     : scaler
        %   Output :
        %       gtPatches    : cell of patches (1, nImages)
        %==================================================================
        function gtPatches = obtain_gt_patches(images, gtMasks, patchDim)
            
            nImages = numel(images);            
            
            if ~nImages
                gtPatches = [];
                return;
            end
            
            % allocate outputs and helpers
            gtPatches  = cell(1, nImages);                        
               
            % obtain patches
            for i=1:nImages
                
                fprintf('Processing %d/%d\n', i, nImages);
                
                IMAGE_DIM = size(images{i});
                nBands    = size(images{i},3);
                
                stats   = regionprops(logical(gtMasks{i}),'BoundingBox');                                                
                patches = zeros(patchDim*patchDim*nBands ,numel(stats));
                
                for j=1:numel(stats)

                    centerx = floor(stats(j).BoundingBox(2) + (stats(j).BoundingBox(4) / 2));
                    centery = floor(stats(j).BoundingBox(1) + (stats(j).BoundingBox(3) / 2));
                    
                    idxStart = max([1, floor(centerx - (patchDim/2) )]);
                    idyStart = max([1, floor(centery - (patchDim/2) )]);
                    idxEnd   = min([IMAGE_DIM(1), idxStart + patchDim - 1]);
                    idyEnd   = min([IMAGE_DIM(2), idyStart + patchDim - 1]);
                    
                    imgCrop = images{i}(idxStart:idxEnd,idyStart:idyEnd,:);
                    
                    patches(:,j) = imgCrop(:);
                end 
                
                gtPatches{1,i} = patches;
                
            end
        end
        
        
        %==================================================================
        % Obtain all patches without rotating, within masks        
        %   Inputs : 
        %       images   : cell of images (nImages,1)
        %       masks    : cell of masks (nImages,1)
        %       patchDim : scaler
        %   Output :
        %       patches  : cell of patches (1, nImages)
        %==================================================================
        function patchesCell = obtain_all_patches_within_mask(images, masks, patchDim)
            
            nImages = numel(images);            
            
            if ~nImages
                patchesCell = [];
                return;
            end
            
            % allocate outputs and helpers
            patchesCell = cell(1, nImages);                                               
                        
            for m=1:nImages
                
                fprintf('Processing %d/%d\n', m, nImages);
                
                map0 = find(masks{m}==0);            
                IMAGE_DIM = size(images{m});
                patches = [];
                
                counter = 1;
                for i=1:patchDim:IMAGE_DIM(1)
                    for j=1:patchDim:IMAGE_DIM(2)                        
                        
                        idxStart = i;
                        idyStart = j;
                        idxEnd   = idxStart + patchDim - 1;
                        idyEnd   = idyStart + patchDim - 1;
                        
                        if (idxEnd > IMAGE_DIM(1)) || (idyEnd > IMAGE_DIM(2))
                           continue; 
                        end
                        
                        linearInd = sub2ind([IMAGE_DIM(1), IMAGE_DIM(2)], repmat(idxStart,1,idxEnd-idxStart+1), (idyStart:idyEnd) );

                        linearInd = bsxfun(@plus,repmat(linearInd,idxEnd-idxStart+1,1),(0:(idyEnd-idyStart))');
                            
                        if sum(ismember(linearInd(:),map0(:))) == 0
                            patch = images{m}(idxStart:idxEnd,idyStart:idyEnd,:);
                            patches(:,counter) = double(patch(:));
                            counter = counter + 1;    
                        end          
                                                    
                    end
                end
                patchesCell{m} = patches;
                clear patches
            end
           
        end
        
        
        %==================================================================
        % Obtain gt patches with rotating, centering gt within patchdim
        % sized square window, all bands of original image is taken.
        % Rotation is applied 3 times with 90 degrees each, total 4 samples
        % are taken for each patch.
        %   Inputs : 
        %       images       : cell of images (nImages,1)
        %       gtMasks      : cell of masks (nImages,1)
        %       imageDim     : scaler
        %   Output :
        %       gtPatches    : cell of patches (1, nImages)
        %==================================================================
        function gtPatches = obtain_gt_patches_rot90(images, gtMasks, patchDim)
            
            nImages = numel(images);            
            
            if ~nImages
                gtPatches = [];
                return;
            end
            
            % allocate outputs and helpers
            gtPatches  = cell(1, nImages);                        
               
            % obtain patches
            for i=1:nImages
                
                fprintf('Processing %d/%d\n', i, nImages);
                
                IMAGE_DIM = size(images{i});
                nBands    = size(images{i},3);
                
                stats   = regionprops(logical(gtMasks{i}),'BoundingBox');                                                
                patches = zeros(patchDim*patchDim*nBands ,numel(stats)*4);
                
                for j=1:numel(stats)

                    centerx = floor(stats(j).BoundingBox(2) + (stats(j).BoundingBox(4) / 2));
                    centery = floor(stats(j).BoundingBox(1) + (stats(j).BoundingBox(3) / 2));
                    
                    idxStart = max([1, floor(centerx - (patchDim/2) )]);
                    idyStart = max([1, floor(centery - (patchDim/2) )]);
                    idxEnd   = min([IMAGE_DIM(1), idxStart + patchDim - 1]);
                    idyEnd   = min([IMAGE_DIM(2), idyStart + patchDim - 1]);
                    
                    imgCrop    = images{i}(idxStart:idxEnd,idyStart:idyEnd,:);
                    imgCrop90  = imrotate(imgCrop,90);  
                    imgCrop180 = imrotate(imgCrop,180);  
                    imgCrop270 = imrotate(imgCrop,270);  
                    
                    ind = (j-1)*4 + 1;
                    
                    patches(:,ind)   = imgCrop(:);
                    patches(:,ind+1) = imgCrop90(:);
                    patches(:,ind+2) = imgCrop180(:);
                    patches(:,ind+3) = imgCrop270(:);
                end 
                
                gtPatches{1,i} = patches;
                
            end
        end        
           
        
        %==================================================================
        % Obtain gt component responses given gt mask and convolved image
        % with learned representations.
        %   Inputs : 
        %       convolvedImg : <nRows-patchDim+1,nCols-patchDim+1,nFeatures> matrix
        %       gtMask       : <nRows, nCols> binary gt mask
        %       minArea      : scaler to indicate minimum component area
        %   Output :
        %       gtComponentResps : cell of component responses (1, nComps)
        %==================================================================
        function gtComponentResps = obtain_gt_component_responses(convolvedImg, gtMask, minArea)
            
            % consider all components by default
            if nargin < 3, minArea = 0; end 
            
            % correct convolution artifacts if exist
            nRows = size(convolvedImg,1);
            nCols = size(convolvedImg,2);
            nFeatures = size(convolvedImg,3);
            nElements = nRows * nCols; 
                                    
            if nRows ~= size(gtMask,1), gtMask(nRows+1:end,:) = []; end % discard surplus rows
            if nCols ~= size(gtMask,2), gtMask(:,nCols+1:end) = []; end % discard surplus cols
                                        
            % extract connected components in gt mask
            stats  = regionprops(logical(gtMask),'PixelIdxList');           
            nComps = numel(stats);
            
            % allocate output
            gtComponentResps = zeros(nFeatures, nComps);
                        
            % reference interval index vector to be used for indexing
            % convolved images for linear fetching
            refInterval = (0:nFeatures-1) .* nElements;            
            
            % process each component in binary mask            
            for i=1:nComps
                               
                if (numel(stats(i).PixelIdxList)) < minArea, continue; end
                
                % calculate index for each feature
                indsThis = bsxfun(@plus, repmat(stats(i).PixelIdxList,1,nFeatures), refInterval);
                
                % obtain corresponding responses and take mean
                gtComponentResps(:,i) = mean(convolvedImg(indsThis));
                
            end            
        end
        
        
        %==================================================================
        % Apply given function to the patches of images by overlaping 
        %==================================================================        
        function outimg = generate_heat_map_using_func(img, imgFunc, patchDim, interval)
            
            IMAGE_DIM = size(img);

            img = double(img);
            
            outimg = zeros(IMAGE_DIM(1),IMAGE_DIM(2));
            
            nRows = (IMAGE_DIM(1)-patchDim+1);
            nCols = (IMAGE_DIM(2)-patchDim+1);
            
            h = waitbar(0,'Please wait...');
            steps = ceil(nRows/interval) * ceil(nCols/interval);
            step  = 1;
            
            for i=1:interval:nRows
                for j=1:interval:nCols

                    waitbar(step / steps); step = step + 1;
                    
                    idxStart = i;
                    idyStart = j;
                    idxEnd   = idxStart + patchDim - 1;
                    idyEnd   = idyStart + patchDim - 1;
                   
                    % imgFunc is expected to be a classifier test function
                    [pred acc] = imgFunc(img(idxStart:idxEnd,idyStart:idyEnd,:));
                    
                    if pred == 1
                        outimg(idxStart:idxEnd, idyStart:idyEnd) =  1 + outimg(idxStart:idxEnd, idyStart:idyEnd);                    
                    end
                end
            end               
            close(h); 
        end
        
        
        %==================================================================
        % Apply given function to the patches of images by overlaping
        % within given mask
        %==================================================================        
        function outimg = generate_heat_map_using_func_within_mask(img, mask, imgFunc, patchDim, interval)
            
            IMAGE_DIM = size(img);

            img    = double(img);
            map0   = find(mask==0);
            outimg = zeros(IMAGE_DIM(1),IMAGE_DIM(2));
            
            nRows = (IMAGE_DIM(1)-patchDim+1);
            nCols = (IMAGE_DIM(2)-patchDim+1);
            
%             h = waitbar(0,'Please wait...');
            steps = ceil(nRows/interval) * ceil(nCols/interval);
            step  = 1;
            
            for i=1:interval:nRows
                for j=1:interval:nCols
                                        
                    idxStart = i;
                    idyStart = j;
                    idxEnd   = idxStart + patchDim - 1;
                    idyEnd   = idyStart + patchDim - 1;
                   
%                     waitbar(step / steps); step = step + 1;
                    if mod(step,1000) == 0, fprintf('processing... %d/%d\n',step,steps); end
                    step = step + 1;
                    
                    if PatchSampler.check_boudary_condition_linear(idxStart, idyStart, IMAGE_DIM(1), IMAGE_DIM(2), patchDim, map0)
                    
                        % imgFunc is expected to be a classifier test function
                        [pred acc] = imgFunc(img(idxStart:idxEnd,idyStart:idyEnd,:));

                        if pred == 1
                            outimg(idxStart:idxEnd, idyStart:idyEnd) =  1 + outimg(idxStart:idxEnd, idyStart:idyEnd);                    
                        end
                        
                    end
                end
            end            
%             close(h); 
        end
        
        
        %==================================================================
        % Apply given function to the patches of images within given mask
        %==================================================================        
        function patches = apply_func_to_patches_within_mask(images, masks, patchDim, func, validRatio)
            
            nImages = numel(images);            
            
            if ~nImages
                patches = [];
                return;
            end
            
            % allocate outputs and helpers
            patches  = cell(1, nImages);                        
               
            % obtain patches
            for i=1:nImages                
                fun = @(block_struct)PatchSampler.func_to_patch(block_struct, func, masks{i}, validRatio);                                
                patches{i} = blockproc(images{i},[patchDim patchDim],fun);                
            end
        end
       
        
        %==================================================================
        % Utility function, do not use explicity, use as a handle for
        % blockproc (see function : apply_func_to_patches_within_mask)
        %==================================================================        
        function newPatch = func_to_patch(block_struct, func, mask, validRatio)
        
            newPatch = zeros(size(block_struct.data));
            
            aoe = mask(block_struct.location(1):block_struct.location(1)+block_struct.blockSize(1)-1,...
                       block_struct.location(2):block_struct.location(2)+block_struct.blockSize(2)-1);
            
            if validRatio <= sum(logical(aoe(:))) / block_struct.blockSize(1)*block_struct.blockSize(2)
                newPatch = func(block_struct.data);
            end
        end
        
        
        %==================================================================
        % Apply given function to the images, where connected components of 
        % corresponding binary masks are overlaid on images. Note
        % function is applied directly to the connected components, if you
        % prefer bounding box of components use function *_bb
        %==================================================================        
        function outImgs = apply_func_to_components(images, masks, func)
        
            nImages = numel(images);            
            
            % all input images must be logical (binary mask)
            if ~nImages || any(cellfun(@(x)~islogical(x),masks))
                outImgs = [];
                return;
            end
            
            % allocate outputs 
            outImgs  = cell(1, nImages);                        
               
            % process each image
            for i=1:nImages                
                
                nRows  = size(images{i},1);
                nCols  = size(images{i},2);
                nBands = size(images{i},3);
                
                outImgThis = zeros(nRows, nCols);
                
                stats = regionprops(masks{i}, 'PixelIdxList');
                
                for j=1:numel(stats)                                                       
                    
                    compImg = uint8(zeros(1, numel(stats(j).PixelIdxList), nBands));
                    
                    for k=1:nBands       
                        band = images{i}(:,:,k);
                        compImg(:,:,k) = band(stats(j).PixelIdxList);
                    end
                    
                    tmpres = func(compImg);
                    outImgThis(stats(j).PixelIdxList) = tmpres(:,:,1);
                    
                end
                
                outImgs{i} = outImgThis;
                
            end
        end

        
        %==================================================================
        % Apply given function to the bounding box of connected components 
        % in images : TODO not completed
        %==================================================================        
        function outImgs = apply_func_to_components_bb(images, func)
        
            nImages = numel(images);            
            
            % all input images must be logical (binary mask)
            if ~nImages || any(cellfun(@(x)~islogical(x),masks))
                outImgs = [];
                return;
            end
            
            % allocate outputs 
            outImgs  = cell(1, nImages);                        
               
            % process each image
            for i=1:nImages                
                
                nRows  = size(images{i},1);
                nCols  = size(images{i},2);
                nBands = size(images{i},3);
                
                outImgThis = zeros(nRows, nCols);
                
%                 stats = regionprops(masks{i}, 'PixelIdxList','BoundingBox');
%                 
%                 for j=1:numel(stats)                                                       
%                     
%                     compImg = uint8(zeros(1, numel(stats(j).PixelIdxList), nBands));
%                     
%                     for k=1:nBands       
%                         band = images{i}(:,:,k);
%                         compImg(:,:,k) = band(stats(j).PixelIdxList);
%                     end
%                     
%                     tmpres = func(compImg);
%                     outImgThis(stats(j).PixelIdxList) = tmpres(:,:,1);                    
%                 end                
                outImgs{i} = outImgThis;                
            end
        end
            
        
        %==================================================================
        % Utility function to check boundary conditions given reference
        % linear indices 
        %==================================================================
        function isValid = check_boudary_condition_linear(row, col, nRows, nCols, patchsize, refInd)
           
            isValid = false;
                        
            linearInd = sub2ind([nRows, nCols], repmat(row,1,patchsize), (col:col+patchsize-1) );                                     
            
            linearInd = bsxfun(@plus,repmat(linearInd,patchsize,1),(0:patchsize-1)');
            
            if sum(ismember(linearInd(:),refInd(:))) == 0
                isValid = true;
            end
        end
        
        
        %==================================================================
        % Utility function to rotate each patch 4 times each 90 degrees
        % given patches matrix where each patch is in its columns
        %==================================================================
        function patches90 = rotate90_patches(patches, patchDim)
        
            nPatches  = size(patches,2);            
            patches90 = zeros(size(patches,1), nPatches*4);
            
            patches90(:,1:4:end) = patches;
            
            for i=1:nPatches                
                
                patch = reshape(patches(:,i), patchDim);
                patch90  = imrotate(patch,90);  
                patch180 = imrotate(patch,180);  
                patch270 = imrotate(patch,270);
            
                ind = (i-1)*4+2;
                patches90(:,ind:ind+2) = [patch90(:), patch180(:), patch270(:)];
                
            end
        end
        
        
        %==================================================================
        % Utility function to split image into tiles, tiles are determined
        % by chunk size on x and y dimensions additional patch dimensions
        % are for convolution artifacts, if no convolution will be applied
        % then set patchDimX and patchDimY to -1.
        %==================================================================        
        function tiles = split_image_to_tiles(Img, tileSizeX, tileSizeY, patchDimX, patchDimY)
            
            dimX = size(Img,1);
            dimY = size(Img,2);
            
            startIndsX = 1:tileSizeX:dimX;
            startIndsY = 1:tileSizeY:dimY;
            
            endIndsX = startIndsX + tileSizeX - 1;  endIndsX(dimX<endIndsX) = dimX;
            endIndsY = startIndsY + tileSizeY - 1;  endIndsY(dimY<endIndsY) = dimY;
            
            tiles = cell(ceil(dimX/tileSizeX),ceil(dimY/tileSizeY));
           
            for i=1:numel(startIndsX)
                
                startIdx = startIndsX(i); 
                if i~=1, startIdx = startIndsX(i) - patchDimX + 1; end
                
                for j=1:numel(startIndsY)
                    
                    startIdy = startIndsY(j);
                    if j~=1, startIdy = startIndsY(j) - patchDimY + 1; end
                    
                    tiles{i,j} = Img(startIdx:endIndsX(i),startIdy:endIndsY(j),:);                    
                end                
            end
            
            
        end
        
        
        %==================================================================
        % Utility function to split image into tiles, tiles are determined
        % by chunk size on x and y dimensions additional patch dimensions
        % are for convolution artifacts, if no convolution will be applied
        % then set patchDimX and patchDimY to -1. This second version crops
        % buffer region from both dimensions not only lower-right part
        %==================================================================        
        function tiles = split_image_to_tiles2(Img, tileSizeX, tileSizeY, patchDimX, patchDimY, imageDimX, imageDimY)
            
            dimX = size(Img,1);
            dimY = size(Img,2);
            
            startIndsX = 1:tileSizeX:dimX;
            startIndsY = 1:tileSizeY:dimY;
            
            endIndsX = startIndsX + tileSizeX - 1;  endIndsX(dimX<endIndsX) = dimX;
            endIndsY = startIndsY + tileSizeY - 1;  endIndsY(dimY<endIndsY) = dimY;
            
            tiles = cell(ceil(dimX/tileSizeX),ceil(dimY/tileSizeY));
           
            for i=1:numel(startIndsX)
                
                startIdx = startIndsX(i); 
                if i~=1, startIdx = startIndsX(i) - (patchDimX+imageDimX) + 1; end
                  
                endIdx = min([ (endIndsX(i) + imageDimX + patchDimX - 1), dimX]);
                
                for j=1:numel(startIndsY)
                    
                    startIdy = startIndsY(j);
                    if j~=1, startIdy = startIndsY(j) - (patchDimY+imageDimY) + 1; end
                
                    endIdy = min( [ (endIndsY(j) + imageDimY + patchDimX - 1), dimY]);
                    
                    tiles{i,j} = Img(startIdx:endIdx,startIdy:endIdy,:);                    
                end                
            end                        
        end

        
        %==================================================================
        % TODO : Complete comments
        %==================================================================
        function tiles = split_image_to_tiles_noBuffer(Img, tileSizeUD, tileSizeLR, patchDimX, patchDimY, imageDimX, imageDimY)
            
            dimX = size(Img,1);
            dimY = size(Img,2);
            
            startIndsX = 1:tileSizeUD:dimX;
            startIndsY = 1:tileSizeLR:dimY;
            
            endIndsX = startIndsX + tileSizeUD - 1;  endIndsX(dimX<endIndsX) = dimX;
            endIndsY = startIndsY + tileSizeLR - 1;  endIndsY(dimY<endIndsY) = dimY;
            
            tiles = cell(ceil(dimX/tileSizeUD),ceil(dimY/tileSizeLR));
           
            for i=1:numel(startIndsX)
                
                startIdx = startIndsX(i); 
                if i~=1, startIdx = startIndsX(i) - (imageDimX - 1); end
                  
                endIdx = min([ (endIndsX(i) + imageDimX + (imageDimX - patchDimX) - 1), dimX]);
                
                for j=1:numel(startIndsY)
                    
                    startIdy = startIndsY(j);
                    if j~=1, startIdy = startIndsY(j) - (imageDimY - 1); end
                
                    endIdy = min( [ (endIndsY(j) + imageDimY + (imageDimY - patchDimY) - 1), dimY]);
                    
                    tiles{i,j} = Img(startIdx:endIdx,startIdy:endIdy,:);
                    disp(['<[' num2str(startIdx) ':' num2str(endIdx) '],[' num2str(startIdy) ':' num2str(endIdy) ']>']);
                end                
            end                        
        end
        
        
        %==================================================================
        % TODO : Complete comments
        %==================================================================
        function [cropXleft, cropXright, cropYup, cropYdown] = ...
                calculate_cropping_window_noBuffer( TILE_DIM,IMG_DIM,...
                                                    tileSizeX,tileSizeY,...
                                                    imageDimX,imageDimY,...
                                                    currInd)
            
            [tileIdy,tileIdx] = ind2sub(TILE_DIM,currInd);
            
            % default indices
            cropXleft  = 1;
            cropXright = IMG_DIM(2);
            cropYup    = 1;
            cropYdown  = IMG_DIM(1);
            
            % determine left boundary            
            if  tileIdx ~= 1
                cropXleft = imageDimX;
            end
                
            % determine right boundary            
            if tileIdx ~= TILE_DIM(2)
                cropXright = min((cropXleft + tileSizeX - 1), IMG_DIM(2));
            end
                
            % determine upper boundary
            if tileIdy ~= 1
                cropYup = imageDimY;
            end
                
            % determine lower boundary
            if tileIdy ~=TILE_DIM(1)
                cropYdown = min((cropYup + tileSizeY - 1),IMG_DIM(1));
            end
            
        end
        
        
        %==================================================================
        % TODO : Complete comments
        %==================================================================
        function [tileSizeX, tileSizeY] = calculate_tile_size(IMG_DIM, imageDimX, imageDimY, patchDimX, patchDimY)
            
            maxtileSizeX = 500;
            maxtileSizeY = 500;
        
            minTileSizeX = 250;
            minTileSizeY = 250;
            
            minBufferX = imageDimX + (imageDimX - patchDimX);
            minBufferY = imageDimY + (imageDimY - patchDimY);
            
            dimX = IMG_DIM(2);
            dimY = IMG_DIM(1);
            
            for tileSizeX=maxtileSizeX:-50:minTileSizeX
                startIndsX = 1:tileSizeX:dimX;                
                endIndsX = startIndsX + tileSizeX - 1;  endIndsX(dimX<endIndsX) = dimX;              
                if all( (endIndsX - startIndsX + 1) > minBufferX), break; end
            end
            
            for tileSizeY=maxtileSizeY:-50:minTileSizeY
                startIndsY = 1:tileSizeY:dimY;
                endIndsY = startIndsY + tileSizeY - 1;  endIndsY(dimY<endIndsY) = dimY;
                if all( (endIndsY - startIndsY + 1) > minBufferY), break; end
            end
            
            disp(['Optimum tile size <' num2str(tileSizeY) ',' num2str(tileSizeX) '>']);
            
        end
        
   
        function [cropXleft, cropXright, cropYup, cropYdown] = ...
                calculate_cropping_window(TILE_DIM,IMG_DIM,imageDimX,imageDimY,patchDimX,patchDimY,currInd)
            
            [tileIdy,tileIdx] = ind2sub(TILE_DIM,currInd);
            
            % default indices
            cropXleft  = 1;
            cropXright = IMG_DIM(2);
            cropYup    = 1;
            cropYdown  = IMG_DIM(1);
            
            % determine left boundary            
            if  tileIdx ~= 1
                cropXleft = imageDimX + patchDimX;
            end
                
            % determine right boundary            
            if tileIdx ~= TILE_DIM(2)
                cropXright = IMG_DIM(2) - imageDimX;
            end
                
            % determine upper boundary
            if tileIdy ~= 1
                cropYup = imageDimY + patchDimY;
            end
                
            % determine lower boundary
            if tileIdy ~=TILE_DIM(1)
                cropYdown = IMG_DIM(1) - imageDimY;
            end
            
        end
        
        
        %==================================================================
        % Collect window samples over components in masks
        %   Inputs : 
        %       images          : cell of images (nImages,1)
        %       masks           : cell of masks (nImages,1)
        %       minCompArea     : scaler
        %       minOverlapRatio : scaler between [0-1]
        %   Output :
        %       patchesCell  : cell of patches (1, nImages)
        %==================================================================
        function patchesCell = obtain_patches_within_components(images, masks, patchDim, varargin)

            if nargin<3, error('not enough input arguments!'); end
                                    
            p = inputParser;
            p.addOptional('minCompArea',0, @(x) isscalar(x));
            p.addOptional('minOverlapRatio',.5, @(x) isscalar(x));
            p.addOptional('samplingStride',floor(patchDim / 2), @(x) isscalar(x));    
            p.parse(varargin{:});
            
            minCompArea     = p.Results.minCompArea;
            minOverlapRatio = p.Results.minOverlapRatio;
            samplingStride  = p.Results.samplingStride;
                       
            nImages = numel(images);            
            
            if ~nImages
                patchesCell = [];
                return;
            end
            
            % allocate outputs and helpers
            patchesCell = cell(1, nImages);                                               
                        
            for m=1:nImages
                
                fprintf('Processing %d/%d\n', m, nImages);
                
                % helpers
                IMAGE_DIM  = size(images{m});
                nBands     = size(images{m},3);
                patchArea  = patchDim*patchDim;
                                
                % get region statistics and filter small components
                stats   = regionprops(logical(masks{m}),'BoundingBox','Area');  
                stats(arrayfun(@(x)x.Area<minCompArea, stats)) = [];
                
                % allocate samples
                patches = [];
                
                for j=1:numel(stats)

                    idxStart = max([1,ceil(stats(j).BoundingBox(1))]);
                    idyStart = max([1,ceil(stats(j).BoundingBox(2))]);
                    
                    idxEnd   = min([IMAGE_DIM(2),idxStart + stats(j).BoundingBox(3)]);
                    idyEnd   = min([IMAGE_DIM(1),idyStart + stats(j).BoundingBox(4)]);
                                               
                    for xx = idxStart:samplingStride:(idxEnd-patchDim+1)
                        for yy = idyStart:samplingStride:(idyEnd-patchDim+1)                                                
                            if (sum(sum(masks{m}(yy:yy+patchDim-1,xx:xx+patchDim-1))) / patchArea) > minOverlapRatio
                                patches = [patches,...
                                    reshape(images{m}(yy:yy+patchDim-1,xx:xx+patchDim-1,:), [patchArea*nBands, 1])];
                            end
                        end
                    end                                        

                end 
                
                patchesCell{m} = patches;
                clear patches
            end           
        end
        
        
        %==================================================================
        % Collect window samples over components in masks
        %   Inputs :
        %       data       : data sources for sampling, it can either
        %                    be a 2D matrix <nSignals,nTimepoints>, or a 
        %                    3D matrix <nSignals,nTimepoints,nDataSources>
        %       windowSize : length of window to be sampled
        %       nSamples   : number of samples to be sampled
        %       invalidIdx : a vector at the length of nTimepoints, which
        %                    will be used to discard sampled windows that 
        %                    overlap with invalidIdx position is 1
        %   Output :
        %       temporalWindows : 2D or 3D sampled windows
        %==================================================================
        function temporalWindows = sample_temporal_windows_1D(data, windowSize, nSamples, invalidIdx)
            tic
            % helpers
            nSignals     = size(data,1);
            nTimesteps   = size(data,2);
            nDataSources = size(data,3);
            
            temporalWindows = zeros(windowSize, nSamples, nDataSources); % pre-allocation
            
            for ss=1:nDataSources
                for ii=1:nSamples
                    if (mod(ii,10000) == 0), fprintf('Data source %d/%d ,Extracting window: %d / %d\n', ss, nDataSources, ii, nSamples); end

                    v = random('unid', nSignals);

                    while(true) % crucial part, continue until valid sample
                        r = random('unid', nTimesteps - windowSize + 1);
                        if sum(invalidIdx(r:r+windowSize-1))==0
                            break;
                        end
                    end

                    temporalWindows(:,ii,ss) = data(v,r:r+windowSize-1,ss);
                end
            end
            toc
        end
        
        
        %==================================================================
        % Collect window samples over components in masks, fast version
        %   Inputs :
        %       data       : data sources for sampling, it can either
        %                    be a 2D matrix <nSignals,nTimepoints>, or a 
        %                    3D matrix <nSignals,nTimepoints,nDataSources>
        %       windowSize : length of window to be sampled
        %       nSamples   : number of samples to be sampled
        %       invalidIdx : a vector at the length of nTimepoints, which
        %                    will be used to discard sampled windows that 
        %                    overlap with invalidIdx position is 1
        %   Output :
        %       temporalWindows : 2D or 3D sampled windows
        %==================================================================
        function temporalWindows = sample_temporal_windows_1D_fast(data, windowSize, nSamples, invalidIdx)
            tic
            
            % remove invalid samples
            data(:,invalidIdx) = [];
            
            % helpers            
            nSignals     = size(data,1);
            nTimesteps   = size(data,2);
            nDataSources = size(data,3);
            
            temporalWindows = zeros(windowSize, nSamples, nDataSources); % pre-allocation
            
            for ss=1:nDataSources
                for ii=1:nSamples
                    if (mod(ii,10000) == 0), fprintf('Data source %d/%d ,Extracting window: %d / %d\n', ss, nDataSources, ii, nSamples); end
                    v = random('unid', nSignals);
                    r = random('unid', nTimesteps - windowSize + 1);
                    temporalWindows(:,ii,ss) = data(v,r:r+windowSize-1,ss);
                end
            end
            toc
        end
        
        
        %==================================================================
        % Collect window samples over components in masks / TODO complete
        %   Inputs :
        %       data       : data sources for sampling, it can either
        %                    be a 2D matrix <nSignals,nTimepoints>, or a 
        %                    3D matrix <nSignals,nTimepoints,nDataSources>
        %       windowSize : length of window to be sampled
        %       nSamples   : number of samples to be sampled
        %       invalidIdx : a vector at the length of nTimepoints, which
        %                    will be used to discard sampled windows that 
        %                    overlap with invalidIdx position is 1
        %   Output :
        %       temporalWindows : 2D or 3D sampled windows
        %==================================================================
        function samples2D = sample_spatial_volumes_2D(data, windowSize, nSamples, invalidIdx, xyz)
        
            
            % window size must be an odd number to center the window
            assert(mod(windowSize-1,2)==0,'Window size must be an odd number!');
            borderSize = (windowSize-1)/2;            
            
            fprintf('Calculating valid voxels...\n');
            
            % calculate helpers
            [boxedVolume, validIdx, coords] = ConvUtils.bound_volume(xyz);
                                                                          
            % mark valid voxels, by a sliding window of 1 
            blkfun = @(block_struct)all(block_struct.data(:)); 
            validVoxels = arrayfun(@(x)blockproc(boxedVolume(:,:,x), [1, 1], blkfun,'BorderSize',[borderSize,borderSize],'TrimBorder',false),...
                                        1:size(boxedVolume,3),'UniformOutput',false);                                
            % convert to a 3d matrix
            validVolume = reshape(cell2mat(validVoxels),size(boxedVolume));
            volumeIdx   = find(validVolume);
            volumeSize  = size(validVolume);      
            nValid      = numel(volumeIdx);
            
            nnList = cell(nValid,1);
            
            % collect all the valid samples
            for i=1:nValid                
                nnList{i} = ConvUtils.get_patch_idx_on_2DSlice(volumeIdx(i), volumeSize, borderSize, validIdx);
            end
            
            fprintf('Collecting 2d samples...\n');
            
            % randomly choose within collected set
            samples2D = zeros(nSamples,numel(nnList{1}));
            data(:,invalidIdx)=[];
            for i=1:nSamples/2
                
                t  = random('unid', size(data,2));                
                r1 = random('unid', numel(nnList));
                r2 = random('unid', numel(nnList));
                
                samples2D((i-1)*2+1,:) = data(nnList{r1}(:),t);
                samples2D((i-1)*2+2,:) = data(nnList{r2}(:),t);
                
            end            
        end
        
        
        %==================================================================
        % Collect window samples over components in masks
        %   Inputs :
        %       data       : data sources for sampling, it can either
        %                    be a 2D matrix <nSignals,nTimepoints>, or a 
        %                    3D matrix <nSignals,nTimepoints,nDataSources>
        %       windowSize : length of window to be sampled
        %       nSamples   : number of samples to be sampled
        %       invalidIdx : a vector at the length of nTimepoints, which
        %                    will be used to discard sampled windows that 
        %                    overlap with invalidIdx position is 1
        %   Output :
        %       temporalWindows : 2D or 3D sampled windows
        %==================================================================
        function spatialWindows = sample_spatial_windows_1D(data, windowSize, nSamples, invalidIdx)
            tic
            
            % eliminate invalid data
            data(:,invalidIdx) = [];
            
            % helpers
            nSignals     = size(data,1);
            nTimesteps   = size(data,2);
            nDataSources = size(data,3);
            
            spatialWindows = zeros(windowSize, nSamples, nDataSources); % pre-allocation
                                    
            for ss=1:nDataSources
                for ii=1:nSamples
                    if (mod(ii,10000) == 0), fprintf('Data source %d/%d ,Extracting window: %d / %d\n', ss, nDataSources, ii, nSamples); end

                    v = random('unid', nSignals - windowSize + 1);
                    r = random('unid', nTimesteps);

                    spatialWindows(:,ii,ss) = data(v:v+windowSize-1,r,ss);
                end
            end
            toc
        end
        
        
    end
end

