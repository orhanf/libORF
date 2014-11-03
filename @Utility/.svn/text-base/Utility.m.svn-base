classdef Utility
%==========================================================================
% Utility Class with static member functions, mostly used for image
% processing utilities & shortcuts & validation checks etc. Also, some fMRI
% volumetric data conversion subroutines provided.
%       Do not instantiate!
%
% orhanf - (c) 2012 Orhan FIRAT <orhan.firat@ceng.metu.edu.tr>
%==========================================================================
    
    
    methods(Static)
        
        %==================================================================
        % overlay given detection mask onto image with specified band,
        % keeps other bands intact.
        %==================================================================
        function img = overlay_img(img, detection, band)
            
            [nRow1 nCol1] = Utility.get_rows_cols(img);
            [nRow2 nCol2] = Utility.get_rows_cols(detection);
            
            if sum( [(nRow1-nRow2),(nCol1-nCol2) ]) ~= 0
                errordlg('Input image and mask are not in the same size.', 'Input error');
                return;
            end
            
            oBand = img(:,:,band);
            oBand(detection) = 255;
            
            img(:,:,band) = oBand;
            
        end
        
        
        %==================================================================
        % draws given detection mask onto image with specified band but
        % setting surplus bands to zero.
        %==================================================================
        function img = overdraw_img(img, detection, band)
            
            idx = [255 0 0; 0 255 0; 0 0 255];
            
            mask = detection;
            
            [nRow1 nCol1] = Utility.get_rows_cols(img);
            [nRow2 nCol2] = Utility.get_rows_cols(mask);
            
            if sum( [(nRow1-nRow2),(nCol1-nCol2) ]) ~= 0
                errordlg('Input image and mask are not in the same size.', 'Input error');
                return;
            end
            
            rBand       = img(:,:,1);  gBand       = img(:,:,2);  bBand       = img(:,:,3);
            rBand(mask) = idx(band,1); gBand(mask) = idx(band,2); bBand(mask) = idx(band,3);
            img(:,:,1)  = rBand;       img(:,:,2)  = gBand;       img(:,:,3)  = bBand;
            
        end
        
        
        %==================================================================
        % draws given point (2xn) coordinates onto the image opened before
        % hand and passed as its figure handle.
        %==================================================================
        function overdraw_points_to_figure(figHandle, points, color)
            if ~isempty(figHandle)
                hold on;
                plot(figHandle,points(1,:),points(2,:),'x','color',color);
                hold off;
            end
        end
        
        
        %==================================================================
        % draws given point (2xn) coordinates onto binary image by setting
        % corresponding coordinate to one.
        %==================================================================
        function bwimg = overdraw_points_to_bw(bwimg, points)
            bwimg(points) = 1;
            bwimg = imdilate(bwimg,strel('diamond',2));
        end
        
        
        %==================================================================
        % as the name refers.
        %==================================================================
        function show_image(img)
            if (nargin>0)
                figure,imshow(Utility.correct_img(img));
            end
        end
        
        
        %==================================================================
        % as the name refers.
        %==================================================================
        function [nRows nCols] = get_rows_cols(data)
            nRows = size(data,1);
            nCols = size(data,2);
        end
        
        
        %==================================================================
        % corrects image by converting it into 3 band and setting bit
        % depth to 8bit if image is a 4 band - 16bit.
        %==================================================================
        function img = correct_img(img)
            if Utility.get_nBands(img)>3
                img = (convertTo3Bands(img));
            end
            if isa(img,'uint16')
                img = applyRadTransform(img);
            end
        end
        
        
        %==================================================================
        % as the name refers.
        %==================================================================
        function nBands = get_nBands(img)
            try
                nBands = size(img,3);
            catch err
                nBands = 1;
            end
        end
        
        
        %==================================================================
        % as the name refers
        %==================================================================
        function outImg = map_values_to_image(vals, linIdx, sizeImg)
            outImg = zeros(sizeImg);
            outImg(linIdx) = vals;
        end
        
        
        %==================================================================
        % as the name refers
        %==================================================================
        function filteredSegIdx = filter_segs_with_mask(labelMap, responseMask , segIdx, th)
            filteredSegIdx = [];
            for i=1:length(segIdx)
                
                linearInd = find(labelMap==segIdx(i));
                
                overlayArea = sum(responseMask(linearInd));
                segmentArea = length(linearInd);
                
                if overlayArea/segmentArea > th
                    filteredSegIdx = [filteredSegIdx, segIdx(i)];
                end
            end
        end
        
        
        %==================================================================
        % Obtain gt masks by traversing directories
        %==================================================================
        function [masks dirs] = get_gt_masks_from_disk(dirname, gtext, scale, dirs)
            
            if nargin < 4 || ( nargin == 4 && isempty(dirs) )
                dirs = dir(dirname);
                dirs(~arrayfun(@(x)(exist([ dirname '/' x.name '/' x.name '_' gtext ],'file')),dirs)) = [];
            end
            
            numImgs  = numel(dirs);
            masks    = cell(numImgs,1);
            scaleOrj = scale;
            
            for i=1:numImgs
                
                try
                    filename = dirs(i).name;
                    
                    if strcmp(filename(1),'I'), scale = scaleOrj * 2; else scale = scaleOrj;  end
                    
                    mask = imread([dirname '/' filename '/' filename '_' gtext]);
                    
                    masks{i}  = imresize(mask,scale);
                    
                catch err
                    disp(err.message);
                end
            end
        end
        
        
        %==================================================================
        % Obtain gt masks by traversing directories without scaling
        %==================================================================
        function [masks dirs] = get_gt_masks_from_disk_wo_scaling(dirname, gtext, dirs)
            
            if nargin < 3 || ( nargin == 3 && isempty(dirs) )
                dirs = dir(dirname);
                dirs(~arrayfun(@(x)(exist([ dirname '/' x.name '/' x.name '_' gtext ],'file')),dirs)) = [];
            end
            
            numImgs  = numel(dirs);
            masks    = cell(numImgs,1);
            
            for i=1:numImgs
                
                try
                    filename = dirs(i).name;
                    masks{i} = imread([dirname '/' filename '/' filename '_' gtext]);
                    
                catch err
                    disp(err.message);
                end
            end
        end
        
        
        %==================================================================
        % Obtain gt directories by traversing given root directory
        %==================================================================
        function dirs = get_gt_dirs(dirname, gtext)
            dirs = dir(dirname);
            try
                dirs(~arrayfun(@(x)(exist([ dirname '/' x.name '/' x.name '_' gtext ],'file')),dirs)) = [];
            catch err
                disp(err.message);
            end
        end
        
        
        %==================================================================
        % Draws black grid structure over image - for visualization
        %==================================================================
        function newImg = draw_grid_on_img(img, gridSize)
            
            newImg = img;
            
            for i=1:size(img,3)
                % draw horizonral lines
                newImg(1:gridSize(1):end,:,i) = 0;
                
                % draw vertical lines
                newImg(:,1:gridSize(2):end,i) = 0;
            end
        end
        
        
        %==================================================================
        % Converts volumetric fMRI data to VT format, VT format means
        % <#voxels x #timesteps>, size variables returned optionally
        %==================================================================
        function [dataVT, volSizes] = convert_volume_to_VT(data4D)
            dataVT   = [];
            volSizes = [];
            if ~isempty(data4D)
                % get sizes
                [sizeX, sizeY, sizeZ, nTimesteps] = size(data4D);
                nVoxels = sizeX*sizeY*sizeZ;
                
                % pre-allocate result
                dataVT  = zeros(nVoxels,nTimesteps);
                for t = 1:nTimesteps
                    dataVT(:,t) = reshape(data4D(:,:,:,t),1,[]);
                end
                
                % make the xyz coordinates
                [coorX, coorY, coorZ] = ind2sub([sizeX sizeY sizeZ],1:nVoxels);
                
                % set output fields
                volSizes.sizeX = sizeX;
                volSizes.sizeY = sizeY;
                volSizes.sizeZ = sizeZ;
                volSizes.coorX = coorX;
                volSizes.coorY = coorY;
                volSizes.coorZ = coorZ;
                volSizes.nVoxels    = nVoxels;
                volSizes.nTimesteps = nTimesteps;
            end
        end
        
        %==================================================================
        % Load specified fields from the specified file
        %
        %   Inputs :
        %       fileToLoad : Path for the file to load
        %       fieldNamesToLoad : cell array of fieldnames to acquire
        %   Outputs:
        %       variable size output for fields
        %==================================================================
        function [s, varargout] = load_fields( fileToLoad, fieldNamesToLoad )
            
            s = [];
            varargout = cell(0,0);
            
            if exist(fileToLoad,'file')
                
                % load data from disk
                loadedData = load(fileToLoad);
                                
                if nargin==2 && ischar(fieldNamesToLoad)
                    s = loadedData.(fieldNamesToLoad);
                    return;
                end
                
                if nargin < 2 || sum(cellfun(@(x)isempty(x),fieldNamesToLoad)) > 0
                    s = loadedData;
                    return;
                end                                
                
                for i=1:numel(fieldNamesToLoad)
                    if isfield(loadedData,fieldNamesToLoad{i})
                        varargout{i} = loadedData.(fieldNamesToLoad{i});
                    else
                        varargout{i} = [];
                    end
                end
                
                s = varargout{1};
                varargout(1) = [];
                
                if nargout > numel(varargout) + 1
                    varargout = [varargout; cell(nargout-numel(varargout)-1,1)];
                end
            else
                warning(['file does not exist "' fileToLoad '"' ]);
            end
            
        end
        
        
        %==================================================================
        % Converts a vector of labels into one-of-k encoding
        %==================================================================
        function oneOfK = convert_to_one_of_k_encoding(vec2convert)
            oneOfK = [];
            if ~isempty(vec2convert)
                maxEl = max(vec2convert);
                M = eye(maxEl);
                oneOfK = M(:,vec2convert);
            end
        end
        
    end
    
end
