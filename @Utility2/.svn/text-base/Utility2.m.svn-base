classdef Utility2
%==========================================================================
%   Static Utility class for displaying, may be obsolete
%
% orhanf - (c) 2012 Orhan FIRAT <orhan.firat@ceng.metu.edu.tr>
%==========================================================================
    
    properties(Constant = true)
        ALPHA  =  0.01;
        N_ITER =  1500;
    end
    
    
    methods(Static)
        
        
        function [h, array] = display_images(A)
            try
                addpath(genpath([pwd '/misc/']));
                figure;
                [h, array] = display_network(A);
            catch err
                disp(['ERROR while displaying representation:', err.message]);
            end
        end;
        
        
        function display_color_images(A)
            try
                addpath(genpath([pwd '/misc/']));
                figure;
                displayColorNetwork(A);
            catch err
                disp(['ERROR while displaying representation:', err.message]);
            end
        end
        
        
        function handle = display_detections(testFilename, detections, nBands)
            try
                assert(sum(isfield(detections,{'heatMap','detectionArea','detectionMask'})) == 3,...
                    'ERROR: detections struct is inappropriate! ');
                
                I = applyRadTransform(imread(testFilename));
                
                if nBands == 3 || nBands == 4
                    I = I(:,:,[3,2,1]);
                elseif nBands == 1
                    I = rgb2gray(I(:,:,[3,2,4]));
                else
                    error('Number of bands must be 1,3 or 4!');
                end
                
                handle = figure('units','normalized','outerposition',[0 0 1 1],'color','w');
                subplot(2,2,1),imshow(I),freezeColors, axis image;
                    title('Input Image'); 
                subplot(2,2,2),imagesc(detections.heatMap), axis image,colormap('jet'), freezeColors;
                    title('Detection Heat Map'); 
                subplot(2,2,3),imagesc(detections.detectionArea), axis image,colormap('jet'), freezeColors;
                    title('Detection Areas');             
                subplot(2,2,4),imagesc(detections.detectionMask), axis image,colormap('jet'), freezeColors;
                    title('Detection Mask');     
                    
            catch err
                handle = [];
                disp(err.message);
            end
        end
        
        
        function generatedName = get_temp_name()
            [nouse1, generatedName, nouse2]  = fileparts(tempname);
        end
        
        
        function mask = getMaskWithShape(testFilename, env_shp_name)
            try
                assert(exist(env_shp_name,'file')>0, ...
                    ['Shape file does not exist for image [' Utility2.getDirname(testFilename) ']']);
                mask = logical(shp2mask(testFilename,env_shp_name));
            catch err
                mask = [];
                disp(err.message);
            end
        end
        
        
        function dirname = getDirname(testFilename)
            try
                [~, name, ~] = fileparts(testFilename);
                dirname = name(1:max([regexp(name,'_MS'),regexp(name,'_Pansharp')])-1);
            catch err
                dirname = '';
                disp(['ERROR while parsing filename:' ,err.message])
            end
        end
        
        
        
        function dirContent = getDirContents(dirName)
            dirContent = dir(dirName);
            dirContent(arrayfun(@(x)(strcmp(x.name,'.')||(strcmp(x.name,'..'))),dirContent)) = [];
        end
            

        function success = delfun( path,dirname,filename)
            try 
                filename = fullfile(path,dirname,filename);
                if ~exist(filename,'file')
                   error('%s does not exist',filename); 
                end
                delete(filename);
                success = 1;
            catch err
                success = 0;
                disp(err.message);
            end
        end
        
    end
    
    
end

