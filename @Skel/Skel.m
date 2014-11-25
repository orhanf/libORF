classdef Skel
% Skeletonization utility class with static member functions
%       Do not instantiate!
%
% orhanf
    
    methods(Static)
        
        
        % compute skeleton of given binary mask
        function [skg,rad] = compute_skeleton(img)            
            [skg,rad] = skeleton(img);                                
        end
                    
        
        % compute anatomical skeleton with given threshold value(s)
        function anaskels = compute_anaskel(skg,th)                        
            len = length(th);
            anaskels = struct('th',[],'bwimg',[],'exy',[],'jxy',[]);
            for i=1:len
                anaskels(i).bwimg = bwmorph(skg>th(i),'skel',Inf);
                [~, anaskels(i).exy, anaskels(i).jxy] = anaskel(anaskels(i).bwimg);
                anaskels(i).th = th(i);
            end            
        end
                               
                
        % piles up the skeleton result(s) on given @Img object
        function skgStack = generate_skel_stack(obj)
            
            anaskels = obj.anaskels;
            
            nLayers  = numel(anaskels);            
            skgStack = zeros(size(anaskels(1).bwimg,1),size(anaskels(1).bwimg,2));
            skgStack(:) = nLayers + 1;            
            
            runway_mask = obj.runwayMask;
            
            for i=1:nLayers
                skgStack = imsubtract(skgStack,double(anaskels(i).bwimg));
            end            
            skgStack(skgStack==(nLayers+1)) = 0;      
            skgStack = immultiply(skgStack,~logical(runway_mask));
        end
       
        
        % piles up the circle result(s) on given @Img object
        function circStack = generate_circ_stack(obj,res)
            
            airfield_mask   = obj.airfieldMask;
            runway_mask     = obj.runwayMask;
            nLayers         = numel(obj.anaskels);  
            
            circStack = zeros(size(airfield_mask,1),size(airfield_mask,2));
            
            for i=1:nLayers
            
                end_points  = obj.anaskels(i).exy;
                junc_points = obj.anaskels(i).jxy;
                
                center_points  = findDispersalAreas( airfield_mask , res , end_points, junc_points, runway_mask);
                
                if isempty(center_points), continue; end
                
                center_points(:,3) = sub2ind(size(airfield_mask),center_points(:,2),center_points(:,1));
                
                circStack = imadd(circStack,...
                    double(Utility.overdraw_points_to_bw(false(size(airfield_mask,1),size(airfield_mask,2)), center_points(:,3))));
                
            end            
        end
        
        
        % as the name refers
        function probMap = pass_gaussian_filter(map, mu, sigma)        
            F = fspecial('gaussian',mu,sigma);
            probMap = imfilter(map,F,'same');
        end
        
        
        % as the name refers
        function resMap = combine_prob_maps(map1,map2)
            resMap = mat2gray(imadd(mat2gray(map1),mat2gray(map2)));
        end
        
        
    end
        
end