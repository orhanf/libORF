classdef ConvUtils
    %==========================================================================
    %
    % orhanf - (c) 2014 Orhan FIRAT <orhan.firat@ceng.metu.edu.tr>
    %==========================================================================
    
    
    methods(Static)
        
        %==================================================================
        %
        %==================================================================
        function [boxedCoords, validIdx] = bound_volume(coords)
            
            % Coordinates must be in 3 dimensional space
            assert(ismember(3,size(coords)),'The coordinate system must be 3-dimensional!');
            
            % Transpose data if necessary
            if size(coords,2) ~= 3,  coords = coords'; end
            
            mins = min(coords);
            maxs = max(coords);
            
            boxedCoords = false(maxs-mins+1);            
            validIdx    = sub2ind(size(boxedCoords),coords(:,1)-mins(1)+1,...
                                                    coords(:,2)-mins(2)+1,...
                                                    coords(:,3)-mins(3)+1);            
            
            boxedCoords(validIdx) = true;
            
%             ind2sub
            
        end
        
        
        %==================================================================
        %   Wrapper for maxoutFprop mex, for details, refer maxoutFprop.c 
        %==================================================================
        function [a, b] = maxout_Fprop(data,layerOpt)
            [a,b] = maxoutFprop(data,...
                                layerOpt.poolSize,...
                                layerOpt.stride,...
                                layerOpt.isRandom,...
                                layerOpt.isDebug);
        end
        
    end
end

