classdef Edge
%
% orhanf

    methods(Static)
        
        function [outMask lineStats] = run_lsd_on_gray(input)
            outMask = zeros(size(input,1),size(input,2));
            if size(input,3)==1
                
                lsdPath = 'D:\workspace\6B_phase2\core\lsd\';
                density = 0.7;   % [0,1], default value 0.7
                scale   = 2;     % [0,inf], default value 0.8
                
                output = round( lsd(input,'density_th',density,'scale',scale, 'lsdPath', lsdPath )+1);
    
                p1=[];
                p2=[];
                for j = 1:size(output,1)
                    % length of line
                    output(j,6) = (sqrt( ((output(j,3)- output(j,1))^2) + ((output(j,4)- output(j,2))^2)  ));
                    % slope of line
                    output(j,7) =(output(j,3)- output(j,1)) /  (output(j,4)- output(j,2));
                    % angle of line
                    output(j,8) = atand( (output(j,3)- output(j,1)) /  (output(j,4)- output(j,2)));
                    p1=[p1;output(j,2), output(j,1)];%creating for drawline()
                    p2=[p2;output(j,4), output(j,3)];%creating for drawline()
                end

                [ind labels]=drawline(p2,p1,size(outMask));
                outMask(ind)=labels;
                lineStats = output;                
                
            end
        end
        
        
        % aaa
        function outMask = draw_given_angleRange(img, points ,angles, range)
            
            idx = (angles > min(range(:)) & angles < max(range(:)));
            
            p1 = [points(idx,2), points(idx,1)];
            p2 = [points(idx,4), points(idx,3)];
            
            outMask = zeros(size(img,1),size(img,2));
            [ind labels] = drawline(p2,p1,size(outMask));
            outMask(ind) = labels;
        end        
        
        
        % 
        function pts = filter_pts_with_mask(pts, mask)
            
            tic
            
            nRows  = size(mask,1);            
            
            outPts = pts(:,[2,1,4,3]);
            
            linIdx = outPts(:,[2,4]) .* nRows + outPts(:,[1,3]);
            
            intIdx = ismember(linIdx(:),find(mask(:)));
            
            midIdx = length(intIdx)/2;
            
            refIdx = or(intIdx(1:midIdx), intIdx(midIdx+1:end));
            
            pts = pts(refIdx,:);
            
            toc
            
        end
        
        
        %
        function Gabor = get_gabor_sin()
            Gabor = [];
        end
        
        
        
    end



end

