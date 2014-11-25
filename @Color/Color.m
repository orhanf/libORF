classdef Color
    % 
    % orhanf

    methods(Static)
       
        
        % Power-Law Transformation                
        function outImg = apply_power_law(img ,gamma, constant)
                
            img = im2double(img);
            
            if nargin < 2
                constant = 1;            
                gamma    = 0.05;
            end
            
            outImg = constant .* (img .^ gamma);
                        
        end
        
        
        % Convert 3 band image to gray level image mean subtraction
        function outImg = convert_3b2gray1(img)
            outImg = double(img(:,:,1)-.5*img(:,:,2)-.5*img(:,:,3));
            outImg = outImg-mean(outImg(:));            
        end
        
        
        % Convert 3 band image to gray level image mean subtraction and
        % division by std
        function outImg = convert_3b2gray2(img)
            outImg = double(img(:,:,1)-.5*img(:,:,2)-.5*img(:,:,3));
            outImg = (outImg-mean(outImg(:))) ./ (std(outImg(:)));            
        end
        
        
        % Convert 3 band image to gray level image mean subtraction and
        % division by max-min
        function outImg = convert_3b2gray3(img)
            outImg = double(img(:,:,1)-.5*img(:,:,2)-.5*img(:,:,3));
            outImg = (outImg-mean(outImg(:))) ./ (max(outImg(:))-min(outImg(:)));            
        end
        
        
        % as the name refers - adapted from applyRadTransform
        function im8 = convert_16bit_img_to_8bit(im16)
            nBands = size(im16,3);
            im8    = zeros(size(im16));
            for i=1:nBands
                maxVal = double(max(max(im16(:,:,i))));
                im8(:,:,i) = round(double(im16(:,:,i))./maxVal*255);
            end
            im8 = uint8(im8);
        end
        
        % as the name refers - adapted from applyRadTransform
        function strechedImg = stretch_img(img)            
            nBands = size(img,3);
            strechedImg = uint8(img);
            for i=1:nBands
                strechedImg(:,:,i) = imadjust(img(:,:,i),stretchlim(img(:,:,i)));
            end
        end
        
        
    end
    

end

