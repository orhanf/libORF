function display_img(obj, img)
%DÝSPLAY_ÝMG Summary of this function goes here
%   Detailed explanation goes here
%
% orhanf
%%
    
    hFigure = figure;
    
    I = obj.data;
    
    if nargin>1    
        imshow(img);
    else
    
        if obj.nBands>3                
            imshow(applyRadTransform(convertTo3Bands(I)));
        else
            imshow(applyRadTransform(I));
        end

    end
    
    obj.hFigure = hFigure;

end

