%==================================================================== >
%> @brief This function detect taxiroutes by analysing airfield mask and
%> runway mask. Current detections and surplus candidate detections
%> returned as two discrete masks.
%>
%> @author orhanf&okant
%>
%> Change                                      Date       Performed By 
%>
%> @param airfield_mask Initial binary mask of airfield
%> @param runway_mask Binary mask that includes all airfields as ones.
%>
%> @retval taxiroute_mask Binary mask that includes all taxiroutes as ones.
%> @retval taxiroute_surplus_mask Binary mask of taxi-route candidates,
%>      excluding regions that are provided in taxiroute mask
%> 
%> @example 
%> [runway_mask runways_idx runways_prop]= detect_runways(attr,airfield_mask);
%> [taxiroute_mask taxiroute_surplus_mask] = detect_taxiroutes(airfield_mask,runway_mask);
%====================================================================
function [taxiroute_mask taxiroute_surplus_mask] = detect_taxiroutes(airfield_mask,runway_mask)
%DETECT_TAXIROUTES Summary of this function goes here
%   Detailed explanation goes here
%
%   TODO : çok terleme, çok yorulma, girdaplarýnda boðulma, yalnýzlýðýna
%   çok da alýþma
%
% orhanf - okantt

%%
global DEBUG_FLAG;

taxiroute_width = 20;
taxiroute_rad = taxiroute_width / 2;
imageres = 4;

minRad = (taxiroute_rad / imageres)-3;
maxRad = (taxiroute_rad / imageres)+4;

[skg, rad] = skeleton(airfield_mask);
orgRad=rad;
rad = sqrt(rad);

    tr_cand = (skg > 10) & (rad < maxRad) & (rad > minRad);
    tr_cand = tr_cand & ~runway_mask;
    tr_cand = bwmorph(tr_cand, 'thin', Inf);
    tr_cand = bwlabel(tr_cand);
    tr_cand_props = regionprops(logical(tr_cand), 'Area', 'PixelIdxList');
    for j=1:length(tr_cand_props)
        if(tr_cand_props(j).Area < 25 / imageres) % eliminate detections shorter than 100 m
            tr_cand(tr_cand_props(j).PixelIdxList) = 0;
        end
    end
    
    taxiroute_cand_mask=(imsubtract(imdilate(logical(tr_cand),strel('disk',4))&airfield_mask,runway_mask));

    taxiroute_cand_mask=bwlabel(taxiroute_cand_mask==1);
    taxiroute_lbls = unique(immultiply(taxiroute_cand_mask,logical(imdilate(runway_mask,  strel('disk',4)))));    
    taxiroute_lbls(1)=[];
    taxiroute_mask=ismember(taxiroute_cand_mask,taxiroute_lbls);
    taxiroute_surplus_mask = imsubtract(logical(taxiroute_cand_mask),logical(taxiroute_mask));
    if DEBUG_FLAG
        figure,imshow(taxiroute_mask);
        figure,imshow(taxiroute_surplus_mask);
    end
%     imwrite(taxiroute_mask,['outputs\detection_results\' attr.name '\' attr.name '_taxiroutes.png'],'png');
%     imwrite(taxiroute_surplus_mask,['outputs\detection_results\' attr.name '\' attr.name '_taxiroutesurplus.png'],'png');
    
end

