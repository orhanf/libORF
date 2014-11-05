%====================================================================
%> @brief This function detects the dispersal areas from the given masks.
%>
%> @author orhanf
%>
%> Change                                      Date       Performed By 
%>
%> @param airfield_mask Mask containing whole airfield region
%> @param runway_mask Mask containing detected runways
%> @param new_taxiroute_mask Mask containing detected taxiroutes
%>
%> @retval dispersal_points Binary mask of dispersal area points 
%> 
%> @example
%> airfield_mask = attr.airfield_mask;
%> [runway_mask runways_idx runways_prop]= detect_runways(attr,airfield_mask);
%> [taxiroute_mask taxiroute_surplus_mask] = detect_taxiroutes(attr,airfield_mask,runway_mask);
%> [parkarea_mask parkarea_surplus_mask] = detect_parkareas(attr,airfield_mask,...
%>                                              runway_mask,taxiroute_mask,taxiroute_surplus_mask, sc_labels, mec_labels);
%> [new_parkarea_mask new_taxiroute_mask] = rectify_detections(airfield_mask,runway_mask,...
%>                                              taxiroute_mask,taxiroute_surplus_mask, parkarea_mask);

%====================================================================
function dispersal_points = detect_dispersals(airfield_mask,runway_mask,new_taxiroute_mask)
%%
global DEBUG_FLAG;

[skg,rad] = skeleton(airfield_mask);

[dmap, exy, jxy] = anaskel(bwmorph(skg>5,'skel',Inf));
da_points = findDispersalAreas( new_taxiroute_mask , 4 , exy, jxy, runway_mask);

if DEBUG_FLAG
    figure,imshow(new_taxiroute_mask); hold on;
    if ~isempty(da_points)
        plot(da_points(:,1),da_points(:,2),'x','LineWidth',2,'Color','red');
    end
    hold off;
end
dispersal_points = da_points;
%     figure,imshow(applyRadTransform(img(:,:,1:3)));

end