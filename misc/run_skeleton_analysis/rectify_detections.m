%====================================================================
%> @brief This function rectifies the detections with the expert rule that
%> a runway can never be adjacent with a parking area. The adjacent
%> detections of runways and parking areas are further processed to improve
%> and correct detection. Runway detections kept unchanged but parking area
%> detections which are adjacent to the runways are decomposed by opening
%> operation and each subpart is processes with further expert rules.
%>
%> @author orhanf
%>
%> Change                                      Date       Performed By 
%>
%> @param attr Structure containing properties related to the input image
%> @param runway_mask Mask containing detected runways
%> @param taxiroute_mask Mask containing detected taxiroutes
%> @param taxiroute_surplus_mask Mask containing taxiroute candidates
%> @param parkarea_mask Mask containing detected parkareas 
%>
%> @retval new_parkarea_mask Binary mask of rectified park area
%> @retval new_taxiroute_mask Binary mask of rectified taxiroutes
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
function [new_parkarea_mask new_taxiroute_mask] = rectify_detections(airfield_mask,runway_mask,...
                                                taxiroute_mask, taxiroute_surp_mask,parkarea_mask)
%RECTÝFY_DETECTÝONS Summary of this function goes here
%   Detailed explanation goes here
%
% orhanf

    % holy threshold for the min area of parking areas (by heart)
    holy_th = 50;
    holy_th_max_surp_taxi = 50000;
    
    park_lbl = bwlabel(parkarea_mask);
    runway_mask_d = imdilate(runway_mask,strel('disk',1));
    intersections = (unique(runway_mask_d.*park_lbl));intersections(1)=[];
        
    if isempty(intersections)
        new_parkarea_mask = parkarea_mask;
        new_taxiroute_mask = taxiroute_mask;
        return;
    end
    
    taxiroute_mask_d = imdilate(taxiroute_mask,strel('disk',1));
    taxiroute_surp_mask_d = imdilate(taxiroute_surp_mask,strel('disk',1));
        
    new_parkarea_mask = logical(imsubtract(parkarea_mask,ismember(park_lbl,intersections)));
    new_taxiroute_mask = logical(taxiroute_mask);
    
    % for every parking are which intersects with runways
    for i=1:length(intersections)
        
        new_components = imopen(logical(park_lbl==intersections(i)),strel('disk',2));
        new_comp_lbls = bwlabel(new_components);
        
        % for each component of the original parking area
        for j=1:max(max(new_comp_lbls))
            
            tmpcomp = new_comp_lbls==j;
            
            % check if it intersects with runways
            is_adj_with_runway = max(unique(logical(runway_mask_d).*logical(tmpcomp)));
            
            % check if it intersects with taxiroutes
            is_adj_with_taxiroute = max(unique(logical(taxiroute_mask_d).*logical(tmpcomp)));
            
            % check if it intersects with surplus taxiroutes
            is_adj_with_surp_taxiroute = max(unique(logical(taxiroute_surp_mask_d).*logical(tmpcomp)));
            
            if is_adj_with_surp_taxiroute
               taxiroute_surp_lbl = bwlabel(taxiroute_surp_mask);
               taxi_surp_intersect_lbls = unique(immultiply(taxiroute_surp_lbl,imdilate(tmpcomp,strel('disk',2)))); 
               taxi_surp_intersect_lbls(1)=[];
            end
                                    
            % if intersects with both runway and taxiroute mark as
            % taxiroute
            if (is_adj_with_runway == 1) && (is_adj_with_taxiroute == 1)
                if sum(tmpcomp(:)) < holy_th_max_surp_taxi
                    new_taxiroute_mask(tmpcomp(:)) = true;
                else
                    new_parkarea_mask(tmpcomp(:)) = true;
                end
                continue;
            end
            
            % if only intersects with taxiroute 
            if (is_adj_with_runway == 0) && (is_adj_with_taxiroute == 1)
                
                % and below a threshold mark as taxiroute
                if (sum(sum(tmpcomp)))< holy_th
                    new_taxiroute_mask(tmpcomp(:)) = true;
                else % mark as park area
                    new_parkarea_mask(tmpcomp(:)) = true;
                end
                continue;
            end
                 
            % if does not intersect with anything mark as a parking area
            if (is_adj_with_runway == 0) && (is_adj_with_taxiroute == 0)
                new_parkarea_mask(tmpcomp(:)) = true;
                continue;
            end
            
            % if intersect with both runway and surplus taxiroute mask and 
            % add both surplus component and component as taxiroute
            if (is_adj_with_runway == 1) && (is_adj_with_surp_taxiroute == 1)
                new_taxiroute_mask(tmpcomp(:)) = true;
                
                for k=1:length(taxi_surp_intersect_lbls)
                    if sum(tmpcomp(:)) < holy_th_max_surp_taxi
                        new_taxiroute_mask(taxiroute_surp_lbl==(taxi_surp_intersect_lbls(k))) = true;
                    end
                end
                continue;
            end
            
            % if only intersects with runway delete it 
            if (is_adj_with_runway == 1) && ( (is_adj_with_taxiroute)== 0)
               continue; 
            end
            
            
        end
    end
%     disp(['detected occlusion in [' filename ']']);
%     imwrite(new_parkarea_mask,['outputs\detection_results\' filename '\' filename '_parkarea_mask_enhanced.png'],'png');
%     imwrite(new_taxiroute_mask,['outputs\detection_results\' filename '\' filename '_taxiroutes_enhanced.png'],'png'); 
    
end

