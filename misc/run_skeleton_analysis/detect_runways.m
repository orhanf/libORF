%==================================================================== >
%> @brief This function initiates runway detection routine on the skeleton
%> of the airfield mask.
%>
%> @author orhanf&okant
%>
%> Change                                      Date       Performed By 
%>
%> @param attr Struct that includes all image attributes
%> @param airfield_mask Initial binary airfield mask
%>
%> @retval runway_mask Initial airfield mask 
%> @retval runways_idx Cell containing runway masks as its elements.Each
%>      runway mask is kept as linear indexes.
%> @retval runways_prop Cell containing runway propersties as its elements.
%> 
%> @example 
%> [initial_mask additional_mask] = extract_overlapping_cluster_with_segs(labels_sc, labels_mec);
%> airfield_mask = run_select_final_mask(initial_mask, additional_mask);
%> [runway_mask runways_idx runways_prop]=
%>              detect_runways(attr,airfield_mask);
%====================================================================
function [runway_mask runways_idx runways_prop]= detect_runways(attr,airfield_mask)
%DETECT_RUNWAYS Summary of this function goes here
%   Detailed explanation goes here
%
%   TODO : add max length boundary (avoid too long linear roads)
%   TODO : dont expect resolution as a parameter
%
% orhanf-okantt
%%
global DEBUG_FLAG;
% expert rules
runway_width = double(45);
runway_rad = runway_width / 2;
runway_minlen = 750;
imageres = attr.resolution; % TODO read it from attr with geotiffinfo
runway_maxlen = 5000; % TODO optimize

% boundary image
edgeIm = bwmorph(airfield_mask,'remove');

% run anaskel
[skg, rad] = skeleton(airfield_mask);
rad = sqrt(rad);

% generate initial candidate mask for runway
pist_cand = (skg > 5) .* (rad > runway_rad / imageres - 3) & (rad < runway_rad / imageres + 8);

% apply hough
[H,T,R] = hough(pist_cand,'RhoResolution',0.9,'Theta',-90:0.1:89.5);
P  = houghpeaks(H,10,'threshold',0.5 * ceil(max(H(:))));
lines = houghlines(pist_cand,T,R,P,'FillGap',70,'MinLength',7);

% cell to be saved, each element will be a runway linear idx
runways_idx = cell(1,1);
runway_mask = false(size(airfield_mask));
ctr=1;

% cell containing runway properties
runways_prop = cell(1,1);

% for each line result
for k = 1:length(lines)
    % obtain line
    xy = [lines(k).point1; lines(k).point2];
    % calculte length of the line for further processing
    len = norm(lines(k).point1 - lines(k).point2);
    
    if or((runway_mask(lines(k).point1(2), lines(k).point1(1))),(runway_mask(lines(k).point2(2), lines(k).point2(1))))
        continue;
    end
    
    % only draw if length is more than runway minimum expected length
    if len > runway_minlen / imageres
        runway_region = false(size(attr.im,1), size(attr.im,2));
        [ind labels] = drawline([lines(k).point1(2) lines(k).point1(1)],[lines(k).point2(2) lines(k).point2(1)],size(runway_region));
        runway_region(ind) = 1;
        
        % visualise runway
        if DEBUG_FLAG
            figure,imshow(runway_region); hold on;
            plot(xy(:,1),xy(:,2),'LineWidth',2,'Color','white');
            hold off;
        end
        
        % analyse line, estimate width orientation and dilate to
        % generate mask for that particular runway, save it into the
        % detection result files both as a cell and binary mask
        [dists, int_prox] = liner(edgeIm, [xy(2,2) xy(2,1)],[xy(1,2) xy(1,1)] ); % get distances and intersections in proximity
        
        runway_orientations = regionprops(logical(runway_region),'Orientation');
        left_ptile = ([prctile(dists(:,1), 20) prctile(dists(:,1), 80)]);
        left_avg = mean(dists((dists(:,1) >= left_ptile(1)) & (dists(:,1) <= left_ptile(2)), 1));
        right_ptile = ([prctile(dists(:,2), 20) prctile(dists(:,2), 80)]);
        right_avg = mean(dists((dists(:,2) >= right_ptile(1)) & (dists(:,2) <= right_ptile(2)), 2));
        
        runway_region = bwmorph(imdilate(runway_region, strel('line',round(left_avg+right_avg),runway_orientations.Orientation+90)),'fill');
        runway_intersections = bwmorph(imdilate(runway_region, strel('line',2.5*round(left_avg+right_avg),runway_orientations.Orientation+90)),'fill');
        runway_intersections = runway_intersections - bwmorph(imdilate(runway_region, strel('line',1.4*round(left_avg+right_avg),runway_orientations.Orientation+90)),'fill');
        
        int_mask = false(size(runway_region));
        int_mask(sub2ind(size(int_mask), int_prox(:,1), int_prox(:,2))) = true;
        runway_intersections = runway_intersections & int_mask;
            % TODO: find a way to return runway coordinates as parameter
%             figure, imshow(runway_intersections);
%             title([ 'Feature ' attr.name,' ',num2str(sum(runway_intersections(:))/norm(xy(1,:)-xy(2,:))) ])
%             fprintf(1,'%s intersection points: %g\n',attr.name,sum(runway_intersections(:))/norm(xy(1,:)-xy(2,:)));
            if sum(runway_intersections(:))/norm(xy(1,:)-xy(2,:)) > 0.1
%                 fprintf(1,'not taken\n');
%                 imwrite(runway_region,['outputs\detection_results\' attr.name '\' attr.name '_runway_' num2str(ctr) 'nottaken.png'],'png');
            else
%                 output_file = fopen(['outputs\detection_results\' attr.name '\' attr.name '_runway_' num2str(ctr) '.txt'], 'w');
                mid_point = (xy(1,:)+xy(2,:)) / 2;
                % file print order: start pt, end pt, width, length, orientation
                runways_prop{ctr}.mid_point_x = mid_point(1);
                runways_prop{ctr}.mid_point_y = mid_point(2);
                runways_prop{ctr}.width = (left_avg+right_avg)*imageres;
                runways_prop{ctr}.length = (norm(xy(1,:)-xy(2,:)))*imageres;
                runways_prop{ctr}.orientation = 90-runway_orientations.Orientation;
%                 fprintf(output_file, '%g %g %g %g %g\n', mid_point(1), mid_point(2),...
%                     (left_avg+right_avg)*imageres, (norm(xy(1,:)-xy(2,:)))*imageres, ...
%                     90-runway_orientations.Orientation);
%                 imwrite(runway_region,['outputs\detection_results\' attr.name '\' attr.name '_runway_' num2str(ctr) '.png'],'png');
                runways_idx{ctr} = find(runway_region);
                runway_mask = or(runway_mask,runway_region);
                ctr=ctr+1;
%                 fclose(output_file);
            end
    end
end

% if ~isempty(runways_idx{1})
%     save(['outputs\detection_results\' attr.name '\' attr.name '_runways_idx.mat'],'runways_idx');
% end

end

