%====================================================================
%> @brief This function uses Circular Hough to detect dispersal areas
%>Each circular structure tested with given r value(s), circular hough 
%>       threshold parameter is calculated by scaling radious and ground
%>       resolution. For each given radius, calculated candidate center
%>       points are tested for closeness to the end points. Only the circles
%>       near to the end points are accepted as dispersal areas.
%> @author orhanf
%>
%> Change                                      Date       Performed By 
%> Rev 2 : Junction pts excluded from detections
%> Rev 1 : Bug fix
%> Rev 0 : Initial Release
%>
%> @param airfield_mask Mask containing whole airfield region
%> @param res ground resolution of original image
%> @param end_points 2xN matrix of end points calculated from skeleton,
%>                             obtained from Anaskel
%>
%> @retval center_points 2xN matrix of founded dispersal area centers
%> 
%> @example
%> [skg,rad] = skeleton(airfield_mask);
%> [dmap, exy, jxy] = anaskel(bwmorph(skg>5,'skel',Inf));
%> da_points = findDispersalAreas( new_taxiroute_mask , 4 , exy, jxy, runway_mask);

%====================================================================
function  center_points  = findDispersalAreas( airfield_mask , res , end_points, junc_points, runway_mask)
%FINDDISPERSALAREAS 
%   Circular Hough to detect dispersal areas
%   Each circular structure tested with given r value(s), circular hough 
%       threshold parameter is calculated by scaling radious and ground
%       resolution. For each given radius, calculated candidate center
%       points are tested for closeness to the end points. Only the circles
%       near to the end points are accepted as dispersal areas.
% 
%	Inputs :
%       airfield_mask       : regions mask of the whole image, is logical
%       res                 : ground resolution of original image
%       end_points          : 2xN matrix of end points calculated from skeleton,
%                             obtained from Anaskel
% 
%	Output :
%		center_points       : 2xN matrix of founded dispersal area centers

% Author: orhanf & okant
% Revision History:
% Rev 2 : Junction pts excluded from detections
% Rev 1 : Bug fix
% Rev 0 : Initial Release

%% scale with ground resolution to calculate radius and threshold vectors
r= (4/res)*[3 4 5 6 7];

final_x = [];final_y = []; 
x=[];y=[];

% obtain edge image from mask
edgeIm = bwmorph(airfield_mask,'remove');

%% circular hough routine
for j=1:length(r)

    [y0detect,x0detect,Accumulator] = houghcircle(edgeIm,r(j),round(r(j)*pi*0.6));

    % obtain circle center coordinates
    for i=1:length(y0detect)
       if airfield_mask(y0detect(i),x0detect(i))==1
           x =[x x0detect(i)];
           y =[y y0detect(i)];
       end 
    end
end

%% reference mask to look up junction point neighborhoods

junc_mask = zeros(size(airfield_mask));

% mark junction pts
junc_pts_tr = junc_points';
for i=1:size(junc_pts_tr,1)
    junc_mask(junc_pts_tr(i,2),junc_pts_tr(i,1)) = 1;
end

% dilate junction pts
junc_mask = imdilate(junc_mask,strel('disk',3));

%% reference masks to look up end point neighborhoods

r_1mask = zeros(size(airfield_mask));

% mark end points on each ref mask
tr = end_points';
for i=1:size(tr,1)
   r_1mask(tr(i,2),tr(i,1)) = 1; 
end

% radius of selection
% FIXME : why +3
% r1=max(r)+5;
r1 = 5;

r_1mask = imdilate(r_1mask,strel('disk',r1));
r_2mask = imdilate(runway_mask,strel('disk',r1));
% figure,imshow(r_1mask);


%% eliminate points outside the reference masks and near runway

for i=1:size(x,2)
    if r_1mask(y(i),x(i)) == 1 && junc_mask(y(i),x(i)) ~= 1 && r_2mask(y(i),x(i)) == 0
       final_x = [final_x x(i)];
       final_y = [final_y y(i)]; 
    end
end


%% plot dispersal areas
% figure;imshow(airfield_mask);hold on;
% plot(final_x(:),final_y(:),'x','LineWidth',2,'Color','red');

center_points = [final_x' final_y'];

end

