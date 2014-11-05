%==================================================================== >
%> @brief This function calculates distances from line
%>   given by startpt and endpt to nearest points given by contour image.
%>   This function can be used to calculate width of airfield runway given a
%>   line vector representing runway and a bw contour image representing
%>   contour of area including runway, taxiroute, parking area etc.
%>   This function uses drawline function available at:
%>   http://www.mathworks.com/matlabcentral/fileexchange/15374-drawline
%>
%> @author okant
%>
%> Change                                      Date       Performed By 
%> Rev 3 : Returns positions of all intersections with mask contour
%> Rev 2 : Fixed problem with vertical lines and line segments having no
%> intersection with contour
%> Rev 1 : Two intersection points are taken only if they fall to different
%> sides of original line.
%> Rev 0 : Initial release
%>
%> @param contour - n-by-m bw image indicating contour
%> @param startpt - starting point of line vector with length 2
%> @param endpt - end point of line vector with length 2
%>      
%>  Additional parameters and options:
%>
%> @param 'stepsize' - sets how often the width will be calculated along the
%>       line. Default stepsize = 10, calculates width at every 10th pixel
%>       along the line. Smaller stepsize takes longer to calculate. If you
%>       are going to calculate average width, no need to set stepsize too
%>       small.
%>
%> @retval dists An n-by-2 matrix whose columns represent width
%>   from left side of the line to the contour and from right side. Number
%>   of rows is smaller than floor of ((number of pixels in line)/stepsize). 
%> 
%> @example 
%>   contour = imread('inputs/contour.png');
%>   dists = liner(cont, [515,100], [24 84]);
%>   lr_width = mean(dists); % lr_width(1) is avg distance to contour from
%>                           % one side and lr_width(2) is from the other side of line
%>   avg_width = sum(lr_width); % avg width of the region surrounding line
%====================================================================
function [dists, varargout] = liner(contour, startpt, endpt, varargin)
% LINER Line radius
%   dists = liner(contour, startpt, endpt) calculates distances from line
%   given by startpt and endpt to nearest points given by contour image.
%   This function can be used to calculate width of airfield runway given a
%   line vector representing runway and a bw contour image representing
%   contour of area including runway, taxiroute, parking area etc.
%
%   Parameters:
%       contour - n-by-m bw image indicating contour
%
%       startpt - starting point of line vector with length 2
%
%       endpt - end point of line vector with length 2
%
%   Additional parameters and options:
%       'stepsize' - sets how often the width will be calculated along the
%       line. Default stepsize = 10, calculates width at every 10th pixel
%       along the line. Smaller stepsize takes longer to calculate. If you
%       are going to calculate average width, no need to set stepsize too
%       small.
%
%   This function returns a n-by-2 matrix whose columns represent width
%   from left side of the line to the contour and from right side. Number
%   of rows is smaller than floor of ((number of pixels in line)/stepsize).
%
%   Contour must be logical and have same 2D size with original image.
%
%   Sample run:
%   contour = imread('inputs/contour.png');
%   dists = liner(cont, [515,100], [24 84]);
%   lr_width = mean(dists); % lr_width(1) is avg distance to contour from
%                           % one side and lr_width(2) is from the other side of line
%   avg_width = sum(lr_width); % avg width of the region surrounding line

%   This function uses drawline function available at:
%   http://www.mathworks.com/matlabcentral/fileexchange/15374-drawline

% Revision history:
% Rev 3 : Returns positions of all intersections with mask contour
% Rev 2 : Fixed problem with vertical lines and line segments having no
% intersection with contour
% Rev 1 : Two intersection points are taken only if they fall to different
% sides of original line.
% Rev 0 : Initial release
%%

global DEBUG_FLAG;

x1 = startpt(1);
y1 = startpt(2);
x2 = endpt(1);
y2 = endpt(2);

stepsize = 10;
if nargin > 3
    for i = 1:length(varargin)
        switch lower(varargin)
            case 'stepsize'
                stepsize = varargin{i+1};        
        end
    end
end

int_check = false;

if nargout == 2
    int_check = true;
elseif nargout ~= 1
    error('Liner:InvalidNargout', 'Invalid number of output arguments');
end

linenorm = [-(x1-x2) / (y1-y2), 1];
m = -linenorm(1);
if isinf(m)
    m=bitmax;
end
mc = (y1-y2)/(x1-x2); % slope of original line
if isinf(mc)
    mc=bitmax;
end
cc = y1 - mc*x1; % constant of original line
% linenorm_unit = linenorm / norm(linenorm); % uncomment this if unit vector is required
contour = logical(contour(:,:,1));
[lineind, ~] = drawline([x1 y1], [x2 y2], size(contour));

% test_im = false(size(contour));
result = []; % holds distances to main line from left and right
% result = zeros(uint32(length(lineind)/stepsize), 2); 
resulti = 1;

allintersections = [];

if DEBUG_FLAG
    imshow(contour), hold on;
    plot([startpt(2);endpt(2)],[startpt(1);endpt(1)],'LineWidth',2,'Color','green');
    plot(startpt(2),startpt(1),'x','LineWidth',2,'Color','yellow');
    plot(endpt(2),endpt(1),'x','LineWidth',2,'Color','red');
end

for i = 1:stepsize:length(lineind) % if average radius is required this can be converted to picking random n pixels instead of running for each pixel
    skip = false;
    [x,y] = ind2sub([size(contour,1), size(contour,2)], lineind(i));
    c = -y - m * x; % equation of intersecting line
    endpts = zeros(2,2);
    ii = 0;
    
    yy = -c - m; %% xx = 1
    if yy >= 0 && yy <= size(contour,2)% intersects x=0 inside image
        ii = ii + 1;
        endpts(ii, : ) = [1, yy];
    end
    yy = -c - m*size(contour,1); % xx = size(contour,1)
    if yy >= 0 && yy <= size(contour,2)
        ii = ii+1;
        endpts(ii, :) = [size(contour,1), yy];
    end
    
    xx = (-1 - c) / m; % yy = 1
    if xx >= 0 && xx <= size(contour,1)
        ii = ii+1;
        endpts(ii, :) = [xx, 1];     
    end
    
    xx = (-size(contour,2) - c ) / m; %% yy = size(contour,2)
    if xx >= 0 && xx <= size(contour,1)
        ii = ii+1;
        endpts(ii, :) = [xx, size(contour,2)];
    end
    
    if ii < 2
        error('Liner:BUG', 'line intersects with less than 2 boundaries');
    end
    
    endpts = round(endpts);
    endpts(endpts == 0) = 1; % if rounded towards zero, change to 1

    if size(endpts,1)>2
        endpts(2,:)=[];
    end
    
    if DEBUG_FLAG
        plot(endpts(:,2),endpts(:,1),'LineWidth',2,'Color','green');
    end
    [line2ind, ~]  = drawline(endpts(1,:), endpts(2,:), size(contour));
    line2mask = false(size(contour));
    line2mask(line2ind) = true;
    line2mask = imdilate(line2mask, strel('disk', 1));
    
    isects = line2mask & contour;
    isects = bwmorph( isects, 'shrink', Inf);
    [ix, iy] = find(isects);
    if int_check
        allintersections = [allintersections; [ix iy]];
    end
    coords = [ix, iy]; % coordinates of intersections
    if DEBUG_FLAG
        plot(coords(:,2), coords(:,1), 'x','LineWidth',2,'Color','yellow');
    end
        
    dists = pdist2(coords, [x y]); % result is size(coords,1) by 1
    minind1 = dists == min(dists);
    mincoord1 = coords(minind1, :);
    if min(size(mincoord1)) == 0
        continue;
    end
    dists(minind1) = Inf;
    
    while true
        if size(mincoord1,1) > 1
            % both sides have equal distance to center
            mincoord2 = mincoord1(2,:);
            break;
        end
        minind2 = dists == min(dists);
        if dists(minind2) == Inf
            % All points fall to the same side of line skip this step
            skip = true;
            break;
        end
        dists(minind2) = Inf;
        mincoord2 = coords(minind2, :);
        if (mincoord1(1,2) - mc * mincoord1(1,1) - cc) * ... % check if points fall to different sides of original line
                (mincoord2(1,2) - mc * mincoord2(1,1) - cc ) < 0
            break;
        end
    end
    
    if ~skip
%         test_im(mincoord1(1,1),mincoord1(1,2) ) = 1;
%         test_im(mincoord2(1,1),mincoord2(1,2) ) = 1;
        result(resulti,:) = [norm(mincoord1(1,:) - [x y]), norm(mincoord2(1,:) - [x y])];
        resulti = resulti+1;
    end
    if DEBUG_FLAG
        plot([mincoord1(1,2);mincoord2(1,2)],[mincoord1(1,1);mincoord2(1,1)],'LineWidth',2,'Color','red');
    end
end

% hold off;

dists = result(1:resulti-1, :);
if int_check
    varargout{1} = allintersections;
%     tmp = false(size(contour));
%     tmp(sub2ind(size(contour),allintersections(:,1), allintersections(:,2))) = 1;
%     figure, imshow(tmp);
end

% figure, imshow(test_im);

end