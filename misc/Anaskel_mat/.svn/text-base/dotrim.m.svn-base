% //****************************************************************************
% //
% // dotrim() trims unnecessary points from the skeleton.
% //
% // R/O inp:  binary pixel array
% // R/O nrow, ncol:  dimensions of binary image
% // W/O skel:  newly trimmed skeleton
% //
function skel=dotrim(inp, nrow, ncol,skel,connected_nbrs,nbr_branches) 
  for j = 1:  ncol-1, 
    for i = 1: nrow, 
	skel(i+j*nrow) = (inp(i+j*nrow));
    end;
  end;
  for (j = 1:ncol-1), 
    for (i = 1: nrow), 
      if (skel(i+j*nrow)) 
        hood = neighborhood(skel,i,j,nrow,ncol);
        skel(i+j*nrow) = (connected_nbrs(hood) > 1)|(nbr_branches(hood)==1);
      else 
        skel(i+j*nrow) = 0;
      end;
    end; % end for
 end;% end for