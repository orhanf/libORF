% //****************************************************************************
% //
% // neighborhood() returns a byte value representing the immediate neighborhood
% // of the specified point in bit form, clockwise from 12:00.
% //
% // R/O arr:  binary pixel array
% // R/O i,j:  coordinates of point
% // R/O nrow, ncol:  dimensions of binary image
% // Returns:  bit representation of 8-neighbors
% //
function NeighborOut=neighborhood(arr,i, j,  nrow,  ncol) 
   p = i+j*nrow;
   condition = 8*(i <= 1)+4*(j <= 1)+2*(i >= nrow-1)+(j >= ncol-1);
NeighborOut=0;
%   //mexPrintf("Condition:  %d\n",condition);
  switch (condition) 
  case 0,  %// all sides valid
    if (p-nrow-1)<1 | (p+nrow+1)>nrow*ncol,
        % too low or too high
        NeighborOut=0;
    else,
        NeighborOut= arr(p-1) + 2*arr(p+nrow-1) + 4*arr(p+nrow) +...
      8*arr(p+nrow+1)+ 16*arr(p+1) + 32*arr(p-nrow+1) +...
      64*arr(p-nrow) + 128*arr(p-nrow-1);
    end;
  case 1,  %// right side not valid
    NeighborOut= arr(p-1) + 16*arr(p+1) + 32*arr(p-nrow+1) +...
      64*arr(p-nrow) + 128*arr(p-nrow-1);
  case 2, % // bottom not valid
    NeighborOut =arr(p-1) + 2*arr(p+nrow-1) + 4*arr(p+nrow) +...
      64*arr(p-nrow) + 128*arr(p-nrow-1);
  case 3, % // bottom and right not valid
    NeighborOut= arr(p-1) + 64*arr(p-nrow) + 128*arr(p-nrow-1);
  case 4, % // left side not valid
    NeighborOut= arr(p-1) + 2*arr(p+nrow-1) + 4*arr(p+nrow) +...
      8*arr(p+nrow+1) + 16*arr(p+1);
  case 5, %// left and right sides not valid
     NeighborOut= arr(p-1) + 16*arr(p+1);
  case 6,  %// left and bottom sides not valid
    NeighborOut= arr(p-1) + 2*arr(p+nrow-1) + 4*arr(p+nrow);
  case 7,  % left, bottom, and right sides not valid
    NeighborOut= arr(p-1);
  case 8, % top side not valid
    NeighborOut= 4* arr(p+nrow) + 8*arr(p+nrow+1) + 16*arr(p+1) +... 
      32*arr(p-nrow+1) + 64*arr(p-nrow);
  case 9,  % top and right not valid
    NeighborOut= 16*arr(p+1) + 32*arr(p-nrow+1) + 64*arr(p-nrow);
  case 10,  % top and bottom not valid
    NeighborOut= 4*arr(p+nrow) + 64*arr(p-nrow);
  case 11,  % top, bottom and right not valid
    NeighborOut= 64*arr(p-nrow);
  case 12,  % top and left not valid
    NeighborOut= 4*arr(p+nrow) + 8*arr(p+nrow+1) + 16*arr(p+1);
  case 13,  % top, left and right sides not valid
    NeighborOut= 16*arr(p+1);
  case 14,  % top, left and bottom sides not valid
    NeighborOut= 4*arr(p+nrow);
  case 15,  % no sides valid
    NeighborOut= 0;
  end; % end switch condition
NeighborOut=NeighborOut+1; % because in Matlab, indices start at 1 not 0