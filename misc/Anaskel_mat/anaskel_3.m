% //***************************************************************************
% //
% // Matlab C routine file:  anaskel.cpp
% // (converted into a .m file by Matt Fetterman
% // Written 8/04 by N. Howe
% //
% // Input:
%     nrow: number of rows
%     ncol: number of columns
% //  Input Image: input binary image
% //
% // Output:
% //   xxxx dmap:  directional maps (you can put this back in but I did not
% find it that useful)
% //   exy:   coordinates of endpoints...2D matrix of endpoints.
% //   jxy:   coordinates of junction points...2D matrix of junction
% points.
% //
% //***************************************************************************
function [exy,jxy]=anaskel_3(nrow,ncol,InputImage)

% this array is 256 length.
nbrs = [ ...
  0,1,1,1,1,1,1,1,...
  1,2,2,2,1,1,1,1,...
  1,2,2,2,1,1,1,1,...
  1,2,2,2,1,1,1,1,...
  1,2,2,2,2,2,2,2,...
  2,3,3,3,2,2,2,2,...
  1,2,2,2,1,1,1,1,...
  1,2,2,2,1,1,1,1,...
  1,1,2,1,2,1,2,1,...
  2,2,3,2,2,1,2,1,...
  1,1,2,1,1,1,1,1,...
  1,1,2,1,1,1,1,1,...
  1,1,2,1,2,1,2,1,...
  2,2,3,2,2,1,2,1,...
  1,1,2,1,1,1,1,1,...
  1,1,2,1,1,1,1,1,...
  1,1,2,1,2,1,2,1,...
  2,2,3,2,2,1,2,1,...
  2,2,3,2,2,1,2,1,...
  2,2,3,2,2,1,2,1,...
  2,2,3,2,3,2,3,2,...
  3,3,4,3,3,2,3,2,...
  2,2,3,2,2,1,2,1,...
  2,2,3,2,2,1,2,1,...
  1,1,2,1,2,1,2,1,...
  2,2,3,2,2,1,2,1,...
  1,1,2,1,1,1,1,1,...
  1,1,2,1,1,1,1,1,...
  1,1,2,1,2,1,2,1,...
  2,2,3,2,2,1,2,1,...
  1,1,2,1,1,1,1,1,...
  1,1,2,1,1,1,1,1 ...
];

nbr_branches = [...
  0,1,1,1,1,2,1,2,...
  1,2,2,2,1,2,2,2,...
  1,2,2,2,2,3,2,3,...
  1,2,2,2,2,3,2,3,...
  1,2,2,2,2,3,2,3,...
  2,3,3,3,2,3,3,3,...
  1,2,2,2,2,3,2,3,...
  2,3,3,3,2,3,3,3,...
  1,2,2,2,2,3,2,3,...
  2,3,3,3,2,3,3,3,...
  2,3,3,3,3,4,3,4,...
  2,3,3,3,3,4,3,4,...
  1,2,2,2,2,3,2,3,...
  2,3,3,3,2,3,3,3,...
  2,3,3,3,3,4,3,4,...
  2,3,3,3,3,4,3,4,...
  1,1,2,2,2,2,2,2,...
  2,2,3,3,2,2,3,3,...
  2,2,3,3,3,3,3,3,...
  2,2,3,3,3,3,3,3,...
  2,2,3,3,3,3,3,3,...
  3,3,4,4,3,3,4,4,...
  2,2,3,3,3,3,3,3,...
  3,3,4,4,3,3,4,4,...
  1,2,2,2,2,3,2,3,...
  2,3,3,3,2,3,3,3,...
  2,3,3,3,3,4,3,4,...
  2,3,3,3,3,4,3,4,...
  2,2,3,3,3,3,3,3,...
  3,3,4,4,3,3,4,4,...
  2,3,3,3,3,4,3,4,...
  3,3,4,4,3,4,4,4,...
];

isN = [...
  0,0,0,0,0,0,0,0,...
  0,1,0,1,0,1,0,1,...
  0,1,1,1,0,1,1,1,...
  0,1,1,1,0,1,1,1,...
  0,1,0,1,0,1,0,1,...
  0,1,0,1,0,1,0,1,...
  0,1,1,1,0,1,1,1,...
  0,1,1,1,0,1,1,1,...
  0,0,0,0,0,0,0,0,...
  0,1,0,1,0,1,0,1,...
  0,1,1,1,0,1,1,1,...
  0,1,1,1,0,1,1,1,...
  0,1,0,1,0,1,0,1,...
  0,1,0,1,0,1,0,1,...
  0,1,1,1,0,1,1,1,...
  0,1,1,1,0,1,1,1,...
  0,0,0,0,0,0,0,0,...
  0,1,0,1,0,1,0,1,...
  1,1,1,1,1,1,1,1,...
  1,1,1,1,1,1,1,1,...
  0,1,0,1,0,1,0,1,...
  0,1,0,1,0,1,0,1,...
  1,1,1,1,1,1,1,1,...
  1,1,1,1,1,1,1,1,...
  0,0,0,0,0,0,0,0,...
  0,1,0,1,0,1,0,1,...
  1,1,1,1,1,1,1,1,...
  1,1,1,1,1,1,1,1,...
  0,1,0,1,0,1,0,1,...
  0,1,0,1,0,1,0,1,...
  1,1,1,1,1,1,1,1,...
  1,1,1,1,1,1,1,1,...
];

isNE = [...
  0,0,0,0,0,0,0,0,...
  0,0,0,0,0,0,0,0,...
  0,0,1,1,0,0,1,1,...
  0,0,1,1,0,0,1,1,...
  0,1,1,1,1,1,1,1,...
  0,1,1,1,1,1,1,1,...
  0,1,1,1,1,1,1,1,...
  0,1,1,1,1,1,1,1,...
  0,0,1,1,0,0,1,1,...
  0,0,1,1,0,0,1,1,...
  0,0,1,1,0,0,1,1,...
  0,0,1,1,0,0,1,1,...
  0,1,1,1,1,1,1,1,...
  0,1,1,1,1,1,1,1,...
  0,1,1,1,1,1,1,1,...
  0,1,1,1,1,1,1,1,...
  0,0,0,0,0,0,0,0,...
  0,0,0,0,0,0,0,0,...
  0,0,1,1,0,0,1,1,...
  0,0,1,1,0,0,1,1,...
  0,1,1,1,1,1,1,1,...
  0,1,1,1,1,1,1,1,...
  0,1,1,1,1,1,1,1,...
  0,1,1,1,1,1,1,1,...
  0,0,1,1,0,0,1,1,...
  0,0,1,1,0,0,1,1,...
  0,0,1,1,0,0,1,1,...
  0,0,1,1,0,0,1,1,...
  0,1,1,1,1,1,1,1,...
  0,1,1,1,1,1,1,1,...
  0,1,1,1,1,1,1,1,...
  0,1,1,1,1,1,1,1];
isE=[  0,0,0,0,0,0,0,0,...
  0,0,0,0,0,0,0,0,...
  0,0,0,0,0,0,0,0,...
  0,0,0,0,0,0,0,0,...
  0,0,0,0,1,1,1,1,...
  0,0,0,0,1,1,1,1,...
  0,0,0,0,1,1,1,1,...
  0,0,0,0,1,1,1,1,...
  0,0,1,1,1,1,1,1,...
  1,1,1,1,1,1,1,1,...
  0,0,1,1,1,1,1,1,...
  1,1,1,1,1,1,1,1,...
  0,0,1,1,1,1,1,1,...
  1,1,1,1,1,1,1,1,...
  0,0,1,1,1,1,1,1,...
  1,1,1,1,1,1,1,1,...
  0,0,0,0,1,1,1,1,...
  0,0,0,0,1,1,1,1,...
  0,0,0,0,1,1,1,1,...
  0,0,0,0,1,1,1,1,...
  0,0,0,0,1,1,1,1,...
  0,0,0,0,1,1,1,1,...
  0,0,0,0,1,1,1,1,...
  0,0,0,0,1,1,1,1,...
  0,0,1,1,1,1,1,1,...
  1,1,1,1,1,1,1,1,...
  0,0,1,1,1,1,1,1,...
  1,1,1,1,1,1,1,1,...
  0,0,1,1,1,1,1,1,...
  1,1,1,1,1,1,1,1,...
  0,0,1,1,1,1,1,1,...
  1,1,1,1,1,1,1,1];

isSE = [...
  0,0,0,0,0,0,0,0,...
  0,1,0,1,0,1,0,1,...
  0,0,0,0,0,0,0,0,...
  0,1,0,1,0,1,0,1,...
  0,0,0,0,0,0,0,0,...
  0,1,0,1,0,1,0,1,...
  0,0,0,0,0,0,0,0,...
  0,1,0,1,0,1,0,1,...
  0,0,0,0,0,0,0,0,...
  1,1,1,1,1,1,1,1,...
  0,0,0,0,0,0,0,0,...
  1,1,1,1,1,1,1,1,...
  0,0,0,0,0,0,0,0,...
  1,1,1,1,1,1,1,1,...
  0,0,0,0,0,0,0,0,...
  1,1,1,1,1,1,1,1,...
  0,0,0,0,1,1,1,1,...
  1,1,1,1,1,1,1,1,...
  1,1,1,1,1,1,1,1,...
  1,1,1,1,1,1,1,1,...
  0,0,0,0,1,1,1,1,...
  1,1,1,1,1,1,1,1,...
  1,1,1,1,1,1,1,1,...
  1,1,1,1,1,1,1,1,...
  0,0,0,0,1,1,1,1,...
  1,1,1,1,1,1,1,1,...
  1,1,1,1,1,1,1,1,...
  1,1,1,1,1,1,1,1,...
  0,0,0,0,1,1,1,1,...
  1,1,1,1,1,1,1,1,...
  1,1,1,1,1,1,1,1,...
  1,1,1,1,1,1,1,1];

% //****************************************************************************
% //
% // This is main function 
% // Well actually the main function is declared at the top, but here it
% kicks in.
% //****************************************************************************

% function [exy,jxy]=anaskel_3(nrow,ncol,InputImage) 
  % check for proper number of arguments
%   if nargin~=1,
%       disp('Exactly one input argument required.');
%   end;
%   if nargout>2,
%       disp('ERROR: Too many output arguments.');
%   end;
  % check format of arguments
 % errCheck(mxIsUint8(prhs(0))||mxIsLogical(prhs(0))||mxIsDouble(prhs(0)),
  %         "Input must be binary image.");
  skel = zeros(1,nrow*ncol);
  % trim skeleton
  skel=dotrim(InputImage,nrow,ncol,skel,nbrs,nbr_branches);
  % allocate output space
%   cell = mxCreateDoubleMatrix(nrow, ncol, mxREAL);
%   mxSetCell(plhs(0),0,cell);
%   north = mxGetPr(cell);
%   cell = mxCreateDoubleMatrix(nrow, ncol, mxREAL);
%   mxSetCell(plhs(0),1,cell);
%   northeast = mxGetPr(cell);
%   cell = mxCreateDoubleMatrix(nrow, ncol, mxREAL);
%   mxSetCell(plhs(0),2,cell);
%   east = mxGetPr(cell);
%   cell = mxCreateDoubleMatrix(nrow, ncol, mxREAL);
%   mxSetCell(plhs(0),3,cell);
%   southeast = mxGetPr(cell);

  %// analyze...
  for j = 1: ncol-1, 
    for i = 0:nrow, 
      p = i+j*nrow;
      if (skel(p)) 
        	hood = neighborhood(skel,i,j,nrow,ncol);
      end;
    end;
end;
% 	north(p) = isN(hood);
% 	northeast(p) = isNE(hood);
% 	east(p) = isE(hood);
% 	southeast(p) = isSE(hood);
% 	//mexPrintf("Point (%d,%d):  Hood %d -> %d, %d, %d, %d.\n",
% 	//	  i,j,hood,isN(hood),isNE(hood),isE(hood),isSE(hood));
%       } else {
% 	north(p) = northeast(p) = east(p) = southeast(p) = 0;
     
% % extra data if necessary
%   if (nlhs > 1) {
%     % count junctions and endpoints
%     njunc = 0;
%     nend = 0;
%     nbar = 0;
%     for (j = 0; j < ncol; j++) {
%       for (i = 0; i < nrow; i++) {
% 	if (skel(i+j*nrow)) {
% 	  hood = neighborhood(skel,i,j,nrow,ncol);
% 	  switch (nbr_branches(hood)) {
% 	  case 0:
% 	  case 1:
% 	    nend++;
% 	    break;
% 	  case 2:
% 	    nbar++;
% 	    break;
% 	  case 3:
% 	  case 4:
% 	    njunc++;
% 	    break;
% 	  }
% 	}
%       }
%     }
%     //mexPrintf("Counted.\n");
% ************
% the exy points are endpoints.let's make it a 2D array unlike the c++
% program.
    iend=0;
    exy=0;
  for j = 1:  ncol-1,
  for i = 1: nrow,
	if (skel(i+j*nrow)) 
	  hood = neighborhood(skel,i,j,nrow,ncol);
	  if (nbr_branches(hood) < 2) ,
	    exy(1,iend+1) = j+1;
	    exy(2,iend+1) = i+1;
	    iend =iend+1;
      end; % end if
    end; % end if
  end;% end for
  end; %end for
 % the jxy are junctions. let's make it a 2D array unlike the c++ program.
    jxy = 0;
    ijunc = 0;
    for j=1: ncol-1,
      for i = 1: nrow,
        if (skel(i+j*nrow)) ,
    	  hood = neighborhood(skel,i,j,nrow,ncol);
	   if (nbr_branches(hood) > 2) ,
         jxy(1,ijunc+1) = j+1;
         jxy(2,ijunc+1) = i+1;
         ijunc =ijunc+ 1;
       end;% end if
        end; %end if;
      end; %end for
    end; %end for
 % end main function. next comes nested functions.
%/*************************************************************************
%***/
% //***************************************************************************
% 
% #define SQR(x) (x)*(x)
% #define MIN(x,y) (((x) < (y)) ? (x):(y))
% #define MAX(x,y) (((x) > (y)) ? (x):(y))
% #define ABS(x) (((x) < 0) ? (-(x)):(x))
% #define errCheck(a,b) if (!(a)) mexErrMsgTxt((b));
% #define bdCheck(a,b) mxAssert(((a)>=0)&&((a)<(b)),"Bounds error.")
% 
% //***************************************************************************





