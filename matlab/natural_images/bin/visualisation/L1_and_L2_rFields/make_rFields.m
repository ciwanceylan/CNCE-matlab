% --------------------------------------------------
% 
% make_rFields( A, mag, cols,varargin)
%
% A           : the basis, with patches as column vectors
% mag         : magnification factor
% cols        : number of columns (x-dimension of map)
% varargin{1} : Measure for the strength of each column vector
%
% --------------------------------------------------

function [I, maxi, mini] = make_rFields( A, mag, cols, varargin)
	A = A';
	maxi = 255;
	mini = 0;

	% This is the side of the window
	dim = ceil(sqrt(size(A,1)));
	basisNb = size(A,2);

	% Helpful quantities
	dimm = dim-1;
	dimp = dim+1;
	rows = ceil(size(A,2)/cols);

	if nargin ==5
	strength = varargin{1};
	parent = varargin{2};
	elseif nargin==4
	strength = varargin{1};
	elseif nargin == 3
	strength = [];
	end


	% Initialization of the image
	if isempty(strength)
	I = maxi*ones(dim*rows+rows-1,dim*cols+cols-1); 
	for i=0:rows-1
	  for j=0:cols-1
		% This sets the patch
		if (i*cols+j+1<=basisNb)
		  I(i*dimp+1:i*dimp+dim,j*dimp+1:j*dimp+dim) = reshape(A(:,i* ...
															cols+j+1),[dim dim]);
		  end
	  end
	end
	else
	I = maxi*ones(dim*rows+3*(rows-1)+2,dim*cols+cols-1);
	for i=0:rows-1
	  for j=0:cols-1
		if (i*cols+j+1<=basisNb)
		  % This sets the patch
		I(i*(dimp+2)+1:i*(dimp+2)+dim,j*dimp+1:j*dimp+dim) = reshape(A(:,i*cols+j+1),[dim dim]);
		% This sets the strength
		I(i*(dimp+2)+2+dim,j*dimp+1:j*dimp+floor(strength(i*cols+j+1)*dim)) = 0;
		end
	  end
	end

	end


	if (mag == 1)
	else
	I = imresize(I,mag);
	end
end