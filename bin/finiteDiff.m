function [D] = finiteDiff(n)
% finite-difference matrix for TV
%
% example:
%       D = finiteDiff(30);
%
% Input:
%   n - number of gridpoints in each directions
%
% Output:
%   D - the finite difference matrix
%


I = speye(n);
e = ones(n,1);

Dx = spdiags([-1*e 1*e],-1:0,n,n);
Dx(1,end) = -1;

D  = [kron(Dx,I);kron(I,Dx)];
end

