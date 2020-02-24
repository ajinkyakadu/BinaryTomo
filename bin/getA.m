function A = getA(n,m)
%getA generates a tomographic projection matrix for row sums, column sums,
%diagonal sums.
%
% Example
%   x = randn(5)  % create 2D image of size 5 x 5
%   A = getA(5,3) % projection matrix of 5x5 image for row and column sum
% 
% Input:
%   n : image size ( a positive integer) 
%   m : indicator for row, column sum
%       m = 2: row and column sum
%       m = 3: row, column and diagonal sum
%       m = 4: row, column, diagonal and off-diagonal sum
%
% Output:
%   A : a (tomographic) matrix of size k x n^2 (k depends on m)
%
% Created by:
%   - Tristan van Leeuwen, Utrecht University
%   Feb 18, 2020

if nargin < 2
    m = 2;
end

A = [kron(eye(n),ones(n,1))'; kron(ones(n,1),eye(n))'];

if m > 2
    A3 = zeros(2*n-1,n^2);
    l=1;
    for k = 1:n
        I = k + [0:(n-k)]*(n+1);
        A3(l,I) = 1;
        l = l + 1;
    end
    for k = 1:n-1
        I = k*n+1 + [0:(n-k-1)]*(n+1);
        A3(l,I) = 1;
        l = l + 1;
    end
    A = [A;A3];
end

if m > 3
    A4 = zeros(2*n-1,n^2);
    l=1;
    for k = 1:n
        I = k + [0:(k-1)]*(n-1);
        A4(l,I) = 1;
        l = l + 1;
    end
    for k = 2:n
        I = k*n + [0:(n-k)]*(n-1);
        A4(l,I) = 1;
        l = l + 1;
    end
    A = [A;A4];
end
