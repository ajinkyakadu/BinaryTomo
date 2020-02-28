function [xTV,hist] = solveTVmin(A,b,D,sigma,options)
%solveTV A primal-dual algorithm to find a solution to
%
%   min_{x} |D*x|_1  subject to 0.5*|A*x - b|_2^2 < sigma
%
% Input:
%   A : a (tomography) matrix of size m x n
%   b : (tomographic) projection data vector of size m x 1
%   D : a finite-difference matrix of size p x n
%   sigma : regularization parameter for TV
%   options:
%       maxIter : maximum number of iterations (default: 1e4)
%       OptTol  : tolerance level for optimality (default: 1e-6)
%       ProgTol : tolerance level for progress (default: 1e-6)
%       saveHist: an indicator for saving history (default:0)
%
% Output:
%   xTV : solution (size n x 1)
%   hist - history containing values at each iteration
%       f : function value max(0.5*norm(A*x-b)^2-sigma,0)
%       g : function value |D*x|_1
%       cost : sum of two functions f and g
%       er : progress value, |[x;u;v]-[xp;up;vp]|_2
%       opt : optimality value, |(A'*v) + (D'*u)|_2
% 
%
% Created by:
%   - Ajinkya Kadu, Utrecht University
%   Feb 18, 2020

if nargin < 5
    options = [];
end

maxIter = getoptions(options,'maxIter',1e4);
optTol  = getoptions(options,'optTol',1e-6);
progTol = getoptions(options,'progTol',1e-6);
saveHist= getoptions(options,'saveHist',0);


[m,n] = size(A);
[p,n] = size(D);

x = zeros(n,1);
v = zeros(m,1);
u = zeros(p,1);

gamma = 0.95/normest([D;A]);
%%

for k=1:maxIter
    
    % update primal variable x
    xp = x;
    x  = x - gamma*((A'*v)+(D'*u));
    
    dx = xp-2*x;
    
    % update dual variable v
    vp = v;
    v  = proxfd(vp-gamma*(A*dx),gamma,b,sigma);
    
    % update dual variable u
    up = u;
    u  = proxgd(up-gamma*(D*dx),gamma);
   
        
    hist.er(k)   = norm([x;u;v]-[xp;up;vp]);
    hist.opt(k)  = norm((A'*v) + (D'*u));
    if saveHist
        hist.f(k)    = max(0.5*norm(A*x-b)^2-sigma,0);
        hist.g(k)    = norm(D*x,1);
        hist.cost(k) = hist.f(k) + hist.g(k);
    end
    
    % optimality
    if (hist.opt(k) < optTol)
        fprintf('stopped at iteration %d \n',k);
        fprintf('Optimality: %d \n',hist.opt(k));
        fprintf('relative progress: %d \n',hist.er(k));
        break;
    end
    
    % progress
    if (hist.er(k) < progTol)
        fprintf('stopped at iteration %d \n',k);
        fprintf('relative progress: %d \n',hist.er(k));
        fprintf('Optimality: %d \n',hist.opt(k));
        break;
    end
    
    if k==maxIter
        fprintf('completed iterations %d \n',k);
        fprintf('Optimality: %d \n',hist.opt(k));
        fprintf('relative progress: %d \n',hist.er(k));
    end
    
end

xTV = x;

end

function [y] = proxfd(x,gamma,b,sigma)
% proximal for dual of f(x) = {0.5*|x - b|^2 <= sigma}

y = x - gamma*proxf(x/gamma,1/gamma,b,sigma);

end

function [y] = proxf(x,gamma,b,sigma)
% proximal for f(x) = {0.5*|x - b|^2 <= sigma}

dist = norm(x-b);

if dist <= sqrt(2*sigma)
    y = x;
else
    y = b + sqrt(2*sigma)*(x-b)/dist;
end

end

function [y] = proxgd(x,gamma)
% proximal for dual of function g(x) = |x|_1

y = x - gamma*proxg(x/gamma,1/gamma);

end

function [y] = proxg(x,gamma)
% proximal for function g(x) = |x|_1

t = gamma;
y = max(0, x - t) - max(0, -x - t);

end

