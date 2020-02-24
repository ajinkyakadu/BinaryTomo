function [xD,hist] = solveBT(A,b,options)
%solveBT A primal-dual algorithm to find a solution to
%
%   min_{x} 0.5*\|x - b\|_2^2 + \|A'*x\|_1
%
% Input:
%   A : a (tomography) matrix of size m x n
%   b : (tomographic) projection data vector of size m x 1
%   options:
%       maxIter : maximum number of iterations (default: 1e3)
%       optTol  : tolerance level for optimality (default: 1e-6)
%       progTol : tolerance level for progress (default: 1e-6)
%       saveHist: an indicator for saving history (default:0)
%
% Output:
%   xD : solution (size n x 1)
%   hist - history containing values at each iteration
%       f : function value 0.5|x-b|_2^2
%       g : function value |A'*x|_1
%       cost : sum of two functions f and g
%       er : error value |x-xprev|_2 + |u - uprev|_2
% 
%
% Created by:
%   - Ajinkya Kadu, Utrecht University
%   Feb 18, 2020

maxIter = getoptions(options,'maxIter',1000);
optTol  = getoptions(options,'optTol',1e-6);
progTol = getoptions(options,'progTol',1e-6);
saveHist= getoptions(options,'saveHist',0);


[m,n] = size(A);

x = zeros(m,1);
u = zeros(n,1);

gamma = 0.95/normest(A);
%%

for k=1:maxIter
    
    % update primal variable (x)
    xp = x;
    x = proxf(xp-gamma*(A*u),b,gamma);
    
    % update dual variable (u)
    up = u;
    dx = xp-2*x;
    u  = proxgd(up-gamma*(A'*dx),gamma);
    
    
    % history
    hist.er(k)   = norm([x;u]-[xp;up]);
    hist.opt(k)  = norm(x - b + A*u);
    
    if saveHist
        Atx          = A'*x;
        hist.f(k)    = 0.5*norm(x-b)^2;
        hist.g(k)    = norm(Atx,1);
        hist.cost(k) = hist.f(k) + hist.g(k);
    end
    
    if (hist.opt(k) < optTol)
        fprintf('stopped at iteration %d \n',k);
        fprintf('Optimality: %d \n',hist.opt(k));
        fprintf('relative progress: %d \n',hist.er(k));
        break;
    end
    
    if (hist.er(k) < progTol)
        fprintf('stopped at iteration %d \n',k);
        fprintf('relative progress: %d \n',hist.er(k));
        fprintf('Optimality: %d \n',hist.opt(k));
        break;
    end
    
end

xD = u;

end

function [y] = proxf(x,b,gamma)
% proximal for f(x) = 0.5*|x - b|^2

y = (x + gamma*b)/(1+gamma);

end

function [y] = proxgd(x,gamma)
% proximal for dual of function g(x) = |x|_1

y = x - gamma*proxg(x/gamma,1/gamma);

end

function [y] = proxg(x,gamma)
% proximal for function g(x) = |x|_1

y = max(0, x - gamma) - max(0, -x - gamma);

end
