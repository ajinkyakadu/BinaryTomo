function [xD,hist] = solveTVminBT(A,b,D,sigma,options)
%solveTVBT A primal-dual algorithm to find a solution to
%
%   min_{x,z} |A'*x+D'*z|_1
%   subject to  |z|_inf <= 1, 0.5*|x - b|_2^2 <= sigma
%
% Input:
%   A : a (tomography) matrix of size m x n
%   b : (tomographic) projection data vector of size m x 1
%   D : a finite-difference matrix of size p x n
%   sigma : noise level in the data
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
%       g : function value |A'*x+D'*z|_1
%       cost : sum of two functions f and g
%       er : progress of the iterates |[x;u;v]-[xp;up;vp|_2 
%       opt : optimality value |v+K*u|_2
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
[p,n] = size(D);

K = [A;D];

x = zeros(m+p,1);
v = zeros(m+p,1);
u = zeros(n,1);

gamma = 0.95/normest([K speye(m+p)]);
%%

for k=1:maxIter
    
    % update primal variable x
    xp = x;
    x  = xp - gamma*(K*u+v);
    
    dx = xp - 2*x;
        
    % update dual variable (v)
    vp = v;
    v  = proxfd(vp - gamma*dx,gamma,m,b,sigma);
    
    % update dual variable (u)
    up = u;
    u  = proxgd(up - gamma*(K'*dx),gamma);
    
    if saveHist
        Ktx          = K'*x;
        hist.f(k)    = max(0.5*norm(x(1:m)-b)^2-sigma,0) ...
                        + max(norm(x(m+1:end),1)-1,0);
        hist.g(k)    = norm(Ktx,1);
        hist.cost(k) = hist.f(k) + hist.g(k);
    end
    
    hist.er(k)   = norm([x;u;v]-[xp;up;vp]);
    hist.opt(k)  = norm(v+K*u);
    
    if hist.er(k) < progTol
        fprintf('stopped at iteration %d \n',k);
        fprintf('relative progress: %d \n',hist.er(k));
        fprintf('Optimality: %d \n',hist.opt(k));
        break;
    end
    
    if hist.opt(k) < optTol
        fprintf('stopped at iteration %d \n',k);
        fprintf('Optimality: %d \n',hist.opt(k));
        fprintf('relative progress: %d \n',hist.er(k));
        break;
    end
    
end

xD = -u;

end

function [y] = proxfd(x,gamma,m,b,sigma)
% proximal (dual) for f(x) + g(z) = {0.5*|x - b|^2 <= sigma} + {|z|_inf<=1}

x1 = x(1:m);
x2 = x(m+1:end);

% prox for {0.5*|x - b|^2 < sigma}
y1 = proxf(x1,gamma,b,sigma);

% prox (dual) for {|z|_inf<=1}
y2 = max(0, x2 - gamma) - max(0, -x2 - gamma);

y  = [y1;y2];

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

y = max(0, x - gamma) - max(0, -x - gamma);

end

