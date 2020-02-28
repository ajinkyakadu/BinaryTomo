function [xD,hist] = solveTVBT(A,b,D,lambda,options)
%solveTVBT A primal-dual algorithm to find a solution to
%
%   min_{x,z} 0.5*|x - b|_2^2 + |A'*x-D'*z|_1
%   subject to  |z|_inf <= lambda
%
% Input:
%   A : a (tomography) matrix of size m x n
%   b : (tomographic) projection data vector of size m x 1
%   D : a finite-difference matrix of size p x n
%   lambda : regularization parameter for TV
%   options:
%       maxIter : maximum number of iterations (default: 1e4)
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
%       er : error value |[x;u;v]-[xp;up;vp]|_2
%       opt : optimality value |[x;0]+v+K*u|_2
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

K = [A;D];

x = zeros(m+p,1);
v = zeros(m+p,1);
u = zeros(n,1);

gamma = 0.95/normest([K speye(m+p)]);
%%

for k=1:maxIter
    
    % update primal variable x
    xp = x;
    xn = xp-gamma*(K*u+v);
    x  = [proxf(xn(1:m),-b,gamma);xn(m+1:end)];
    
    % update dual variable (v)
    vp = v;
    dx = xp - 2*x;
    vn = vp - gamma*(dx);
    v  = [zeros(m,1);proxhd(vn(m+1:end),gamma,lambda)];
    
    % update dual variable (u)
    up = u;
    u  = proxgd(up-gamma*(K'*dx),gamma);
    
    % history
    hist.er(k)   = norm([x;u;v]-[xp;up;vp]);
    hist.opt(k)  = norm([x(1:m)+b;0*x(m+1:end)]+v+K*u);
    
    if saveHist
        Ktx          = K'*x;
        hist.f(k)    = 0.5*norm(x(1:m)-b)^2;
        hist.g(k)    = norm(Ktx,1);
        hist.cost(k) = hist.f(k) + hist.g(k);
    end
    
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

fprintf('completed iterations %d \n',k);
fprintf('Optimality: %d \n',hist.opt(k));
fprintf('relative progress: %d \n',hist.er(k));

xD = -u;

end

function [y] = proxf(x,b,gamma)
% proximal for f(x) = 0.5*|x - b|^2 

y = (x + gamma*b)/(1+gamma); 

end

function [y] = proxhd(x,gamma,lambda)
% proximal for dual of function h(x) = {|x|_inf < lambda}

t = gamma*lambda;
y = max(0, x - t) - max(0, -x - t);

end

function [y] = proxgd(x,gamma)
% proximal for dual of function g(x) = |x|_1

y = x - gamma*proxg(x/gamma,1/gamma);

end

function [y] = proxg(x,gamma)
% proximal for function g(x) = |x|_1

y = max(0, x - gamma) - max(0, -x - gamma);

end

