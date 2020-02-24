function [totSol,Xint,Xt] = count_solutions_script(A,b,u,sigma)
%count_solutions_script counts the number of possible solutions to the
%problem 
% find x \in u^n  subject to 0.5*|A*x - b| <= sigma
%
% It also gives the solutions and plot maximum 4 of them. It also provides
% the intersection of these solutions
% 
% Input:
%   A : tomographic matrix (m x n matrix)
%   b : tomographic data (m x 1 vector)
%   u : greylevels vector (k x 1 vector)
%   sigma : noise level (scalar positive value)
%
% Output:
%   totSol : total number of solutions
%   Xint   : intersection of solutions 
%   Xt     : all t possible solutions (n x t matrix)
%
% Created by:
%   - Ajinkya Kadu, Utrecht University
%   Feb 18, 2020

n = sqrt(size(A,2));

fprintf('---------- Counting solutions ----------- \n')

[totSol,Xt] = count_solutions(A,b,u,sigma);
fprintf('sigma=%.4f: %d solutions \n',sigma,totSol);


fprintf('plotting solutions...\n')
figure(99);
for i=1:min(totSol,4)
    subplot(1,min(totSol,4),i);imagesc(reshape(Xt(:,i),n,n));axis image;colormap gray;
    axis off; title(['solution ' num2str(i)]);pause(0.001);
end

% intersection
Xint = Xt(:,1);
if totSol > 1
    fprintf('generating intersection of solutions \n')
    for i=2:totSol
        Xint          = Xint.*Xt(:,i);
        Xint(Xint<0)  = 0;
        Xint(Xint==1) = Xt(Xint==1);
    end
    Xint = reshape(Xint,n,n);
end

end

function [totSol,Xt,misfit] = count_solutions(A,y,u,sigma)
% count solutions and give all the solutions


% get all possible combinations
n      = size(A,2);
totPos = length(u)^n;
X      = getX(u',n,totPos);

% compute misfits
misfit = zeros(totPos,1);
for i=1:totPos
    x = X(:,i);
    misfit(i) = 0.5*norm(A*x-y)^2;
end

% compute total number of solutions
solId  = find(misfit<=sigma);
totSol = length(solId);

% generate all solutions
Xt = zeros(n,totSol);
for j=1:totSol
    Xt(:,j) = X(:,solId(j));
end


end

function [x] = getX(u,n,totPos)
% get all possible combinations

x = zeros(n,totPos);
K = length(u);

for i=1:n
    x(i,:) = repmat(kron(u,ones(1,K^(i-1))),1,totPos/(K^i));
end

end