% Tomography test (with CVX package)
%
% 2D Tomgoraphy example
%   min_{x} \| A*x - b \|^2 subject to x \in \{-1, 1\}^n
%   
% b - projected data
% A - tomography matrix
% x - (binary) image
%
% Note: CVX must be installed!
%
% Created by:
%   - Ajinkya Kadu, Utrecht University
%   Feb 18, 2020

clc; clearvars; close all;

s = RandStream('mt19937ar','Seed',1);
RandStream.setGlobalStream(s);

addpath(genpath([pwd '/bin']));

%% generate true image, projection matrix and data

fprintf('------------- Setting up ---------------- \n')

n  = 4;     % size of image: total number of pixels - n^2

% create (random) binary image
xt = -1*ones(n^2,1);
xt(randperm(n^2,floor(0.5*n^2))) = 1;
xt = reshape(xt,n,n);
 
u  = unique(xt(:));


% generate a tomography matrix (2: row + col sum, 3: row + col + diag sum)
A  = getA(n,2);  

if rank(A) < size(A,1)
    [~,idx] = licols(A');
end

A  = A(idx,:);
A  = A/normest(A);
bt = A*xt(:);   

% add noise to data (additive white Gaussian noise)
noiseLevel = 0.0;
noiseB     = randn(size(bt));
noiseB     = noiseLevel*(noiseB/norm(noiseB)*norm(bt));
b          = bt + noiseB;

fprintf('matrix A and data b generated \n');
fprintf('rank of A: %d , m: %d, n: %d \n',rank(full(A)),size(A));

% measure the noise level (sigma value)
sigma = 0.5*norm(A*xt(:)-b)^2;
fprintf(['The noise is ' num2str(sigma) '\n']);

%% count solutions

if n <= 4
    [totSol,Xint] = count_solutions_script(A,b,u,sigma);
else
    totSol = 1;
    Xint = xt;
end

%% PINV 

fprintf('------- Pseudo-inverse solution --------- \n')

xP = pinv(full(A'*A))*(A'*b);

% threshold
xPt       = xP;
xPt(xP<0) = -1;
xPt(xP>0) = 1;
xPt       = reshape(xPt,n,n);

misfitP = norm(A*xPt(:)-b);
jacIdP  = sum(max(xPt(:).*xt(:),0))/nnz(xt(:));
incIdP  = nnz(min(xPt(:).*Xint(:),0)) + nnz(xPt(Xint==0));

fprintf('misfit           = %.4f \n',misfitP);
fprintf('jaccard index    = %.4f \n',jacIdP);
fprintf('Incorrect pixels = %d \n',incIdP);


%% dual - CVX
% 
% solve: min_{p} |A'*p|_1 + 0.5*|p - b|_2^2
%
% We use Gurobi solver in conjunction with CVX
% In case Gurobi is not installed, use standard solver sdpt3

fprintf('------------- Dual solution ------------- \n')


cvx_solver gurobi   % sdpt3
cvx_precision high

[rA,cA] = size(A);

cvx_begin quiet
    variables p(rA) 
    minimize (norm(A'*p,1) + 0.5*sum_square(p - b))
cvx_end


q  = A'*p; 
qt = q;
qt(abs(q) < 1e-10) = 0;   % thresholding of 1e-8
xD = sign(qt);           % solution is signum function applied to dual variable
xD = reshape(xD,n,n);

misfitD = norm(A*xD(:)-b);
jacIdD  = sum(max(xD(:).*xt(:),0))/nnz(xt(:));
jacIdDi = sum(max(xD(:).*Xint(:),0))/nnz(Xint(:));
incIdD  = nnz(min(xD(:).*Xint(:),0)) + nnz(xD(Xint==0));

fprintf('misfit              = %.4f \n',misfitD);
fprintf('jaccard index       = %.4f \n',jacIdD);
fprintf('jaccard index (int) = %.4f \n',jacIdDi);
fprintf('Incorrect pixels    = %d \n',incIdD);


%% compare

figure;
subplot(1,4,1); imagesc(xt,[-1 1]);axis image;
axis off; colormap gray; title('true');
subplot(1,4,2); imagesc(xPt,[-1 1]);axis image;
axis off; colormap gray; title('PINV');
subplot(1,4,3); imagesc(xD,[-1 1]);axis image;
axis off; colormap gray; title('Dual');
if totSol > 1 
    subplot(1,4,4); imagesc(Xint,[-1 1]);axis image;
    axis off; colormap gray; title('Int');
end
