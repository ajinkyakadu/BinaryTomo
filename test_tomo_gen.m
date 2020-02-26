% Generalized Tomography test (with given greylevels)
%
% 2D Tomgoraphy example
%   min_{x} \| A*x - b \|^2 subject to x \in \{u_0, u_1\}^n
%   
% b - projected data
% A - tomography matrix
% x - (binary) image
% u_0, u_1 - greylevels of the image
%
% Created by:
%   - Ajinkya Kadu, Utrecht University
%   Feb 18, 2020

clc; clearvars; 

s = RandStream('mt19937ar','Seed',12);
RandStream.setGlobalStream(s);

addpath(genpath([pwd '/bin']));

%% generate true image, projection matrix and data

fprintf('------------- Setting up ---------------- \n')

I  = imread([pwd '/images/octopus.png']); 
I  = double(I);             % convert image to double
I  = I/max(I(:));           % rescale

k  = 4;                     % sampling
I  = I(1:k:end,1:k:end);    

% convert image to pixel values of u0,u1
u0 = 0;
u1 = 1;
xt = I;
xt(I<0.5) = u0;         
xt(I>0.5) = u1;

p1 = (u1-u0)/2;
p2 = (u1+u0)/2;
xtN= (xt-p2)/p1;

n           = size(xt,1);               % size of image
u           = unique(xt(:));            % unique greylevels


% generate a tomography matrix 
theta       = round(linspace(0,150,8)); % angles (in degrees)
A           = paralleltomo(n,theta);    % parallel-beam geometry
A           = A/normest(A);             % rescale matrix

bt          = A*xt(:);                  % generate (true) data              

% add noise to data (additive white Gaussian noise)
noiseLevel  = 0.0;
noiseB      = randn(size(bt));
noiseB      = noiseLevel*(noiseB/norm(noiseB)*norm(bt));
b           = bt + noiseB;
sigma       = 0.5*norm(A*xt(:)-b)^2;    % measure the noise level

fprintf('matrix A and data b generated \n');
fprintf('matrix A: m: %d, n: %d \n',size(A));
fprintf(['angles = ' num2str(theta) '\n']);
fprintf(['The noise is ' num2str(sigma) '\n']);

%% LSQR solution
% 

fprintf('----------- LSQR solution --------------- \n');

xP = lsqr(A,b,1e-6,1e4);

% threshold
thr      = (u0+u1)/2;
xP(xP<thr) = u0;
xP(xP>thr) = u1;
xP       = reshape(xP,n,n);

xPN      = (xP-p2)/p1;

% performance measures
misfitP  = 0.5*norm(A*xP(:)-b)^2;
jacIdP   = nnz(xPN(:)==xtN(:))/nnz(xtN(:));
incIdP   = nnz(min(xPN(:).*xtN(:),0));

fprintf('misfit           = %.4f \n',misfitP);
fprintf('jaccard index    = %.4f \n',jacIdP);
fprintf('Incorrect pixels = %d \n',incIdP);


%% dual - (first-order optimization method)
% 
% solve: min_{p} |A'*p|_1 + 0.5*|p - b|_2^2
%
% We use first-order method: primal-dual method

fprintf('------------- Dual solution ------------- \n')

options.maxIter = 1e6; 
options.optTol  = 1e-8; 
options.progTol = 1e-8; 
options.savehist= 0;    

[xD,hist]       = solveGenBT(A,b,u,options);

% threshold
xDt = xD;
xDt(xD<=1.02*u0) = u0;
xDt(xD>=0.98*u1) = u1;
xDid = find((abs(xD) > 1.02*u0) & (abs(xD)<0.98*u1));
xDt(xDid) = 0.5*(u0+u1); 
% xDt = p1*sign(xDt) + p2;           
xDt = reshape(xDt,n,n);

xDtN= (xDt-p2)/p1;

% performance measures
misfitD = 0.5*norm(A*xDt(:)-b)^2;
jacIdD  = nnz(xDtN(:)==xtN(:))/nnz(xtN(:));
incIdD  = nnz(min(xDtN(:).*xtN(:),0));
undetD  = nnz(xDtN(:)==0);

fprintf('misfit              = %.4f \n',misfitD);
fprintf('jaccard index       = %.4f \n',jacIdD);
fprintf('Incorrect pixels    = %d \n',incIdD);
fprintf('Undetermined pixels = %d \n',undetD);

%% compare

figure; subplot(1,2,1);semilogy(hist.opt);title('optimality')
subplot(1,2,2);semilogy(hist.er);title('error')

fig1 = figure; 
subplot(1,3,1); imagesc(xt,[u0 u1]);axis image;
axis off; colormap gray; title('true');
subplot(2,3,2); imagesc(xP,[u0 u1]);axis image;
axis off; colormap gray; title('LSQR');
subplot(2,3,3); imagesc(xDt,[u0 u1]);axis image;
axis off; colormap gray; title('Dual');
subplot(2,3,5); imagesc(xPN-xtN,[-1 1]);axis image;
axis off; colormap gray; title('incorrect pixels');
subplot(2,3,6); imagesc(abs(xDtN).*(xDtN-xtN),[-1 1]);axis image;
axis off; colormap gray; title('incorrect pixels');



