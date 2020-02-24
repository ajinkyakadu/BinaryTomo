% Tomography test
%
% 2D Tomgoraphy example
%   min_{x} \| A*x - b \|^2 subject to x \in \{-1, 1\}^n
%   
% b - projected data
% A - tomography matrix
% x - (binary) image
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

I  = imread([pwd '/images/spring.png']); 
I  = double(I);             % convert image to double
I  = I/max(I(:));           % rescale

k  = 8;                     % sampling
I  = I(1:k:end,1:k:end);    

% convert image to pixel values of -1 and 1
xt          = I;
xt(xt<0.5)  = -1;         
xt(xt>0.5)  =  1;

n           = size(xt,1);           % size of image
u           = unique(xt(:));        % unique greylevels


% generate a tomography matrix
theta   = round(linspace(0,150,10));% angles (in degrees)
A       = fancurvedtomo(n,theta);   % fan-beam geometry
A       = A/normest(A);             % rescale matrix

bt      = A*xt(:);                  % generate (true) data                

% add noise to data (additive white Gaussian noise)
noiseLevel = 0.1;
noiseB     = randn(size(bt));
noiseB     = noiseLevel*(noiseB/norm(noiseB)*norm(bt));
b          = bt + noiseB;

sigma       = 0.5*norm(A*xt(:)-b)^2;% measure the noise level

fprintf('matrix A and data b generated \n');
fprintf('matrix A: m: %d, n: %d \n',size(A));
fprintf(['angles = ' num2str(theta) '\n']);
fprintf(['The noise is ' num2str(sigma) '\n']);


%% TV setup

D      = finiteDiff(n);
D      = D/normest(D);
lambda = 5e-3;

%% TV solution
% solve:
%   min_{x} |b - A*x|_2^2 + lambda * |D*x|_1
% 
% D is a finite difference matrix approximating the gradient
% lambda is a regularization parameter

fprintf('------------ TV solution ---------------- \n');

options.maxIter = 1e6;
options.optTol  = 1e-6;
options.progTol = 1e-6;
options.saveHist= 0;

[xTV,histTV] = solveTV(A,b,D,lambda,options);

% threshold
xTV(xTV<0) = -1;
xTV(xTV>0) = 1;
xTV        = reshape(xTV,n,n);

% performance measures
misfitP  = norm(A*xTV(:)-b);
jacIdP   = nnz(xTV(:)==xt(:))/nnz(xt(:));
incIdP   = nnz(min(xTV(:).*xt(:),0));

fprintf('misfit           = %.4f \n',misfitP);
fprintf('jaccard index    = %.2f \n',jacIdP);
fprintf('Incorrect pixels = %d \n',incIdP);


%% dualTV - (using first-order optimization method)
% 
% solve: min_{p} |A'*p+D'*q|_1 + 0.5*|p - b|_2^2
%        subject to |q|_inf <= lambda
%
% We use first-order method: primal-dual algorithm

fprintf('----------- Dual-TV solution ------------ \n')

options.maxIter = 1e6; 
options.optTol  = 1e-9; 
options.progTol = 1e-9; 
options.savehist= 0;    

[xDTV,hist]     = solveTVBT(A,b,D,lambda,options);

% threshold
xDTVt = xDTV;
xDTVt(abs(xDTV) < 0.99) = 0;  
xDTVt = sign(xDTVt);           
xDTVt = reshape(xDTVt,n,n);

% performance measures
misfitD = norm(A*xDTVt(:)-b);
jacIdD  = nnz(xDTVt(:)==xt(:))/nnz(xt(:));
incIdD  = nnz(min(xDTVt(:).*xt(:),0));
undetD  = nnz(xDTVt(:)==0);

fprintf('misfit              = %.4f \n',misfitD);
fprintf('jaccard index       = %.2f \n',jacIdD);
fprintf('Incorrect pixels    = %d \n',incIdD);
fprintf('Undetermined pixels = %d \n',undetD);

%% compare

figure; subplot(1,2,1); semilogy(histTV.opt); title('optimality - TV');
subplot(1,2,2); semilogy(histTV.er); title('error - TV');

figure; subplot(1,2,1); semilogy(hist.opt); title('optimality - TVDual');
subplot(1,2,2); semilogy(hist.er); title('error - TVDual');

figure; 
subplot(2,3,1); imagesc(xt,[-1 1]);axis image;
axis off; colormap gray; title('true');
subplot(3,3,2); imagesc(xTV,[-1 1]);axis image;
axis off; colormap gray; title('TV');
subplot(3,3,3); imagesc(xDTVt,[-1 1]);axis image;
axis off; colormap gray; title('Dual-TV');
subplot(3,3,5); imagesc(xTV-xt,[-1 1]);axis image;
axis off; colormap gray; title('TV - error');
subplot(3,3,6); imagesc(xDTVt-xt,[-1 1]);axis image;
axis off; colormap gray; title('Dual-TV - error');


