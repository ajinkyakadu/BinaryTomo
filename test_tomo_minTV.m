% Tomography test with minimum total-variation norm
%
% 2D Tomgoraphy example
%   min_{x} |D*x|_1 subject to 0.5*|A*x - b|^2 <= sigma, x \in {-1, 1}^n
%   
% b - projected data
% A - tomography matrix
% x - (binary) image
% D - finite difference matrix for 2D image gradient
% sigma - noise level
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

I  = imread([pwd '/images/bat.png']); 
I  = double(I);             % convert image to double
I  = I/max(I(:));           % rescale

k  = 16;                     % sampling
I  = I(1:k:end,1:k:end);    

% convert image to pixel values of -1 and 1
xt          = I;
xt(xt<0.5)  = -1;         
xt(xt>0.5)  =  1;

n           = size(xt,1);           % size of image
u           = unique(xt(:));        % unique greylevels


% generate a tomography matrix
theta   = round(linspace(0,150,3)); % angles (in degrees)
A       = fancurvedtomo(n,theta);   % fan-beam geometry
A       = A/normest(A);             % rescale matrix

bt      = A*xt(:);                  % generate (true) data                

% add noise to data (additive white Gaussian noise)
noiseLevel = 0.0;
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

trueTV = norm(D*xt(:),1);
fprintf('true total-variation = %.4f \n',trueTV);

%% TV solution
% solve:
%   min_{x} |b - A*x|_2^2 + lambda * |D*x|_1
% 
% D is a finite difference matrix approximating the gradient
% lambda is a regularization parameter

fprintf('---------- TV min solution -------------- \n');

options.maxIter = 1e6;
options.optTol  = 1e-6;
options.progTol = 1e-6;
options.saveHist= 0;

[xTV,histTV] = solveTVmin(A,b,D,sigma,options);

% threshold
xTVt        = xTV;
xTVt(xTV<0) = -1;
xTVt(xTV>0) = 1;
xTVt        = reshape(xTVt,n,n);

% performance measures
misfitP  = 0.5*norm(A*xTVt(:)-b)^2;
tvP      = norm(D*xTVt(:),1);
jacIdP   = nnz(xTVt(:)==xt(:))/nnz(xt(:));
incIdP   = nnz(min(xTVt(:).*xt(:),0));

fprintf('misfit           = %.4f \n',misfitP);
fprintf('total-variation  = %.4f \n',tvP);
fprintf('jaccard index    = %.4f \n',jacIdP);
fprintf('Incorrect pixels = %d \n',incIdP);


%% dualTVmin - (using first-order optimization method)
% 
% solve: min_{p} |A'*p+D'*q|_1 
%        subject to |q|_inf <= 1,  0.5*|p - b|_2^2 <= sigma
%
% We use first-order method: primal-dual algorithm

fprintf('--------- Dual-TV min solution ---------- \n')

options.maxIter = 1e6; 
options.optTol  = 1e-6; 
options.progTol = 1e-6; 
options.savehist= 0;    

[xD,hist]     = solveTVminBT(A,b,D,sigma,options);

% threshold
xDt = xD;
xDt(abs(xD) < 0.99) = 0;  
xDt = sign(xDt);           
xDt = reshape(xDt,n,n);

% performance measures
misfitD = 0.5*norm(A*xDt(:)-b)^2;
tvD     = norm(D*xDt(:),1);
jacIdD  = nnz(xDt(:)==xt(:))/nnz(xt(:));
incIdD  = nnz(min(xDt(:).*xt(:),0));
undetD  = nnz(xDt(:)==0);

fprintf('misfit              = %.4f \n',misfitD);
fprintf('total-variation     = %.4f \n',tvD);
fprintf('jaccard index       = %.4f \n',jacIdD);
fprintf('Incorrect pixels    = %d \n',incIdD);
fprintf('Undetermined pixels = %d \n',undetD);

%% compare

figure; subplot(1,2,1); semilogy(histTV.opt); title('optimality - TV');
subplot(1,2,2); semilogy(histTV.er); title('error - TV');

figure; subplot(1,2,1); semilogy(hist.opt); title('optimality - TVDual');
subplot(1,2,2); semilogy(hist.er); title('error - TVDual');

figure; 
subplot(1,3,1); imagesc(xt,[-1 1]);axis image;
axis off; colormap gray; title('true');
subplot(2,3,2); imagesc(xTVt,[-1 1]);axis image;
axis off; colormap gray; title('TV');
subplot(2,3,3); imagesc(xDt,[-1 1]);axis image;
axis off; colormap gray; title('Dual-TV');
subplot(2,3,5); imagesc(xTVt-xt,[-1 1]);axis image;
axis off; colormap gray; title('incorrect pixels');
subplot(2,3,6); imagesc(abs(xDt).*(xDt-xt),[-1 1]);axis image;
axis off; colormap gray; title('incorrect pixels');


