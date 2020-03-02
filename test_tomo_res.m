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

s = RandStream('mt19937ar','Seed',12);
RandStream.setGlobalStream(s);

addpath(genpath([pwd '/bin']));

%% generate true image, projection matrix and data

fprintf('------------- Setting up ---------------- \n')

imName = 'butterfly';
I  = imread([pwd '/images/' imName '.png']); 
I  = double(I);             % convert image to double
I  = I/max(I(:));           % rescale

k  = 4;                     % sampling
I  = I(1:k:end,1:k:end);    

% convert image to pixel values of -1 and 1
xt          = I;
xt(xt<0.5)  = -1;         
xt(xt>0.5)  =  1;

n           = size(xt,1);               % size of image
u           = unique(xt(:));            % unique greylevels


% generate a tomography matrix 
theta       = round(linspace(0,120,6)); % angles (in degrees)
A           = fancurvedtomo(n,theta);    % parallel-beam geometry
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

s1 = sprintf('matrix A: %d x %d',size(A));
s2 = sprintf(['angles = ' num2str(theta)]);
s3 = sprintf(['The noise is ' num2str(sigma)]);
%% LSQR solution
% 

fprintf('----------- LSQR solution --------------- \n');

xP = lsqr(A,b,1e-6,1e4);

% threshold
xPt      = 0*xP;
xPt(xP<0)= -1;
xPt(xP>0)= 1;
xP       = reshape(xP,n,n);
xPt      = reshape(xPt,n,n);

% performance measures
misfitP  = 0.5*norm(A*xPt(:)-b)^2;
jacIdP   = nnz(xPt(:)==xt(:))/nnz(xt(:));
incIdP   = nnz(min(xPt(:).*xt(:),0));

fprintf('misfit           = %.4f \n',misfitP);
fprintf('jaccard index    = %.4f \n',jacIdP);
fprintf('Incorrect pixels = %d \n',incIdP);

sP1 = sprintf('misfit = %.4f',misfitP);
sP2 = sprintf('jaccard index = %.4f',jacIdP);
sP3 = sprintf('Incorrect pixels = %d',incIdP);

%% dual - (first-order optimization method)
% 
% solve: min_{p} |A'*p|_1 + 0.5*|p - b|_2^2
%
% We use first-order method: primal-dual method

fprintf('------------- Dual solution ------------- \n')

options.maxIter = 1e6; 
options.optTol  = 1e-6; 
options.progTol = 1e-10; 
options.savehist= 0;    

[xD,hist]       = solveBT(A,b,options);

% threshold
xDt = xD;
xDt(abs(xD) < 0.999) = 0;  
xDt = sign(xDt);           
xDt = reshape(xDt,n,n);

% performance measures
misfitD = 0.5*norm(A*xDt(:)-b)^2;
jacIdD  = nnz(xDt(:)==xt(:))/nnz(xt(:));
incIdD  = nnz(min(xDt(:).*xt(:),0));
undetD  = nnz(xDt(:)==0);

sD1 = sprintf('misfit = %.4f',misfitD);
sD2 = sprintf('jaccard index = %.4f',jacIdD);
sD3 = sprintf('Incorrect pixels = %d',incIdD);
sD4 = sprintf('Undetermined pixels = %d',undetD);

fprintf('misfit              = %.4f \n',misfitD);
fprintf('jaccard index       = %.4f \n',jacIdD);
fprintf('Incorrect pixels    = %d \n',incIdD);
fprintf('Undetermined pixels = %d \n',undetD);

%% compare

figure;semilogy(hist.opt); hold on;semilogy(hist.er); hold off;
xlabel('iterate');legend('optimality','progress');

fig1 = figure; 
set(fig1, 'Position',  [ 557, 170, 1212, 682])
subplot(2,4,1); imagesc(xt,[-1 1]);axis image;
axis off; colormap gray; title('true');
subplot(2,4,5); text(0,0.75,sprintf('%s\n%s\n%s',s1,s2,s3));axis off;
subplot(3,5,3); imagesc(xP,[-1 1]);axis image;
axis off; colormap gray; title('LSQR');
subplot(3,5,4); imagesc(xPt,[-1 1]);axis image;
axis off; colormap gray; title('(LSQR)_\tau');
subplot(3,5,5); imagesc(xDt,[-1 1]);axis image;
axis off; colormap gray; title('Dual');
subplot(3,5,8); imagesc(xP-xt,[-1 1]);axis image;
axis off; colormap gray; title('difference');
subplot(3,5,9); imagesc(xPt-xt,[-1 1]);axis image;
axis off; colormap gray; title('incorrect pixels');
subplot(3,5,10); imagesc(abs(xDt).*(xDt-xt),[-1 1]);axis image;
axis off; colormap gray; title('incorrect pixels');
subplot(3,5,14); text(0,1,sprintf('%s\n%s\n%s',sP1,sP2,sP3));axis off;
subplot(3,5,15); text(0,1,sprintf('%s\n%s\n%s\n%s',sD1,sD2,sD3,sD4));axis off;

% saveas(fig1,[pwd '/results/' imName],'png');



