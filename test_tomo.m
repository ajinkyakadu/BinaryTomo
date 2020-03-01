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

clc; clearvars; 

s = RandStream('mt19937ar','Seed',12);
RandStream.setGlobalStream(s);

addpath(genpath([pwd '/bin']));

%% generate true image, projection matrix and data

fprintf('------------- Setting up ---------------- \n')

I  = imread([pwd '/images/apple.png']); 
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
theta       = round(linspace(0,150,4)); % angles (in degrees)
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
xPt      = xP;
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


%% dual - (first-order optimization method)
% 
% solve: min_{p} |A'*p|_1 + 0.5*|p - b|_2^2
%
% We use first-order method: primal-dual method

fprintf('------------- Dual solution ------------- \n')

options.maxIter = 1e6; 
options.optTol  = 1e-6; 
options.progTol = 1e-6; 
options.savehist= 0;    
options.updateGamma = 1; 

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

fprintf('misfit              = %.4f \n',misfitD);
fprintf('jaccard index       = %.4f \n',jacIdD);
fprintf('Incorrect pixels    = %d \n',incIdD);
fprintf('Undetermined pixels = %d \n',undetD);

%% compare

figure;semilogy(hist.opt); hold on;semilogy(hist.er); hold off;
xlabel('iterate');legend('optimality','progress');

fig1 = figure; 
subplot(1,4,1); imagesc(xt,[-1 1]);axis image;
axis off; colormap gray; title('true');
subplot(2,4,2); imagesc(xP,[-1 1]);axis image;
axis off; colormap gray; title('LSQR');
subplot(2,4,3); imagesc(xPt,[-1 1]);axis image;
axis off; colormap gray; title('(LSQR)_\tau');
subplot(2,4,4); imagesc(xDt,[-1 1]);axis image;
axis off; colormap gray; title('Dual');
subplot(2,4,7); imagesc(xPt-xt,[-1 1]);axis image;
axis off; colormap gray; title('incorrect pixels');
subplot(2,4,8); imagesc(abs(xDt).*(xDt-xt),[-1 1]);axis image;
axis off; colormap gray; title('incorrect pixels');



