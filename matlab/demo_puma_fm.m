
% load data
load('datasets/puma.mat');
addpath misc_toolbox/
     
% Latent dimensionality (specified by the user)
K = 10;

% Number of inducing variables (specified by the user)
M = 50;
% kernel used: either seMaha (squared Mahalanobis) or seARD
kern = 'seMaha'; % 'seMaha' or seARD

% Likelihood (specified by the user.... only Gaussian is allowed at the moment)
Likelihood = 'Gaussian'; 

% create the model 
model = dmgpCreate(X, Y, Likelihood, K, M, kern);

%optimize the model 
% keep signal power and nosie variacne fixed for initilization of the variational
% distrbution
model.fixedSigmas = 1; 
model.Likelihood.logtheta(2) = 0.5*log(0.01); 
model.Likelihood.sigma2 = exp(2*model.Likelihood.logtheta(2)); 
% optimizer 
opt = 'scgFG'; 
init = 'pca';
% number of iterations 
Iters = 30; 
[model, vardist, F] = dmgpTrain(model, Iters, opt, init);


% Let signal to noise free to be learned
model.fixedSigmas = 0;
Iters = 100;
[model, vardist, F] = dmgpTrain(model, Iters, opt, init, vardist);

% checking gradient with Rasmusses code 
%W = extractOptimizedParams(model, vardist);
%checkgrad('dmgpGaussBound', W, 0.000001, model, vardist)

% visualize the projection
load('datasets/oildata.mat');
if strcmp(model.GP.type,'seMaha')
  Mu = model.X*vardist.means';
%elseif strcmp(model.GP.type, 'seARD')
%  Mu = model.X.*repmat(vardist.means,model.N,1);
end

[i, j] = sort(model.priorA.sigma2,'descend');
figure; 
hold on; 
plot(Mu(Y(:,1)==1,j(1)),Mu(Y(:,1)==1,j(2)),'r+'); 
plot(Mu(Y(:,2)==1,j(1)),Mu(Y(:,2)==1,j(2)),'bo');
plot(Mu(Y(:,3)==1,j(1)),Mu(Y(:,3)==1,j(2)),'gx');
plot(model.Xu(:,j(1)), model.Xu(:,j(2)), 'md', 'Linewidth', 2, 'MarkerSize',8);

% visualize the Initial projection
if strcmp(model.GP.type,'seMaha')
  Mu = model.X*vardist.meansInit';
elseif strcmp(model.GP.type, 'seARD')
  Mu = model.X;
end
figure; 
hold on; 
plot(Mu(Y(:,1)==1,j(1)),Mu(Y(:,1)==1,j(2)),'r+'); 
plot(Mu(Y(:,2)==1,j(1)),Mu(Y(:,2)==1,j(2)),'bo');
plot(Mu(Y(:,3)==1,j(1)),Mu(Y(:,3)==1,j(2)),'gx');
plot(model.Xuinit(:,j(1)), model.Xuinit(:,j(2)), 'md', 'Linewidth', 2, 'MarkerSize',8);

