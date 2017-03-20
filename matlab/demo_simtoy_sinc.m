% Generate Data
N = 200;
D = 5;
d = 2;
sigma_f = 0.1;
sigma = 0.001;

[Y, X, W] = Sim_toy_sinc(N, D, d, sigma);

% Latent dimensionality (specified by the user)
K = 4;

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
Iters = 100; 
[model, vardist, F] = dmgpTrain(model, Iters, opt, init);


% Let signal to noise free to be learned
model.fixedSigmas = 0;
Iters = 100;
[model, vardist, F] = dmgpTrain(model, Iters, opt, init, vardist);

model.Likelihood
model.priorA

[i, j] = sort(model.priorA.sigma2,'descend');

proj_X = model.X*vardist.means';
idx = j(3);
Proj = (proj_X(:, idx) - min(proj_X(:, idx))) / (max(proj_X(:, idx)) - min(proj_X(:, idx)));
Orig = (X * W' - min(X * W')) / (max(X * W') - min(X * W'));
hold on
scatter(Proj, model.Y, 'x')
scatter(Orig, Y, 'o')
hold off