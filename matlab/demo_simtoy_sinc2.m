% Generate Data
N = 200;
D = 10;
d = 10;
sigma_f = 0.1;
sigma = 0.001;

[Fv, Y, X, W] = Sim_toy_sinc2(N, D, d, sigma);

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


for dim = 1:2
    figure
    for order = 1:4
        proj_X =  model.X*vardist.means';
        idx = j(order);
        %Proj = (proj_X(:, idx) - min(proj_X(:, idx))) / (max(proj_X(:, idx)) - min(proj_X(:, idx)));
        Proj = proj_X(:, idx);
        XW = X * W';
        XW = XW(:, dim);
        %Orig = (XW - min(XW)) / (max(XW) - min(XW));
        Orig = XW;
        subplot(2,2,order)
        hold on
        scatter(Proj, Fv(:, dim), 'x')
        scatter(Orig, Fv(:, dim), 'o')
        hold off
    end
end

for dim = 1:2
    figure
    for order = 1:4
        proj_X = model.X*vardist.means';
        idx = j(order);
        Proj = (proj_X(:, idx) - min(proj_X(:, idx))) / (max(proj_X(:, idx)) - min(proj_X(:, idx)));
        XW = X * W';
        XW = XW(:, dim);
        Orig = (XW - min(XW)) / (max(XW) - min(XW));
        subplot(2,2,order)
        hold on
        scatter(Proj, Y, 'x')
        scatter(Orig, Y, 'o')
        hold off
    end
end