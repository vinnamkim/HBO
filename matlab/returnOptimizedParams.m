function [model, vardist] = returnOptimizedParams(model, vardist, W)
%

% place W to the variational distribution and to the model structure

if strcmp(model.GP.type, 'seMaha')
    vardist.means = reshape( W(1:(model.D*model.K)), model.K, model.D);
    vardist.logsigma = reshape( W((model.D*model.K+1):(2*model.K*model.D)), model.K, model.D);
    vardist.covars = reshape( exp(2*W((model.D*model.K+1):(2*model.K*model.D))), model.K, model.D);
    % inducing variables parameters
    st = 2*model.K*model.D + 1;
    en = 2*model.K*model.D + model.M*model.K; 
    model.Xu = reshape(W(st:en), model.M,  model.K);    
elseif strcmp(model.GP.type, 'seARD')
    vardist.means = reshape( W(1:model.D), 1, model.D);
    vardist.logsigma = reshape( W((model.D+1):(2*model.D)), 1, model.D);
    vardist.covars = reshape( exp(2*W((model.D+1):(2*model.D))), 1, model.D);
    % inducing variables parameters
    st = 2*model.D + 1;
    en = 2*model.D + model.D;
    model.V = reshape( W(st:en), 1, model.D);
    model.Xu = model.X(model.subsetInd,:).*repmat(model.V, model.M , 1) ;  
end

% likelihood hyperparameters
st = en + 1; 
en = en + model.Likelihood.nParams;
WW = W(st:en);
model.Likelihood.logtheta = WW(:)'; 
model.Likelihood.sigma2f = exp(2*model.Likelihood.logtheta(1));
model.Likelihood.sigma2 = exp(2*model.Likelihood.logtheta(2));

% Variances of the Gaussian prior over the the matrix A
st = en + 1; 
en = en + model.priorA.nParams;
WW = W(st:en);
WW = WW(:)';
model.priorA.logsigma = WW; % each row in A share the same variance
model.priorA.sigma2 = exp(2*WW); % each row in A share the same variance

