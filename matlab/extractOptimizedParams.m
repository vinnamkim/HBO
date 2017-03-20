function W = extractOptimizedParams(model, vardist)
%
%

M = vardist.means; 

logV = vardist.logsigma; 

if strcmp(model.GP.type,'seMaha')
    Xu = model.Xu;  
    W = [M(:); logV(:); reshape(Xu, model.M*model.K, 1); model.Likelihood.logtheta'; model.priorA.logsigma'];
elseif strcmp(model.GP.type,'seARD')
    W = [M(:); logV(:); model.V(:); model.Likelihood.logtheta'; model.priorA.logsigma'];
end