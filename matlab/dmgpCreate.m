function model = dmgpCreate(X, Y, Likelihood, K, M, kern)
%function model = dmgpCreate(X, Y, Likelihood, K, M)
%
%
%

if nargin < 6
    kern = 'seMaha';
end
if strcmp(kern, 'seARD')
    K=size(X,2); 
end

model.type = 'dmgpmodel';
    
model.Likelihood.type = Likelihood;

[N D] = size(X);
[N Q] = size(Y);
model.N = N;
model.D = D;
model.Q = Q;
model.K = K;
model.M = M;
model.Y = Y; 
model.X = X;
model.XX = model.X.*model.X;   
% randomly initialize pseudo inputs 
model.Xu = randn(M, K);    
model.Xuinit = model.Xu;

model.jitter = 1e-5; 

switch model.Likelihood.type
%    
    case 'Gaussian' % standard regression
    %
         model.Likelihood.nParams = 2; 
         % log(sigma) parameter,  i.e the nosie variance sigma2 =
         % exp(2*log(sigma))  and sigmaf
         model.Likelihood.logtheta = [0.5*log(var(Y(:))) 0.5*log(var(Y(:))/100)];
         % precompute also the dot product of the outputs
         model.YY= sum(Y.*Y,1);
         model.YYall= sum(model.YY);
    case 'Probit'  % binary classifcation
     %
         model.Likelihood.nParams = 1;
         model.Likelihood.learnParams = 0; 
         model.Likelihood.logtheta = 0; 
%
end

model.fixedSigmas = 0;

% Squared-exponential Mahalanobis kernel 
% (!!!this never used with the variational method!!!!)
% it will only be used when we do maximum likelihood estimation of the A
model.GP.type = kern;
if strcmp(model.GP.type, 'seMaha')
   model.GP.A = randn(model.K, model.D);  
   model.GP.nParams = model.K*model.D;
   model.GP.constDiag = 1;
   
   % Gaussian prior over the the matrix A 
   model.priorA.mu = zeros(model.K, model.D);
   % Each row of A has a different variance that allows for ARD 
   % inference
   % (log of the standard deviation)  
   model.priorA.logsigma = zeros(1, model.K); % each row in A share the same variance  
   model.priorA.sigma2 = ones(1, model.K); % each row in A share the same variance
   model.priorA.nParams = K;
elseif strcmp(model.GP.type, 'seARD')
   model.K = D; 
   model.GP.A = randn(1, model.D);  
   model.GP.nParams = model.D;
   model.GP.constDiag = 1;
   
   % Gaussian prior over the the matrix A 
   model.priorA.mu = zeros(1, model.D);
   % Each row of A has a different variance that allows for ARD 
   % inference
   
   % Each row of A has a different variance that allows for ARD 
   % inference
   % (log of the standard deviation)  
   model.priorA.logsigma = zeros(1, model.D); % each row in A share the same variance  
   model.priorA.sigma2 = ones(1, model.D); % each row in A share the same variance
   model.priorA.nParams = D;
end
   