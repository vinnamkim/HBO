function [model vardist, F] = dmgpTrain(model, Iters, opt, init, vardist)
%
%

if nargin == 2
    opt = 'minimize';
    init = 'random';
elseif nargin == 3
    init = 'random'; 
end

if nargin < 5
%
  if strcmp(model.GP.type, 'seMaha')
     % initialize variational distribution 
     if strcmp(init,'pca')
        [v, vec] = pca(model.X); 
        %[U,S,V] = svd(cov(model.X));
        vardist.means= vec(:,1:model.K)';
     elseif strcmp(init,'random') 
        vardist.means = randn(model.K,model.D);
     end
     Mu = model.X*vardist.means';
     inputScales = 10./(((max(Mu)-min(Mu))).^2);
     vardist.means  = vardist.means.*repmat(inputScales', 1, model.D);
     vardist.meansInit = vardist.means;
     vardist.covars = 1/model.D + (0.001/model.D)*randn(model.K,model.D);
     vardist.logsigma = 0.5*log(vardist.covars);

     %initialize inducing points; 
     Mu = model.X*vardist.means'; 
     perm = randperm(model.N); 
     model.Xu = Mu(perm(1:model.M),:); 
     model.Xuinit = model.Xu; 
  elseif strcmp(model.GP.type, 'seARD')
     vardist.means = 1./((max(model.X)-min(model.X)));
     vardist.meansInit = vardist.means;
     vardist.covars = 1/model.D + (0.001/model.D)*randn(1,model.D);
     vardist.logsigma = 0.5*log(vardist.covars);
     
     %initialize inducing points; 
     Mu = model.X.*repmat(vardist.means,model.N,1); 
     perm = randperm(model.N); 
     
     model.subsetInd = perm(1:model.M);
     model.V = vardist.means;
     model.Xu = model.X(model.subsetInd,:).*repmat(model.V, model.M, 1);
     model.Xuinit = model.Xu; 
  end
%     
end


% visualize the Initial projection
%Mu = model.X*vardist.meansInit';
%figure; 
%hold on; 
%plot(Mu(model.Y(:,1)==1,1),Mu(model.Y(:,1)==1,2),'r+'); 
%plot(Mu(model.Y(:,2)==1,1),Mu(model.Y(:,2)==1,2),'bo');
%plot(Mu(model.Y(:,3)==1,1),Mu(model.Y(:,3)==1,2),'gx');
%plot(model.Xuinit(:,1), model.Xuinit(:,2), 'md', 'Linewidth', 2, 'MarkerSize',8);
%pause

% number of iterations for the minimizer  
FuncEval = -Iters;

% extract the vector of optimized parameters from the model structure
% and optimize the model 
W = extractOptimizedParams(model, vardist);
[F, DF] = dmgpGaussBound(W, model, vardist);

if strcmp(opt, 'minimize')
      % optimize the parameters Carl Rasmussen minimize function 
      [W, fX] = minimize(W, 'dmgpGaussBound',  FuncEval, model, vardist);
      % final value of the variational lower bound 
      F = -fX(end);
elseif strcmp(opt, 'scgFG')
      load  dfoptions; 
      options(2) = 0.1*options(2); 
      options(3) = 0.1*options(3);

      options(1) = 1;
      options(14) = Iters;

      % NETLAB style optimization.
      [W, options] = scgFG('dmgpGaussBound', W',  options, model, vardist); 
          
      % final value of the variational lower bound 
      F = -options(8);
end
                
% place back the optimized parameters in the model structure and variational distribution 
[model, vardist] = returnOptimizedParams(model, vardist, W);