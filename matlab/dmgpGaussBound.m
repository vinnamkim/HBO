function [F, DF] = dmgpGaussBound(W, model, vardist)
%
%

[model, vardist] = returnOptimizedParams(model, vardist, W);
sigma2 = model.Likelihood.sigma2;
sigma2f = model.Likelihood.sigma2f;
sigma2fsigma2 = sigma2f/sigma2; 

% COVARIANCE FUNCTION QUANTITIES AND PSI STATISTICS needed in the sparse method: 
% K_mm, psi0, Psi1,  Psi2 

model.Kmm = model.Xu*model.Xu';
dgK = diag(model.Kmm);
model.Kmm =  bsxfun(@plus, dgK, dgK') - 2*model.Kmm;
model.dXXu = model.Kmm;
model.Kmm = exp(-0.5*model.Kmm);
          
[model.psi0, model.Psi1, model.Psi2, model.outPsi2, model.sumPsi2, vardist.Mu, vardist.XnSigmaXn] = kernVardistPsiCompute(model, vardist);
 
% upper triangular Cholesky decomposition 
% (we add jitter to Kmm which implies jitter inducing variables
% however the matrix that is stored is jitter free. 
% The jitter-free matrix is used to compute more precise derivatives; see
% documentation)
Lm = chol(model.Kmm + model.jitter*eye(model.M)); % m x m: L_m^T where L_m is lower triangular   ---- O(m^3) 
invLm = Lm\eye(model.M);                             % m x m: L_m^{-T}                              ---- O(m^3)
Psi1InvLm = model.Psi1*invLm;                       % n x m: K_nm L_m^{-T}                         ---- O(n m^2)  !!!expensive!!!  

C = invLm'*(model.Psi2*invLm);                       % m x m: L_m^{-1}*Kmn*Knm*L_m^{-T}             ---- O(n m^2)  !!!expensive!!! 
A = sigma2*eye(model.M) + sigma2f*C;           % m x m: A = sigma2*I + L_m^{-1}*K_mn*K_nm*L_m^{-T}

% upper triangular Cholesky decomposition 
La = chol(A);                                     % m x m: L_A^T                      ---- O(m^3)
invLa =  La\eye(model.M);                  % m x m: L_A^{-T}                   ---- O(m^3) 

% useful precomputed quantities
YPsi1InvLm = model.Y'*Psi1InvLm;              % 1 x m: y^T*Knm*L_m^{-T}           ---- O(n m)
YPsi1InvLmInvLa = YPsi1InvLm*invLa;          % 1 x m: y^T*Knm*L_m^{-T}*L_A^{-T}  ---- O(m^2)

% COMPUTE NEGATIVE LOWER BOUND
% F_0 + F_1 + F_2 in the report
F012 = - (model.N-model.M)*model.Q*model.Likelihood.logtheta(2) - 0.5*model.N*model.Q*log(2*pi) - ...
         (0.5/sigma2)*(model.YYall) - model.Q*sum(log(diag(La)));
     
% F_3 in the report
F3 = (0.5*sigma2fsigma2)*sum(sum(YPsi1InvLmInvLa.*YPsi1InvLmInvLa));
% Trace term: F_4 + F_5 in the report
TrK = - (0.5*sigma2fsigma2*model.Q)*(model.psi0  - sum(diag(C)) );
F = F012 + F3 + TrK;

% KL divergence term
if strcmp(model.GP.type, 'seMaha')
    varmeanscovs = sum(vardist.means.*vardist.means, 2) + sum(vardist.covars,2);
    KLdiv = - model.D*sum(model.priorA.logsigma) - 0.5*sum(varmeanscovs./(model.priorA.sigma2'))...
            + sum(sum(vardist.logsigma)) + 0.5*model.K*model.D; 
elseif strcmp(model.GP.type, 'seARD')

    varmeanscovs = vardist.means.*vardist.means + vardist.covars;
    KLdiv = - sum(model.priorA.logsigma) - 0.5*sum(varmeanscovs./(model.priorA.sigma2))...
            + sum(vardist.logsigma) + 0.5*model.D;     
end
        
        
% negative bound
F = - (F + KLdiv);

% derivatives
if nargout == 2

% precomputations for the derivatives
invKmm = invLm*invLm';                      % m x m: K_mm^{-1} = L_m^{-T}*L_m^{-1}   
invLmInvLa = invLm*invLa;                   % m x m: L_m^{-T}*L_A^{-T}               
invA = invLmInvLa*invLmInvLa';              % m x m: A^{-1} = L_m^{-T}*L_A^{-T}*L_A^{-1}*L_m^{-1} 
YPsi1InvA = YPsi1InvLmInvLa*invLmInvLa';    % 1 x m: Y^T*Psi1*A^{-1}                

%sigma2*Q*A^{-1} + sigma2f*A^{-1}*Psi1^T*Y*Y^T*Psi1*A^{-1}
TmmPsi2 = (sigma2*model.Q)*invA + sigma2f*(YPsi1InvA'*YPsi1InvA);   

% auxiliary variables useful for the sigma2 and sigma2f derivative
YPsi1InvLmInvLainvLa = YPsi1InvLmInvLa*invLa';      % 1 x m: y^T*Knm*L_m^{-T}*L_A^{-T}*L_A^{-1}   ---- O(m^2)
sigma2aux = sigma2*model.Q*sum(sum(invLa.*invLa)) + sigma2f*sum(sum(YPsi1InvLmInvLainvLa.*YPsi1InvLmInvLainvLa));

% precomputatino for Psi2 
% Q*K_mm^{-1} - sigma2*Q*A^{-1} - sigma2f*A^{-1}*Psi1^T*Y*Y^T*Psi1*A^{-1}
TmmPsi2 = model.Q*invKmm - TmmPsi2;              

% precomputation for Psi1 
TnmPsi1 = model.Y*YPsi1InvA;

% precomputation for Kmm 
% Q*K_mm^{-1} - sigma2*Q*A^{-1} - sigma2f*A^{-1}*Psi1^T*Y*Y^T*Psi1*A^{-1}
% - Q*(sigma2/sigma2f)*K_mm^{-1}*Psi2*K_mm^{-1}
TmmKmm = TmmPsi2 - (model.Q*sigma2fsigma2)*(invLm*(C*invLm'));    

    
% DERIVATIVES OF INDUCING VARIABLE PARAMETERS
DXu = zeros(model.M, model.K);
for k=1:model.K
  DKmm = -( ones(model.M,1)*(model.Xu(:,k)') - model.Xu(:,k)*ones(1,model.M)).*model.Kmm;           
  DXu(:,k) = sum( DKmm.*TmmKmm, 1)';  
end 

[gMmm, gVmm, gIndmm, gMnm, gVnm, gIndnm] = kernelVardistPsiGradient(model, vardist, TmmPsi2, TnmPsi1);

DXu = DXu + (0.5*sigma2fsigma2)*gIndmm + sigma2fsigma2*gIndnm;

if strcmp(model.GP.type, 'seARD')   
  DXu = sum(DXu.*model.X(model.subsetInd,:),1); 
end
   

DM = sigma2fsigma2*gMnm +  (0.5*sigma2fsigma2)*gMmm; 
DV = sigma2fsigma2*gVnm +  (0.5*sigma2fsigma2)*gVmm;


%Add the KL divergence terms  
if strcmp(model.GP.type, 'seMaha')
   DM = DM - vardist.means./repmat(model.priorA.sigma2',1,model.D);
   % variational variances are optimized in the log space
   DV = 2*(DV.*vardist.covars);
  
   DV = DV + 1 - (vardist.covars./repmat(model.priorA.sigma2',1,model.D));
elseif strcmp(model.GP.type, 'seARD')
   
   DM = DM - vardist.means./model.priorA.sigma2;   
   
   % variational variances are optimized in the log space
   DV = 2*(DV.*vardist.covars);
   DV = DV + 1 - vardist.covars./model.priorA.sigma2;
end


% DERIVATIVES OF LIKELIHOOD HYPERPARAMETERS
Dlik = zeros(model.Likelihood.nParams,1);

if model.fixedSigmas == 0 
    %Dlik(1) = sigma2fsigma2*sum(sum(model.Psi2.*TmmPsi2)) + 2*F3 - model.Q*sigma2fsigma2*model.psi0;
    Dlik(1) = - model.Q*model.M + sigma2aux + 2*TrK;
    Dlik(2) = - model.Q*(model.N-model.M) + model.YYall/sigma2 - 2*F3 - sigma2aux - 2*TrK;
end

% DERIVATIVES OF ARD PRIOR VARIANCES
if strcmp(model.GP.type, 'seMaha')
    DpriorA = - model.D + varmeanscovs./(model.priorA.sigma2');
elseif strcmp(model.GP.type, 'seARD')
    DpriorA  = - 1 + varmeanscovs./model.priorA.sigma2;
end

% PUT EVERYTHING TOGETHER AND NEGATE
if strcmp(model.GP.type, 'seMaha')
   DF = - [DM(:); DV(:); reshape(DXu, model.M*model.K, 1); Dlik; DpriorA(:)];
elseif strcmp(model.GP.type, 'seARD')
   DF = - [DM(:); DV(:); DXu(:); Dlik; DpriorA(:)];
end
%
end