function [mustar] = dmgpPredict(model, vardist,  Xtest)
%
%

sigma2 = model.Likelihood.sigma2;
sigma2f = model.Likelihood.sigma2f;

model.Kmm = model.Xu*model.Xu';
dgK = diag(model.Kmm);
model.Kmm =  bsxfun(@plus, dgK, dgK') - 2*model.Kmm;
model.dXXu = model.Kmm;
model.Kmm = exp(-0.5*model.Kmm);          
[model.psi0, model.Psi1, model.Psi2, model.outPsi2, model.sumPsi2, vardist.Mu, vardist.XnSigmaXn] = kernVardistPsiCompute(model, vardist);

Ntest = size(Xtest,1);
M = model.M;
K = model.K;
Z = model.Xu;
% Psi1_test
Mu = Xtest*vardist.means'; 
XnSigmaXn = (Xtest.*Xtest)*vardist.covars';
argExp = zeros(Ntest,M); 
normfactor = ones(Ntest,1);
for k=1:K
%
    normfactor = normfactor.*(XnSigmaXn(:,k)  + 1);
    Mu_q = Mu(:,k); 
    Z_q = Z(:,k)';
    distan = (repmat(Mu_q,[1 M]) - repmat(Z_q,[Ntest 1])).^2;
    argExp = argExp + repmat(1./(XnSigmaXn(:,k) + 1), [1 M]).*distan;
%
end
normfactor = normfactor.^0.5;
Psi1_test = repmat(1./normfactor,[1 M]).*exp(-0.5*argExp); 


if ~isfield(model,'alpha')
    Lm = chol(model.Kmm + model.jitter*eye(model.M));
    invLm = Lm\eye(model.M);            
    C = invLm'*(model.Psi2*invLm);                       
    A = sigma2*eye(model.M) + sigma2f*C;           
    La = chol(A);                                     
    invLa =  La\eye(model.M);        
    invLmInvLa = invLm*invLa;               
    invA = invLmInvLa*invLmInvLa';     
    size(invA)
    size(model.Psi1)
    model.alpha = invA*(model.Psi1'*model.Y);
end

% mean prediction 
mustar = sigma2f*(Psi1_test*model.alpha); 


% if nargout > 1
%    % 
%    % precomputations
%    vard = vardistCreate(zeros(1,model.q), model.q, 'gaussian');
%    Kinvk = (model.invK_uu - (1/model.beta)*Ainv);
%    %
%    for i=1:size(vardistX.means,1)
%       %
%       vard.means = vardistX.means(i,:);
%       vard.covars = vardistX.covars(i,:);
%       % compute psi0 term
%       Psi0_star = kernVardistPsi0Compute(model.kern, vard);
%       % compute psi2 term
%       Psi2_star = kernVardistPsi2Compute(model.kern, vard, model.X_u);
%     
%       vars = Psi0_star - sum(sum(Kinvk.*Psi2_star));
%       
%       for j=1:model.d
%          %[model.alpha(:,j)'*(Psi2_star*model.alpha(:,j)), mu(i,j)^2]
%          varsigma(i,j) = model.alpha(:,j)'*(Psi2_star*model.alpha(:,j)) - mu(i,j)^2;  
%       end
%       varsigma(i,:) = varsigma(i,:) + vars; 
%       %
%    end
%    % 
%    if isfield(model, 'beta')
%       varsigma = varsigma + (1/model.beta);
%    end
%    %
% end
% 
