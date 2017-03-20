function [psi0, Psi1, Psi2, outPsi2, sumPsi2, Mu, XnSigmaXn] = kernVardistPsiCompute(model, vardist)
%
%

N = model.N;
M = model.M;
K = model.K;
Z = model.Xu;

% psi0
psi0 = N;

if strcmp(model.GP.type, 'seMaha')
    Mu = model.X*vardist.means'; 
    XnSigmaXn = model.XX*vardist.covars';
elseif strcmp(model.GP.type, 'seARD') 
    Mu = model.X.*repmat(vardist.means, model.N, 1); 
    XnSigmaXn = model.XX.*repmat(vardist.covars, model.N, 1); 
end
 
% Psi1
argExp = zeros(N,M); 
normfactor = ones(N,1);
for k=1:K
%
    normfactor = normfactor.*(XnSigmaXn(:,k)  + 1);
    Mu_q = Mu(:,k); 
    Z_q = Z(:,k)';
    distan = (repmat(Mu_q,[1 M]) - repmat(Z_q,[N 1])).^2;
    argExp = argExp + repmat(1./(XnSigmaXn(:,k) + 1), [1 M]).*distan;
%
end
normfactor = normfactor.^0.5;
Psi1 = repmat(1./normfactor,[1 M]).*exp(-0.5*argExp); 
 

% Psi2
sumPsi2 = zeros(M,M); 
for n=1:N
    %    
    S = (1 + 2*XnSigmaXn(n,:)).^0.5;  
     
    %Z_n = (repmat(vardist.means(n,:),[M 1]) - Z)*0.5; 
    Z_n = bsxfun(@minus, Mu(n,:), Z)*0.5;
    %Z_n = Z_n.*repmat(sqrt(A)./AS_n,[M 1]);
    Z_n = bsxfun(@times, Z_n, 1./S);
    distZ = dist2(Z_n,-Z_n); 
    
    sumPsi2 = sumPsi2 + exp(-distZ)/prod(S);  
    %
end
    
distZ = dist2(Z,Z);
outPsi2 = exp(-0.25*distZ);

Psi2 = outPsi2.*sumPsi2; 

