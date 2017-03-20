function [gMmm, gVmm, gIndmm, gMnm, gVnm, gIndnm] = kernelVardistPsiGradient(model, vardist, Tmm, Tnm)
%
%


if strcmp(model.GP.type, 'seMaha')
    gMnm = zeros(size(vardist.means)); 
    gVnm = zeros(size(vardist.means)); 
    gMmm1 = zeros(model.N, model.K); 
    gVmm1 = zeros(model.N, model.K); 
elseif strcmp(model.GP.type, 'seARD')       
    gMnm = zeros(size(vardist.means)); 
    gVnm = zeros(size(vardist.means)); 
    gMmm1 = zeros(model.N, model.K); 
    gVmm1 = zeros(model.N, model.K); 
end

%gIndmm = zeros(size(model.Xu)); 
gIndnm = zeros(size(model.Xu)); 

N = model.N;
M = model.M;
K = model.K;
Z = model.Xu;

Psi1Tnm = model.Psi1.*Tnm;

% compute the gradient wrt variational means, variances and inducing inputs
% Psi1 term
for k=1:K
%
    S_k = vardist.XnSigmaXn(:,k);  
    Mu_k = vardist.Mu(:,k); 
    Z_k = model.Xu(:,k)'; 
    
    B_k = (repmat(Mu_k,[1 M]) - repmat(Z_k,[N 1]))./repmat(S_k + 1, [1 M]);
    
    % derivatives wrt variational means and inducing inputs 
    tmp = (B_k.*Psi1Tnm);
    
    % inducing inputs: you sum out the rows 
    gIndnm(:,k) = sum(tmp,1)'; 
    
    B_k = (B_k.*(repmat(Mu_k,[1 M]) - repmat(Z_k,[N 1])));
     
    gV = sum((Psi1Tnm./repmat(S_k + 1, [1 M])).*(B_k - 1),2);
    
    if strcmp(model.GP.type, 'seMaha')
        % variational means: you sum out the columns (see report)
        gMnm(k,:) = -sum(tmp,2)'*model.X; 
        % gradient wrt variational covars (diagonal covariance matrices) 
        gVnm(k,:) = gV'*model.XX;
    elseif strcmp(model.GP.type, 'seARD')
        % variational means: you sum out the columns (see report)
        gMnm(k) = - sum(tmp,2)'*model.X(:,k); 
        % gradient wrt variational covars (diagonal covariance matrices) 
        gVnm(k) = gV'*model.XX(:,k);
    end
    %
end
%

gVnm = 0.5*gVnm;


% PSI2 TERM

% 1) line compute 0.5*(z_mk + z_m'k) for any k and store the result in a "M x K x M" 
%  matrix where M is the number of inducing points and K the latent dimension
% 2) line compute the z_mk - z_m'k, for any k
for k=1:K
  ZmZm(:,k,:) = 0.5*(repmat(Z(:,k),[1 1 M]) + repmat(reshape(Z(:,k),[1 1 M]),[M 1 1]));
  ZmDZm(:,k,:) = repmat(Z(:,k),[1 1 M]) - repmat(reshape(Z(:,k),[1 1 M]),[M 1 1]);
end


% compute the terms 2*Xn'*Sigma_k*Xn + 1, for n and k and srore the result in a 
% "N x K" matrix
XnSigmaXnPlusOne = 2*vardist.XnSigmaXn + 1; 
% compute the terms 1/(2*Xn'*Sigma_k*Xn + 1), for n and k and store the result in a 
% "N x K" matrix
oneDXnSigmaXnPlusOne = 1./XnSigmaXnPlusOne; 

Tmm = Tmm.*model.outPsi2;
Tmm = reshape(Tmm,[M 1 M]);
sumPsi2 = reshape(model.sumPsi2,[M 1 M]);
partInd1 = - sum(ZmDZm.*repmat(sumPsi2.*Tmm,[1 K 1]),3);
partInd2 = zeros(M,K);


% Compute the gradient wrt variational means and variational variances  
% /~For loop over training points  
for n=1:N
    %
    %  
    mu_n = vardist.Mu(n,:);  
    AS_n = XnSigmaXnPlusOne(n,:);  
     
    %MunZmZm = repmat(mu_n, [M 1 M]) - ZmZm; 
    MunZmZm = bsxfun(@minus,mu_n,ZmZm);
    %MunZmZmA = MunZmZm./repmat(AS_n,[M 1 M]);
    MunZmZmA =  bsxfun(@rdivide, MunZmZm, AS_n);
    
    %k2Kern_n = sum((MunZmZm.^2).*repmat(oneDXnSigmaXnPlusOne(n,:),[M 1 M]),2);
    k2Kern_n = sum( bsxfun(@times, (MunZmZm.^2), oneDXnSigmaXnPlusOne(n,:)),2); 
    k2Kern_n = exp(-k2Kern_n)/prod(sqrt(AS_n));
    
    % derivatives wrt to variational means
    %k2ncovG = repmat(k2Kern_n.*Tmm,[1 K 1]);
    %tmp = MunZmZmA.*k2ncovG;   
    k2ncovG = k2Kern_n.*Tmm;
    tmp = bsxfun(@times, MunZmZmA, k2ncovG);
    tmp = sum(tmp,3);
    gMmm1(n,:) = - 2*(sum(tmp,1));
    
    % derivatives wrt inducing inputs 
    partInd2 = partInd2 + tmp;
     
    MunZmZmA = MunZmZmA.*MunZmZm; 
    %gVmm1(n,:) = sum(sum(repmat(oneDXnSigmaXnPlusOne(n,:),[M 1 M]).*(2*MunZmZmA - 1).*k2ncovG,1),3);
    tmp  = bsxfun(@times,  2*MunZmZmA - 1, k2ncovG);
    gVmm1(n,:) = sum(sum( bsxfun(@times, tmp, oneDXnSigmaXnPlusOne(n,:)),1),3);    
    %
end

gIndmm = partInd1 + 2*partInd2; 
 
if strcmp(model.GP.type, 'seMaha')
   gMmm = gMmm1'*model.X; 
   gVmm = gVmm1'*model.XX;
elseif strcmp(model.GP.type, 'seARD')
   gMmm = sum(gMmm1.*model.X,1); 
   gVmm = sum(gVmm1.*model.XX,1);
end