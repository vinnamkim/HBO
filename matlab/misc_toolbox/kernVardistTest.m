function model = kernVardistTest(N, M, K)

%
%
D = 2*K;

x = randn(K,D);
v = rand(K,D);
Z = randn(M,K);

     
model = dmgpCreate(randn(N,D), randn(N,1), 'Gaussian', K, M);
model.Xu = Z;

vardist.means = x;
vardist.covars = v;

[model.psi0, model.Psi1, model.Psi2, model.outPsi2, model.sumPsi2, vardist.Mu, vardist.XnSigmaXn] = kernVardistPsiCompute(model, vardist);

Tnm = ones(N,M);
Tmm = ones(M,M);
[gMmm, gVmm, gIndmm, gMnm, gVnm, gIndnm] = kernelVardistPsiGradient(model, vardist, Tmm, Tnm);

epsilon = 1e-6;
paramsvar = [vardist.means(:)', vardist.covars(:)'];
params = [paramsvar, Z(:)'];
origParams = params;
origModel = model;

for i = 1:length(params);
%    
  model = origModel;  
  params = origParams;
  params(i) = origParams(i) + epsilon;
  vardist.means = reshape( params(1:(K*D)), K, D); 
  vardist.covars = reshape( params((K*D+1):(2*K*D)), K, D);
  model.Xu = reshape( params((2*D*K+1):end), M, K); 

  [psi0, Psi1, Psi2, outPsi2, sumPsi2, Mu, XnSigmaXn] = kernVardistPsiCompute(model, vardist);

  LplusPsi1(i) = sum(sum(Psi1));
  LplusPsi2(i) = sum(sum(Psi2));

  params(i) = origParams(i) - epsilon;
  vardist.means = reshape( params(1:(K*D)), K, D); 
  vardist.covars = reshape( params((K*D+1):(2*K*D)), K, D);
  model.Xu = reshape( params((2*K*D+1):end), M, K); 
  [psi0, Psi1, Psi2, outPsi2, sumPsi2, Mu, XnSigmaXn] = kernVardistPsiCompute(model, vardist);

  LminusPsi1(i) = sum(sum(Psi1));
  LminusPsi2(i) = sum(sum(Psi2));
%  
end

gLDiffPsi1 = .5*(LplusPsi1 - LminusPsi1)/epsilon;
gPsi1 = [gMnm(:)', gVnm(:)', gIndnm(:)'];
index = [1:(K*D)];
varmeansdiffPsi1 =  abs(gPsi1(index) - gLDiffPsi1(index));
[gPsi1(index); gLDiffPsi1(index)]
pause
index = [(K*D+1):(2*K*D)];
varcovarsdiffPsi1 =  abs(gPsi1(index) - gLDiffPsi1(index));
[gPsi1(index); gLDiffPsi1(index)]
pause
index = [(2*K*D+1):(2*K*D + M*K)]; 
varinddiffPsi1 =  abs(gPsi1(index) - gLDiffPsi1(index)); 
[gPsi1(index); gLDiffPsi1(index)]
pause

fprintf('----- Psi1 term ------- \n');
fprintf('Variational means max diff: %2.6g.\n', max(varmeansdiffPsi1));
fprintf('Variational covars max diff: %2.6g.\n', max(varcovarsdiffPsi1));
fprintf('Inducing inputs max diff: %2.6g.\n', max(varinddiffPsi1));
fprintf('\n');

gLDiffPsi2 = .5*(LplusPsi2 - LminusPsi2)/epsilon;
gPsi2 = [gMmm(:)', gVmm(:)', gIndmm(:)'];
index = [1:(K*D)];
varmeansdiffPsi2 =  abs(gPsi2(index) - gLDiffPsi2(index));
[gPsi2(index); gLDiffPsi2(index)]
pause
index = [(K*D+1):(2*K*D)];
varcovarsdiffPsi2 =  abs(gPsi2(index) - gLDiffPsi2(index));
[gPsi2(index); gLDiffPsi2(index)]
pause
index = [(2*K*D+1):(2*K*D + M*K)]; 
varinddiffPsi2 = abs(gPsi2(index) - gLDiffPsi2(index)); 
[gPsi2(index); gLDiffPsi2(index)]
pause

fprintf('----- Psi2 term ------- \n');
E = eig(model.Psi2); 
fprintf('max and min eigenvalue of Psi2: %2.6g, %2.6g.\n',max(E),min(E));
fprintf('Variational means max diff: %2.6g.\n', max(varmeansdiffPsi2));
fprintf('Variational covars max diff: %2.6g.\n', max(varcovarsdiffPsi2));
fprintf('Inducing inputs max diff: %2.6g.\n', max(varinddiffPsi2));
fprintf('\n');
