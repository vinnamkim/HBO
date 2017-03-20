function [F, Y, X, W] = Sim_toy_sinc2(N, D, d, sigma)
    X = rand(N, D);
    W = randn(2, D);
    W(:,d+1:D) = 0;
    [~, S, V] = svd(W);
    W = S * V';
    proj_X = X * W';
    
    F1 = sinc(pi * proj_X(:, 1));
    F2 = atan(proj_X(:, 2));
    F2 = F2 / max(abs(F2));
    
    F = [F1, F2];
    
    Y = F1 + F2 + sigma^2 * randn(N, 1);
end