function [Y, X, W] = Sim_toy(N, D, d, sigma_f, sigma)
    X = rand(N, D);
    W = randn(d, D);
    %[U, S, V] = svd(W);
    %WW = S * V';
    proj_X = X * W';
    
    r = sum(proj_X.^2, 2);
    
    D = r * ones(1, N) - 2 * (proj_X * proj_X') + (r * ones(1, N))';
    
    K_ff = sigma_f^2 * exp(-0.5 * D) + sigma^2 * eye(N);
    
    K_ff_chol = chol(K_ff, 'lower');
    
    Y = K_ff_chol * randn(N, 1);
end