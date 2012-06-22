function [ res ] = test_l2_lsq( args )
%
res = args;
% Set options for minFunc
mf_opts = struct();
mf_opts.Display = 'iter';
mf_opts.Method = 'lbfgs';
mf_opts.Corr = 5;
mf_opts.LS = 3;
mf_opts.LS_init = 3;
mf_opts.MaxIter = 200;
mf_opts.MaxFunEvals = 500;
mf_opts.TolX = 1e-10;
mf_opts.TolFun = 1e-10;

obs_count = 500;
obs_dim = 5;
noise_lvl = 0.1;

A = randn(obs_count,obs_dim);
b = randn(obs_dim,1);

Y = A * b;
Y = Y + ((noise_lvl * std(Y)) * randn(obs_count,1));

X = [A ones(obs_count,1)];

for lam=logspace(-4,1,20),
    fprintf('TESTING LAMBDA: %.4f\n',lam);
    funObj = @( s ) l2_lsq(s, X, Y, lam, 1);
%     fprintf('SGD optimization {\n');
%     w = zeros(obs_dim+1,1);
%     for r=1:50,
%         [ L dLdW ] = funObj(w);
%         w = w - ((2 / (r+10)) * (dLdW ./ norm(dLdW)));
%         err = mean((Y - X*w).^2);
%         fprintf('  round %d, norm(w): %.4f, err: %.8f\n',r,norm(w),err);
%     end
    fprintf('}\n');
    fprintf('minFunc optimization {\n');
    w = minFunc(funObj,zeros(obs_dim+1,1),mf_opts);
    err = mean((Y - X*w).^2);
    fprintf('  norm(w): %.4f, err: %.4f\n',norm(w),err);
    fprintf('}\n');
    fprintf('Analytical optimization {\n');
    w = pinv(X'*X + lam*eye(obs_dim+1)) * X' * Y ;
    err = mean((Y - X*w).^2);
    fprintf('  norm(w): %.4f, err: %.4f\n',norm(w),err);
    fprintf('}\n');
end

return

end

function [ L dLdW ] = l2_lsq(w, X, Y, lam, pen_last)
    % Compute the objective value and gradient for least squares.
    %
    % Parameters:
    %   w: coefficients
    %   X: observations
    %   Y: target outputs
    %   lam: l2 regularization weight for coefficients
    %   pen_last: whether to penalize last (bias) coefficient
    %
    % Output:
    %   L: objective value
    %   dLdF: gradient of objective with respect to values in F
    %
    if ~exist('pen_last','var')
        pen_last = 1;
    end
    F = X * w;
    R = F - Y;
    % Compute least-squares and regularization loss
    L_res = sum(R.^2);
    L_reg = sum(w.^2);
    if (pen_last == 0)
        L_reg = L_reg - w(end).^2;
    end
    L = L_res + L_reg;
    if (nargout > 1)
        % Loss gradient with respect to residual
        dLdR = 2 * R;
        % Loss gradient with respect to coefficients
        dLdW = sum(bsxfun(@times,X,dLdR));
        dLdW = dLdW' + (2 * lam * w);
        if (pen_last == 0)
           dLdW(end) = dLdW(end) - (2 * lam * w(end));
        end
        dLdW = dLdW ./ norm(dLdW);
    end
    return
end