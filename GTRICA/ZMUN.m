function [ X ] = ZMUN( X )
% Set each observation in X to zero mean and unit norm. Smooth by 1e-8 to
% account for rows with zero or near zero norm.

X = bsxfun(@minus, X, mean(X, 2));
X = bsxfun(@rdivide, X, sqrt(sum(X.^2,2) + 1e-8));

return

end

