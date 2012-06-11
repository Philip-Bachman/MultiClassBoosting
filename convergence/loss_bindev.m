function [ L dLdF ] = loss_bindev(F, Y, idx, loss_vec)
% Compute the objective function value and functional gradient for the weighted
% logistic regression implied by the targets in Y
%
% Parameters:
%   F: function value at each observation
%   Y: weighted classification for each observation
%   idx: indices at which to evaluate loss and gradients
%   loss_vec: if this is 1, return a vector of losses
%
% Output:
%   L: objective value
%   dLdF: gradient of objective with respect to values in F
%
if ~exist('idx','var')
    idx = 1:size(F,1);
end
if ~exist('loss_vec','var')
    loss_vec = 0;
end
F = F(idx);
Y = Y(idx);

% Decompose Y into sign and magnitude components
Ys = sign(Y);
Ym = abs(Y);

% Compute objective function value
L = Ym .* log(exp(-Ys.*F) + 1);
if (loss_vec ~= 1)
    L = sum(L) / numel(L);
end

if (nargout > 1)
    % Loss gradient with respect to output at each input observation
    dLdF = Ym .* (-Ys .* (exp(-Ys.*F) ./ (exp(-Ys.*F) + 1)));
end

return
