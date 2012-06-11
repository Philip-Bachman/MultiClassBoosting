function [ L dLdF ] = loss_hyptan(F, Y, idx, loss_vec)
% Compute the objective function value and functional gradient for hyperbolic
% tangent loss, given the values in F and weighted target classes in Y.
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

alp = 1.0;

% Decompose Y into sign and magnitude components
Ys = sign(Y);
Ym = abs(Y);
L = Ym .* (1 - tanh((Ys .* alp) .* F));
if (loss_vec ~= 1)
    % Compute average loss
    L = sum(L) / numel(L);
end


if (nargout > 1)
    % Loss gradient with respect to output at each input observation
    dLdF = (Ym .* ((tanh((Ys .* alp) .* F).^2 - 1) .* (Ys .* alp)));
end

return

end
