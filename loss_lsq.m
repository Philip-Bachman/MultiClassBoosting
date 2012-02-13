function [ L dLdF ] = loss_lsq(F, Y, idx, loss_vec)
% Compute the objective value and functional gradient for least squares.
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

L = (F - Y).^2;
if (loss_vec ~= 1)
    % Compute a vector of losses
    L = sum(L) / numel(L);
end

if (nargout > 1)
    % Loss gradient with respect to output at each input observation
    dLdF = 2 * (F - Y);
end

return

end
