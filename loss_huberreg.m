function [ L dLdF ] = loss_huberreg(F, Y, idx, loss_vec)
% Huberized regression loss, for robust regression. This loss is passed to the
% base learners underlying a meta learner, to guide gradient fitting.
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

d = 1.0;
r = Y - F;
sq_idx = abs(r) <= d;

% Compute objective function value. Loss is scaled absolute loss for residuals
% greater than the threshold (i.e. d) and squared loss for residuals <= d.
L = zeros(size(F));
L(sq_idx) = r(sq_idx).^2;
L(~sq_idx) = (2 * d * abs(r(~sq_idx))) - d^2;
if (loss_vec ~= 1)
    % Compute average loss instead of a vector of losses
    L = sum(L) / numel(L);
end

if (nargout > 1)
    % Loss gradient with respect to output at each input observation
    dLdF = zeros(size(F));
    dLdF(sq_idx) = -2 * r(sq_idx);
    dLdF(~sq_idx) = 2 * d * -sign(r(~sq_idx));
end

return

end
