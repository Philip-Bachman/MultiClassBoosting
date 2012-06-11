function [ L dLdF ] = loss_hingesq(F, Y, idx, loss_vec)
% Compute the objective function value and functional gradient for the squared
% hinge loss for the outputs in F and the (weighted) target classes in Y.
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
fy = F.*Ys;
L = zeros(size(F));
L(fy < 1) = (fy(fy < 1) - 1).^2;
L = Ym .* L;
if (loss_vec ~= 1)
    % Return average loss
    L = sum(L) / numel(Y);
end

if (nargout > 1)
    % Loss gradient with respect to output at each input observation
    dLdF = zeros(size(F));
    dLdF(fy < 1) = (2 * Ys(fy < 1)) .* (fy(fy < 1) - 1);
    dLdF = dLdF .* Ym;
end

return

end
