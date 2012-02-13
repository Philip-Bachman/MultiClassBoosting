function [ fig ] = plot_dann_contours( X, Y, mu, sigma, scale, grid_res, grid_buffer, fig )
%

if ~exist('fig','var')
    fig = figure();
else
    figure(fig);
    cla;
    axis auto;
end
hold on;

% Compute the extremal coordinates of the evaluation grid
x_min = min(X(:,1)) - grid_buffer;
x_max = max(X(:,1)) + grid_buffer;
y_min = min(X(:,2)) - grid_buffer;
y_max = max(X(:,2)) + grid_buffer;

% Compute a suitable set of grid points at which to evaluate the Gauss PDF
[ Xg Yg ] = meshgrid(linspace(x_min, x_max, grid_res), linspace(y_min, y_max, grid_res));
grid_points = [reshape(Xg, numel(Xg), 1) reshape(Yg, numel(Yg), 1)];

bumps = {};
bump = struct();
bump.mu = mu;
bump.sigma = sigma;
bump.scale = scale;
bump.weight = 1.0;
bumps{1} = bump;

% Get the bump effect at each grid point
[ Fg ] = evaluate( grid_points, bumps );

% % Plot the grid points
% colormap('gray');
% scatter(grid_points(:,1), grid_points(:,2), ones(size(grid_points,1),1).*8, Fg);

% Get some quantile levels for contour plots
quants = quantile(Fg, [0.80 0.90 0.95]);

% Reshape PDF values for contour plotting
Fg = reshape(Fg, size(Xg,1), size(Xg,2));



% Plot the bump/dann center
scatter(mu(1), mu(2), 'Marker', '+', 'MarkerEdgeColor', [0.0 0.0 0.0],...
    'LineWidth', 1.0, 'SizeData',32);

% Plot the true points
scatter(X(Y == -1,1), X(Y == -1,2), 'Marker', 'o', 'MarkerEdgeColor', [0.2 0.8 1.0],...
    'LineWidth', 1.0, 'SizeData',32);
scatter(X(Y == 1,1), X(Y == 1,2), 'Marker', 'o', 'MarkerEdgeColor', [1.0 0.6 0.0],...
    'LineWidth', 1.0, 'SizeData',32);

% Add contour lines describing PDF
colormap('gray');
[C,h] = contour(Xg, Yg, Fg, 5);
contour(Xg, Yg, Fg, quants, 'Color', [0.0 0.0 0.0],...
        'LineStyle', '-', 'LineWidth', 1);
contour(Xg, Yg, Fg, [0.5 1.0 2.0], 'Color', [0.5 0.5 0.5],...
        'LineStyle', '-', 'LineWidth', 1);

% [V,D] = eigs(sigma.*scale);
% line([mu(1) mu(1)+(V(1,1)*D(1,1))], [mu(2) mu(2)+(V(2,1)*D(1,1))]);
% line([mu(1) mu(1)+(V(1,2)*D(2,2))], [mu(2) mu(2)+(V(2,2)*D(2,2))]);

axis square;
axis equal;
axis([x_min x_max y_min y_max]);


return

end


function [ F ] = evaluate(X, bumps)
    % Evaluate the bumps underlying this learner.
    %
    % Parameters:
    %   X: observations to evaluate (a bias column will be added)
    %   bumps: bumps to evaluate
    %        
    % Output:
    %   F: joint hypothesis output for each observation in X
    %
    obs_count = size(X,1);
    F = zeros(obs_count,1);
    for i=1:length(bumps),
        b = bumps{i};
        % The remaining bumps require some more computation
        Xp = bsxfun(@minus, X, b.mu);
        dists = sqrt(sum((pinv(b.sigma)^(1/2) * Xp').^2))';
        F = F + (normpdf(dists,0.0,b.scale) .* b.weight);
    end
    return
end



