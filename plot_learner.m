function [ fig ] = plot_learner( X, Y, lrnr, grid_res, grid_buffer, fig )
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

% Compute a suitable set of grid points at which to evaluate the trees
[ Xg Yg ] = meshgrid(linspace(x_min, x_max, grid_res), linspace(y_min, y_max, grid_res));
Fg = zeros(size(Xg));
fprintf('Computing values for grid:');
for col=1:grid_res,
    if (mod(col,round(max(1.0,grid_res/60))) == 0)
        fprintf('.');
    end
    col_points = [Xg(:,col) Yg(:,col)];
    Fg(:,col) = lrnr.evaluate(col_points);
end
fprintf('\n');

fprintf('Plotting test grid and true samples...\n');
grid_points = [Xg(:) Yg(:)];
grid_y = Fg(:);
% Plot the tree classification for the grid points
scatter(grid_points(grid_y > 0,1), grid_points(grid_y > 0,2),...
        'Marker','.', 'MarkerEdgeColor', [1.0 0.6 0.0], 'SizeData', 16);
scatter(grid_points(grid_y < 0,1), grid_points(grid_y < 0,2),...
        'Marker','.', 'MarkerEdgeColor', [0.2 0.8 1.0], 'SizeData', 16);
    
% Plot the true labeled points
scatter(X(Y < 0,1), X(Y < 0,2), 'Marker', 'o', 'MarkerEdgeColor', [0.2 0.8 1.0],...
    'LineWidth', 1.0, 'SizeData',32);
scatter(X(Y > 0,1), X(Y > 0,2), 'Marker', 'o', 'MarkerEdgeColor', [1.0 0.6 0.0],...
    'LineWidth', 1.0, 'SizeData',32);

% Add contour lines describing classifier output
contour(Xg, Yg, Fg, [-1.0 1.0], 'Color', [0.66 0.66 0.66],...
        'LineStyle', '-', 'LineWidth', 1);
contour(Xg, Yg, Fg, [0.000001], 'Color', [0.0 0.0 0.0],...
        'LineStyle', '-', 'LineWidth', 1);

axis square;
axis equal;

% Temporary stuff for surface plotting.
% figure();
% colormap('bone');
% if (abs(min(Fg(:))) > abs(max(Fg(:))))
%     Fg(Fg < -2*max(Fg(:))) = -2*max(Fg(:));
% else
%     Fg(Fg > -2*min(Fg(:))) = -2*min(Fg(:));
% end
% clines = linspace(min(Fg(:)),max(Fg(:)),20);
% contourf(Xg, Yg, Fg, clines, 'EdgeColor','none');
% axis square;
% axis equal;
% colorbar();

% figure();
% hold on;
% plot(Fg(:,1),'b-');
% plot(Fg(1,:),'r-');

return

end



