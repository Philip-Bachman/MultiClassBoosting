function [ l_count ] = plot_influence( X, Y, lrnr, grid_res, grid_buffer )
% Plot output of an influence learner, learned for 2D data.
%

l_count = lrnr.l_count;
% Compute the extremal coordinates of the evaluation grid
x_min = min(X(:,1)) - grid_buffer;
x_max = max(X(:,1)) + grid_buffer;
y_min = min(X(:,2)) - grid_buffer;
y_max = max(X(:,2)) + grid_buffer;

% Compute a suitable set of grid points at which to evaluate the trees
[ Xg Yg ] = meshgrid(linspace(x_min, x_max, grid_res), linspace(y_min, y_max, grid_res));
Hg = zeros(size(Xg,1),size(Xg,2),l_count);
Gg = zeros(size(Xg,1),size(Xg,2),l_count);
fprintf('Computing values for grid:');
for col=1:grid_res,
    if (mod(col,round(max(1.0,grid_res/60))) == 0)
        fprintf('.');
    end
    col_points = [Xg(:,col) Yg(:,col)];
    [F H G] = lrnr.evaluate(col_points);
    Gn = bsxfun(@rdivide, exp(G), sum(exp(G),2));
    for l=1:l_count,
        Hg(:,col,l) = H(:,l);
        Gg(:,col,l) = Gn(:,l);
    end
end
fprintf('\n');

for l=1:l_count,
    H = squeeze(Hg(:,:,l));
    G = squeeze(Gg(:,:,l));
    figure();
    colormap('bone');
    clines = linspace(min(H(:)),max(H(:)),20);
    subplot(1,2,1);
    contourf(Xg, Yg, H, clines, 'EdgeColor','none'); 
    title(sprintf('Hypothesis %d',l));
    axis square;
    %axis equal;
    clines = linspace(min(G(:)),max(G(:)),20);
    subplot(1,2,2);
    contourf(Xg, Yg, G, clines, 'EdgeColor','none'); 
    title(sprintf('Influence %d',l));
    axis square;
    %axis equal;
end

return

end



