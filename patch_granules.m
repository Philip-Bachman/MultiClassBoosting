function [ g_vals g_coords ] = patch_granules( X, granule_count, granule_coords )
% Compute "granule" values for the given square image patch. Granules are
% computed as sums over all 2x2, 4x4, and 8x8 square subpatches.
%
% Parameters:
%   x: input image patch
%   granule_count: number of granules to compute
%   granule_coords: coordinates of the granules to compute (if not given, they
%                   will be randomly chosen from the list of possible coords)
%
% Output:
%   g_vals: granule values for x
%   g_coords: coordinates of each computed granule
%

im_dim = sqrt(size(X,2));

g2x2_count = (im_dim - 1)^2;
g4x4_count = (im_dim - 3)^2;
g8x8_count = 0; %(im_dim - 7)^2;

if ~exist('granule_count','var')
    granule_count = g2x2_count + g4x4_count + g8x8_count;
end

if ~exist('granule_coords','var')
    % Compute list of possible granule coords if none was explicitly given
    granule_coords = zeros(granule_count, 3);
    idx = 1;
    for step=[1 3], %7],
        for row=1:(im_dim - step),
            for col=1:(im_dim - step),
                granule_coords(idx,1) = row;
                granule_coords(idx,2) = col;
                granule_coords(idx,3) = step + 1;
                idx = idx + 1;
            end
        end
    end
    granule_idx = randsample(size(granule_coords,1), granule_count);
    granule_coords = granule_coords(granule_idx,:);
end
granule_count = size(granule_coords,1);
g_coords = granule_coords(:,:);
g_vals = zeros(size(X,1),granule_count);
% Compute a value for each granule described by the coordinate list
fprintf('Computing granules:');
for i=1:size(X,1),
    if (mod(i, round(size(X,1)/50)) == 0)
        fprintf('.');
    end
    im = reshape(X(i,:),im_dim,im_dim);
    for g_num=1:granule_count,
        g_row = granule_coords(g_num,1);
        g_col = granule_coords(g_num,2);
        g_stp = granule_coords(g_num,3) - 1;
        g_vals(i,g_num) = sum(sum(im(g_row:(g_row+g_stp),g_col:(g_col+g_stp))));
    end
end
fprintf('\n');

return

end

