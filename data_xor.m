function [ X Y ] = data_xor(c1_count, c2_count, sigma)
% Make a simple 2D xor dataset to test classifier training.
%
% Parameters:
%   c1_count: the number of class 1 examples
%   c2_count: the number of class 2 examples
%   sigma: the standard deviation of the xor clusters
%
% Output:
%   X: the observations generated
%   Y: the class of each generated observation
%

c1_x = randi(2,c1_count,1);
c1_x = c1_x - ones(c1_count,1);
c1_y = c1_x;
c2_x = randi(2,c2_count,1);
c2_x = c2_x - ones(c2_count,1);
c2_y = ones(c2_count,1) - c2_x;

% Generate XOR data
X = [c1_x c1_y; c2_x c2_y] + (randn(c1_count+c2_count, 2) .* sigma);
% Apply a random rotation to the data
theta = rand() * 2 * pi;
rot_mat = [cos(theta) -sin(theta); sin(theta) cos(theta)];
%X = X * rot_mat;
% Set classes for each observation
Y = [-ones(c1_count,1); ones(c2_count,1)];

return

end

