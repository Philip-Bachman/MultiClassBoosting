% Process stupid matlab crash sucks on linux, can't save files, crashes
% when editing figures, nauseatingly poor port.

c_codes = mc_learners{1}.c_codes;
code_dist_euclid = pdist(c_codes,'euclidean');
code_dist_cosine = pdist(c_codes,'cosine');
code_dist_euclid = zeros(size(code_dist_euclid));
code_dist_cosine = zeros(size(code_dist_cosine));

% Compute average euclidean distance between class code vectors
for i=1:numel(mc_learners),
    code_dist_euclid = ...
        code_dist_euclid + pdist(mc_learners{i}.c_codes,'euclidean');
end
code_dist_euclid = squareform(code_dist_euclid ./ numel(mc_learners));

% Compute average cosine distance between class code vectors
for i=1:numel(mc_learners),
    code_dist_cosine = ...
        code_dist_cosine + pdist(mc_learners{i}.c_codes,'cosine');
end
code_dist_cosine = squareform(code_dist_cosine ./ numel(mc_learners));

% Compute principal components of the pairwise average code distances
[COEFF SCORE LATENT] = princomp(code_dist_euclid);
coclust_counts = zeros(17,17);
for i=1:50,
    [IDX, C, SUMD] = kmeans(SCORE(:,1:15), 4);
    for j=1:17,
        coclust_counts(j,IDX == IDX(j)) = coclust_counts(j,IDX == IDX(j)) + 1;
    end
end
coclust_counts = coclust_counts - diag(diag(coclust_counts));
coclust_vec = squareform((50 - coclust_counts) - (50 * eye(17)));

save('flower17_code_info.mat','code_dist_euclid','code_dist_cosine',...
    'coclust_counts','coclust_vec','class_names');

