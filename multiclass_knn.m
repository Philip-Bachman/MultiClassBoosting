function [ C ] = multiclass_knn( X_train, Y_train, X_test, k )
% Perform a multiclass variant of k-nearest neighbors using the data in X_train
% as the set of reference points, and the set of class labels in Y_train for
% determining classifications of the points in X_test.
%
% Parameters:
%   X_train: points to use as reference for determining nearest neighbors
%   Y_train: class labels for each observation in X_train
%   X_test: points for which to determine classes
%   k: the number of nearest neighbors required to achieve a classification
% Output:
%   C: estimated class label for each observation in X_test
%

obs_count = size(X_test,1);
c_labels = unique(Y_train);
c_count = numel(c_labels);
c_idx = false(size(X_train,1),c_count);
for i=1:c_count,
    c_idx(:,i) = (Y_train == c_labels(i));
end
C = zeros(obs_count,1);
fprintf('Computing classifications:');
for i=1:obs_count,
    if (mod(i, round(obs_count/20)) == 0)
        fprintf('.');
    end
    D = sqrt(sum(bsxfun(@minus,X_train,X_test(i,:)).^2,2));
    [d_val idx_perm] = sort(D,'ascend');
    d_idx = c_idx(idx_perm,:);
    cd_min = sum(d_val);
    cl_min = c_labels(1);
    for c=1:c_count,
        cd_knn = d_val(d_idx(:,c));
        if (sum(cd_knn(1:k)) < cd_min)
            cd_min = sum(cd_knn(1:k));
            cl_min = c_labels(c);
        end
    end
    C(i) = cl_min;
end
fprintf('\n');

return
end

