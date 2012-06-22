function [ Yh ] = my_knn( Xtr, Ytr, Xte, k )
% do k-nearest-neighbors using Xtr as training set and Ytr as training classes.
%
% For each test observation, check which class minimizes knn distance
%

train_count = size(Xtr,1);
test_count = size(Xte,1);
Xd = zeros(test_count,train_count);
for i=1:train_count,
    Xd(:,i) = sqrt(sum(bsxfun(@minus, Xte, Xtr(i,:)).^2,2));
end
%
classes = unique(Ytr);
class_count = numel(classes);
Xkd = zeros(test_count,class_count);
for i=1:class_count,
    Xc = Xd(:,Ytr==classes(i));
    for j=1:test_count,
        knn_dist = sort(Xc(j,:),'ascend');
        Xkd(j,i) = sum(knn_dist(1:k));
    end
end 
[min_xkd min_idx] = min(Xkd,[],2);
Yh = classes(min_idx);
return
end

