% test image segmentation data from statlib/UCI repository
clear;
warning off all;
load('datasets/segment.mat');

mc_opts = struct();
mc_opts.nu = 0.25;
mc_opts.loss_func = @loss_bindev;
mc_opts.l_dim = 10;
mc_opts.l_const = @VecStumpLearner;

obs_count = size(X,1);
test_size = round(obs_count/5);

for t=1:3,
    test_idx = randsample(1:obs_count,test_size);
    train_idx = setdiff(1:obs_count,test_idx);
    X_train = X(train_idx,:);
    Y_train = Y(train_idx);
    X_test = X(test_idx,:);
    Y_test = Y(test_idx);
    mcl_1 = VecMultiClassLearner(X_train,Y_train,mc_opts);
    for r=1:15,
        fprintf('==================================================\n');
        fprintf('META ROUND %d...\n',r);
        mcl_1.lrnr.p = 2.0;
        for i=1:20,
            tidx = randsample(size(X_train,1),round(size(X_train,1)*0.66));
            L = mcl_1.extend(X_train(tidx,:),Y_train(tidx));
            [F H Cf] = mcl_1.evaluate(X_train);
            a_train = sum(Y_train==Cf) / numel(Y_train);
            [F H Cf] = mcl_1.evaluate(X_test);
            a_test = sum(Y_test==Cf) / numel(Y_test);
            fprintf('Round: %d, train_loss: %.4f, train_acc: %.4f, test_acc: %.4f\n',...
                i,L,a_train, a_test);
        end
        mcl_1.set_codewords(X_train,Y_train);
    end
end
