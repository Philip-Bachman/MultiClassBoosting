% Test multiclass clasification on iris data
clear;
load('iris.mat');
obs_count = size(iris_x,1);

l_opts = struct();
l_opts.loss_func = @loss_bindev;
l_opts.l_count = 2;
l_opts.nu = 0.1;
l_opts.l_const = @BumpLearner;
l_opts.l_opts.l_count = 3;
l_opts.extend_all = 1;


for t_num=1:1,
    fprintf('==================================================\n');
    fprintf('OUTER ROUND %d\n',t_num);
    test_idx = randsample(obs_count, round(obs_count/5));
    train_idx = setdiff(1:obs_count,test_idx);
    X_train = iris_x(train_idx,:);
    Y_train = iris_y(train_idx);
    X_test = iris_x(test_idx,:);
    Y_test = iris_y(test_idx);
    lrnr = MultiClassLearner(X_train, Y_train, l_opts);
    for j=1:20,
        for r=1:5,
            L = lrnr.extend(X_train,Y_train);
            [F C] = lrnr.evaluate(X_train);
            acc_train = sum(Y_train==C) / numel(Y_train);
            [F C] = lrnr.evaluate(X_test);
            acc_test = sum(Y_test==C) / numel(Y_test);
            fprintf('Round %d, loss: %.4f, train_acc: %.4f, test_acc: %.4f\n',r,L,...
                acc_train,acc_test);
        end
        lrnr.set_codewords(X_train,Y_train);
    end
end
        
    