% Test simple binary class ring data
clear;
test_count = 10;
opts = struct();
opts.nu = 0.5;
opts.vor_count = 4;
opts.vor_samples = 1000;
opts.max_depth = 2;
opts.do_opt = 1;

for t_num=1:test_count,
    [X_train Y_train] = data_ring(1000, 4, 2.0, 0.75);
    [X_test Y_test] = data_ring(1000, 4, 2.0, 0.75);
    lrnr = VorLearner(X_train,Y_train,opts);
    fprintf('TEST %d\n',t_num);
    for r=1:100,
        L = lrnr.extend(X_train,Y_train);
        F = lrnr.evaluate(X_train);
        acc_train = sum(Y_train.*F > 0) / numel(Y_train);
        F = lrnr.evaluate(X_test);
        acc_test = sum(Y_test.*F > 0) / numel(Y_test);
        fprintf('    Round %d, loss: %.4f, train_acc: %.4f, test_acc: %.4f\n',...
            r,L,acc_train,acc_test);
    end
end