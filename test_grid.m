% Test simple multi modal griddy data
%clear;
opts = struct();
opts.nu = 1.0;
opts.extend_all = 1;
opts.vor_count = 4;
opts.vor_samples = 2000;
opts.max_depth = 1;
opts.bump_count = 1000;
opts.do_opt = 1;


%[X_train Y_train X_test Y_test] = data_grid(1500, 1000, 0.35, 0.45, 4, 7);
fig = figure();
lrnr = BumpLearner(X_train,Y_train,opts);
for r=1:50,
    L = lrnr.extend(X_train,Y_train);
    F = lrnr.evaluate(X_train);
    acc_train = sum(Y_train.*F > 0) / numel(Y_train);
    F = lrnr.evaluate(X_test);
    acc_test = sum(Y_test.*F > 0) / numel(Y_test);
    fprintf('Round %d, loss: %.4f, train_acc: %.4f, test_acc: %.4f\n',...
        r,L,acc_train,acc_test);
%     plot_learner(X_test, Y_test, lrnr, 150, 0.1, fig);
%     drawnow();
%     fig_name = sprintf('vorlrnr_r%d.eps', r);
%     print('-depsc', fig_name);
end