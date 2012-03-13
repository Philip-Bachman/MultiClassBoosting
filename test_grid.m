% Test simple multi modal griddy data
%clear;
opts = struct();
opts.nu = 0.1;
opts.extend_all = 1;
opts.vor_count = 4;
opts.l_count = 4;
opts.vor_samples = 1000;
opts.max_depth = 1;
opts.bump_count = 1000;
opts.do_opt = 1;


%[X_train Y_train X_test Y_test] = data_grid(1500, 1000, 0.35, 0.45, 4, 7);
%shuf_idx = randsample(1:numel(Y_train),round(numel(Y_train)/10));
%Y_train(shuf_idx) = Y_train(shuf_idx(randperm(numel(shuf_idx))));

fig = figure();
lrnr = InfluenceLearner(X_train,Y_train,opts);
for r=1:100,
    L = lrnr.extend(X_train,Y_train);
    F = lrnr.evaluate(X_train);
    acc_train = sum(Y_train.*F > 0) / numel(Y_train);
    F = lrnr.evaluate(X_test);
    acc_test = sum(Y_test.*F > 0) / numel(Y_test);
    fprintf('Round %d, loss: %.4f, train_acc: %.4f, test_acc: %.4f\n',...
        r,L,acc_train,acc_test);
    if (mod(r,10) == 0)
        plot_learner(X_test, Y_test, lrnr, 150, 0.1, fig);
        drawnow();
    end
end

% aviobj = avifile('mc_lrnr.avi','compression','none','fps',3);
% fig = figure();
% lrnr = MCLearner(X_train,Y_train,opts);
% for r=1:100,
%     L = lrnr.extend(X_train,Y_train);
%     F = lrnr.evaluate(X_train);
%     acc_train = sum(Y_train.*F > 0) / numel(Y_train);
%     F = lrnr.evaluate(X_test);
%     acc_test = sum(Y_test.*F > 0) / numel(Y_test);
%     fprintf('Round %d, loss: %.4f, train_acc: %.4f, test_acc: %.4f\n',...
%         r,L,acc_train,acc_test);
%     plot_learner(X_test, Y_test, lrnr, 150, 0.1, fig);
%     drawnow();
%     fr = getframe(fig);
%     aviobj = addframe(aviobj,fr);
% end
% close(fig);
% aviobj = close(aviobj);
% 
% aviobj = avifile('vor_lrnr.avi','compression','none','fps',3);
% fig = figure();
% lrnr = VorLearner(X_train,Y_train,opts);
% for r=1:100,
%     L = lrnr.extend(X_train,Y_train);
%     F = lrnr.evaluate(X_train);
%     acc_train = sum(Y_train.*F > 0) / numel(Y_train);
%     F = lrnr.evaluate(X_test);
%     acc_test = sum(Y_test.*F > 0) / numel(Y_test);
%     fprintf('Round %d, loss: %.4f, train_acc: %.4f, test_acc: %.4f\n',...
%         r,L,acc_train,acc_test);
%     plot_learner(X_test, Y_test, lrnr, 150, 0.1, fig);
%     drawnow();
%     fr = getframe(fig);
%     aviobj = addframe(aviobj,fr);
% end
% close(fig);
% aviobj = close(aviobj);
% 
% opts.nu = 0.5;
% aviobj = avifile('bump_lrnr.avi','compression','none','fps',3);
% fig = figure();
% lrnr = BumpLearner(X_train,Y_train,opts);
% for r=1:100,
%     L = lrnr.extend(X_train,Y_train);
%     F = lrnr.evaluate(X_train);
%     acc_train = sum(Y_train.*F > 0) / numel(Y_train);
%     F = lrnr.evaluate(X_test);
%     acc_test = sum(Y_test.*F > 0) / numel(Y_test);
%     fprintf('Round %d, loss: %.4f, train_acc: %.4f, test_acc: %.4f\n',...
%         r,L,acc_train,acc_test);
%     plot_learner(X_test, Y_test, lrnr, 150, 0.1, fig);
%     drawnow();
%     fr = getframe(fig);
%     aviobj = addframe(aviobj,fr);
% end
% close(fig);
% aviobj = close(aviobj);