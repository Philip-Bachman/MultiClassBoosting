% Test simple multi modal griddy data
clear;
[X_train Y_train X_test Y_test] = data_grid(1000, 500, 0.35, 0.45, 5, 8);
%shuf_idx = randsample(1:numel(Y_train),round(numel(Y_train)/10));
%Y_train(shuf_idx) = Y_train(shuf_idx(randperm(numel(shuf_idx))));

% Set options for boosted learner, and instantiate some learner class
opts = struct();
opts.loss_func = @loss_huberhinge;
opts.nu = 1.0;
opts.nz_count = 10;
opts.rbf_sigma = 0.25 * mean(std(X_train));
opts.rbf_type = 2;
opts.chain_len = 3;
lrnr = RBFLearner(X_train,Y_train,opts);

% For RBF learner, generate a set of potential RBF centers
%[clust_idx,clust_centers] = kmeans(X_train,250,'emptyaction','singleton');
lrnr.set_rbf_centers(X_train, round(size(X_train,1)/4));
lrnr.lam_l2 = 1e-2;

pause on;
fig = figure();
for r=1:60,
    lrnr.rbf_sigma = (((61-r) / 30) + 0.025) * mean(std(X_train));
    lrnr.set_rbf_centers(X_train, round(size(X_train,1)/4));
    %idx = randsample(size(X_train,1),round(size(X_train,1) * 0.66));
    idx = 1:size(X_train,1);
    Xtr = X_train(idx,:);
    Xtr = Xtr + bsxfun(@times, randn(size(Xtr)), (0.1 * std(Xtr)));
    Ytr = Y_train(idx);
    L = lrnr.extend(Xtr,Ytr);
    F = lrnr.evaluate(X_train);
    acc_train = sum(Y_train.*F > 0) / numel(Y_train);
    F = lrnr.evaluate(X_test);
    acc_test = sum(Y_test.*F > 0) / numel(Y_test);
    fprintf('Round %d, loss: %.4f, train_acc: %.4f, test_acc: %.4f\n',...
        r,L,acc_train,acc_test);
    if (mod(r,20) == 0)
        plot_learner(X_test, Y_test, lrnr, 150, 0.1, fig);
%         for i=2:r+1,
%             scatter(lrnr.rbf_funcs{i}.centers(:,1),...
%                 lrnr.rbf_funcs{i}.centers(:,2),'k+','SizeData',32);
%         end
        drawnow();
    end
end

%%%%%%%%%%%%%%%%%%%%
% DEMO VIDEO STUFF %
%%%%%%%%%%%%%%%%%%%%

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