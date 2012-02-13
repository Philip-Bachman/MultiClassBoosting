clear;

load('satimage.mat');

bump_count = 150;
mc_opts = struct();
mc_opts.nu = 0.1;
mc_opts.loss_func = @loss_bindev;
mc_opts.l_count = 10;
mc_opts.l_const = @StumpLearner;
mc_opts.l_opts.extend_all = 1;
mc_opts.l_opts.l_const = @StumpLearner;
mc_opts.l_opts.l_count = 2;
mc_opts.l_opts.l_rounds = 4;
mc_opts.l_opts.bump_count = bump_count;
mc_opts.l_opts.lambda = 1e-2;

mc_learner = MultiClassLearner(X_train,Y_train,mc_opts);
for r=1:30,
    fprintf('==================================================\n');
    fprintf('META ROUND %d...\n',r);
    for i=1:5,
        tidx = randsample(1:size(X_train,1), 4000);
        L = mc_learner.extend(X_train(tidx,:),Y_train(tidx));
        [F Cf] = mc_learner.evaluate(X_train);
        a_train = sum(Y_train==Cf) / numel(Y_train);
        [F Cf] = mc_learner.evaluate(X_test);
        a_test = sum(Y_test==Cf) / numel(Y_test);
        fprintf('Round: %d, train_loss: %.4f, train_acc: %.4f, test_acc: %.4f\n',...
            i,L,a_train, a_test);
    end
    mc_learner.set_codewords(X_train,Y_train);
end

% Yc = unique(Y_test);
% s_styles = {'ro','go','bo','ko','r+','g+','b+','k+'};
% l_styles = {'r-','g-','b-','k-','r-.','g-.','b-.','k-.'};
% [F Cf] = mc_learner.evaluate(X_test);
% figure(); hold on; axis square;
% for i=1:length(Yc),
%     idx = Y_test == Yc(i);
%     c = mc_learner.c_codes(i,:) .* 5;
%     scatter3(F(idx,1),F(idx,2),F(idx,3),s_styles{i});
%     line([0 c(1)],[0 c(2)],[0 c(3)]);
% end

    
% for r=1:20,
%     fprintf('STARTING META ROUND %d...\n',r);
%     tidx = randsample(1:size(X_train,1),3000);
%     Xt = X_train(tidx,:);
%     Yt = Y_train(tidx);
%     F = mc_learner.evaluate(Xt);
%     [L dLdFa] = mc_learner.compute_loss_grad(F, Yt);
%     for i=1:length(mc_learner.l_objs),
%         lrnr=mc_learner.l_objs{i};
%         lrnr.set_possi_bumps(Xt,Yt, abs(dLdFa(:,i)), bump_count);
%     end
%     for i=1:5,
%         L = mc_learner.extend(Xt,Yt);
%         [F Cf] = mc_learner.evaluate(X_test);
%         a = sum(Y_test==Cf) / numel(Y_test);
%         fprintf('Round: %d, train_loss: %.4f, test_acc: %.4f\n',i,L,a);
%     end
%     mc_learner.set_codewords(X_train,Y_train);
% end