clear;

load('satimage.mat');

X_train = ZMUV(X_train);
X_test = ZMUV(X_test);

mc_opts = struct();
mc_opts.nu = 1.0;
mc_opts.loss_func = @loss_hinge;
mc_opts.l_count = 10;
mc_opts.l_const = @InfluenceLearner;
mc_opts.lambda = 1e-2;
mc_opts.l_opts.max_depth = 2;
mc_opts.l_opts.extend_all = 1;
mc_opts.l_opts.lambda = 1e-3;
mc_opts.l_opts.l_const = @StumpLearner;
mc_opts.l_opts.l_count = 2;

% fig = figure();
mc_learner = MultiClassLearner(X_train,Y_train,mc_opts);
for r=1:10,
    fprintf('==================================================\n');
    fprintf('META ROUND %d...\n',r);
    for i=1:5,
        tidx = randsample(1:size(X_train,1), 4400);
        L = mc_learner.extend(X_train(tidx,:),Y_train(tidx));
        [F H C] = mc_learner.evaluate(X_train);
        a_train = sum(Y_train==C) / numel(Y_train);
        [F H C] = mc_learner.evaluate(X_test);
        a_test = sum(Y_test==C) / numel(Y_test);
        fprintf('Round: %d, train_loss: %.4f, train_acc: %.4f, test_acc: %.4f\n',...
            i,L,a_train, a_test);
    end
    mc_learner.set_codewords(X_train,Y_train);
%     s_styles = {'ro','go','bo','ko','r+','g+','b+','k+'};
%     l_styles = {'r-','g-','b-','k-','r-.','g-.','b-.','k-.'};
%     [Ft Ht Ct] = mc_learner.evaluate(X_test);
%     Yt = Y_test;
%     Yc = unique(Y_test);
%     figure(fig); clf(); hold on; axis square;
%     for i=1:length(Yc),
%         idx = Yt == Yc(i);
%         c = mc_learner.c_codes(i,:);
%         if (size(Ft,2) == 2)
%             scatter(Ft(idx,1),Ft(idx,2),s_styles{i});
%             line([0 c(1)],[0 c(2)]);
%         else
%             scatter3(Ft(idx,1),Ft(idx,2),Ft(idx,3),s_styles{i});
%             line([0 c(1)],[0 c(2)],[0 c(3)]);
%         end
%     end
%     drawnow();
end

% s_styles = {'ro','go','bo','ko','r+','g+','b+','k+'};
% l_styles = {'r-','g-','b-','k-','r-.','g-.','b-.','k-.'};
% [Ft Ht Ct] = mc_learner.evaluate(X_test);
% Yt = Y_test;
% Yc = unique(Y_test);
% figure(); hold on; axis square;
% for i=1:length(Yc),
%     idx = Yt == Yc(i);
%     c = mc_learner.c_codes(i,:);
%     if (size(Ft,2) == 2)
%         scatter(Ft(idx,1),Ft(idx,2),s_styles{i});
%         line([0 c(1)],[0 c(2)]);
%     else
%         scatter3(Ft(idx,1),Ft(idx,2),Ft(idx,3),s_styles{i});
%         line([0 c(1)],[0 c(2)],[0 c(3)]);
%     end
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
%         [F H C] = mc_learner.evaluate(X_test);
%         a = sum(Y_test==C) / numel(Y_test);
%         fprintf('Round: %d, train_loss: %.4f, test_acc: %.4f\n',i,L,a);
%     end
%     mc_learner.set_codewords(X_train,Y_train);
% end