clear;

load('datasets/iris.mat');

X_train = ZMUV(iris_x);
Y_train = iris_y;
clear('iris_x','iris_y');

mc_opts = struct();
mc_opts.nu = 0.05;
mc_opts.loss_func = @loss_huberhinge;
mc_opts.l_count = 3;
mc_opts.l_const = @StumpLearner;
mc_opts.lam_nuc = 0e-3;
mc_opts.lam_dif = 5e-2;
mc_opts.lam_l1 = 0.0;

% fig = figure();
% pause on;
% colors = zeros(150,3);
% for i=1:150,
%     colors(i,Y_train(i)) = 1;
% end

cv_rounds = 30;
cv_test_frac = 0.2;
cv_accs = zeros(cv_rounds,1);
for cv_round=1:cv_rounds,
    te_idx = randsample(size(X_train,1),round(size(X_train,1) * cv_test_frac));
    tr_idx = setdiff(1:size(X_train,1),te_idx);
    Xtr = X_train(tr_idx,:);
    Ytr = Y_train(tr_idx);
    Xte = X_train(te_idx,:);
    Yte = Y_train(te_idx);
    mc_learner = MultiClassLearner(X_train,Y_train,mc_opts);
    for r=1:20,
        fprintf('==================================================\n');
        fprintf('META ROUND %d...\n',r);
        for i=1:5,
            tidx = randsample(1:size(Xtr,1), round(size(Xtr,1) * 0.66));
            Xn = Xtr(tidx,:);
            Xn = Xn + bsxfun(@times, randn(size(Xn)), 0.1 * std(X_train));
            L = mc_learner.extend(Xn,Ytr(tidx));
            [F H C] = mc_learner.evaluate(Xtr);
            a_train = sum(Ytr==C) / numel(Ytr);
            [F H C] = mc_learner.evaluate(Xte);
            a_test = sum(Yte==C) / numel(Yte);
            cv_accs(cv_round) = a_test;
            fprintf('Round: %d, train_loss: %.4f, train_acc: %.4f, test_acc: %.4f\n',...
                i,L,a_train, a_test);
        end
        mc_learner.set_codewords(Xtr,Ytr,5);
%         [U S V] = svd(mc_learner.c_codes);
%         F = mc_learner.evaluate(X_train);
%         C = mc_learner.c_codes;
%         F = F * V(:,1:2);
%         C = C * V(:,1:2);
%         figure(fig);
%         cla; hold on;
%         for c=1:3,
%             cidx = find(Y_train == c);
%             if (c == 1) 
%                 scatter(F(cidx,1),F(cidx,2),'ro');
%             end
%             if (c == 2) 
%                 scatter(F(cidx,1),F(cidx,2),'gs');
%             end
%             if (c == 3) 
%                 scatter(F(cidx,1),F(cidx,2),'b^');
%             end
%         end
%         scatter(C(:,1),C(:,2),64,'k*');
%         axis square;
%         set(gca,'XTick',[],'YTick',[]);
%         box on;
%         drawnow;
%         if (r == 5 || r == 15 || r == 40)
%             f_name = sprintf('iris_dif_off_r%d',r);
%             f_eps = sprintf('%s_eps',f_name);
%             f_fig = sprintf('%s_fig',f_name);
%             print('-f1', '-r600', '-depsc', f_name);
%             hgsave(fig, f_fig);
%             pause(2);
%         end
    end
    save('results_iris.mat');
end

%%%%%%%%%%%%%%%%%%%
% SPARE SCRIPTAGE %
%%%%%%%%%%%%%%%%%%%
% 
% for r=1:5,
%     fprintf('==================================================\n');
%     fprintf('META ROUND %d...\n',r);
%     for i=1:5,
%         tidx = randsample(1:size(Xtr,1), round(size(Xtr,1) * 0.66));
%         L = mc_learner.extend(Xtr(tidx,:),Ytr(tidx));
%         [F H C] = mc_learner.evaluate(Xtr);
%         a_train = sum(Ytr==C) / numel(Ytr);
%         [F H C] = mc_learner.evaluate(Xte);
%         a_test = sum(Yte==C) / numel(Yte);
%         fprintf('Round: %d, train_loss: %.4f, train_acc: %.4f, test_acc: %.4f\n',...
%             i,L,a_train, a_test);
%     end
%     mc_learner.set_codewords(Xtr,Ytr,15);
% end