clear;

load('datasets/letters.mat');

X_train = ZMUV(X_train);

mc_opts = struct();
mc_opts.nu = 0.33;
mc_opts.loss_func = @loss_huberhinge;
mc_opts.l_count = round(2 * numel(unique(Y_train)));
mc_opts.l_const = @StumpLearner;
mc_opts.lam_l1 = 5e-2;
mc_opts.lam_l2 = 1e-4;
mc_opts.lam_nuc = 1e-4;
mc_opts.lam_dif = 1e-2;

cv_rounds = 10;
cv_test_frac = 0.2;
test_accs = zeros(cv_rounds,1);
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
            Xtr_n = Xtr(tidx,:);
            Xtr_n = Xtr_n + ...
                bsxfun(@times, randn(size(Xtr_n)), 0.1 * std(Xtr_n));
            L = mc_learner.extend(Xtr_n,Ytr(tidx));
            [F H C] = mc_learner.evaluate(Xtr);
            a_train = sum(Ytr==C) / numel(Ytr);
            [F H C] = mc_learner.evaluate(Xte);
            a_test = sum(Yte==C) / numel(Yte);
            test_accs(cv_round) = a_test;
            fprintf('Round: %d, train_loss: %.4f, train_acc: %.4f, test_acc: %.4f\n',...
                i,L,a_train, a_test);
        end
        mc_learner.set_codewords(Xtr,Ytr,5);
    end
    save('results_letters.mat');
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
