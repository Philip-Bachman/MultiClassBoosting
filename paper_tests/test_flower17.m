clear;

load('datasets/flower17_compact.mat');

X_train = geomean_feats;
Y_train = im_classes;

clear('geomean_feats','mean_feats');

mc_opts = struct();
mc_opts.nu = 0.1;
mc_opts.loss_func = @loss_bindev;
mc_opts.l_count = 3;
mc_opts.l_const = @StumpLearner;
mc_opts.lam_nuc = 0e-2;
mc_opts.lam_dif = 1e-2;

cv_rounds = 15;
cv_test_frac = 0.2;
cv_accs = zeros(cv_rounds,1);
mc_learners = {};
for cv_round=1:cv_rounds,
    te_idx = randsample(size(X_train,1),round(size(X_train,1) * cv_test_frac));
    tr_idx = setdiff(1:size(X_train,1),te_idx);
    Xtr = X_train(tr_idx,:);
    Ytr = Y_train(tr_idx);
    Xte = X_train(te_idx,:);
    Yte = Y_train(te_idx);
    mc_learner = NCNukeClassLearner(X_train,Y_train,mc_opts);
    for r=1:100,
        fprintf('==================================================\n');
        fprintf('META ROUND %d...\n',r);
        for i=1:5,
            tidx = randsample(1:size(Xtr,1), round(size(Xtr,1) * 0.5));
            L = mc_learner.extend(Xtr(tidx,:),Ytr(tidx));
            [F H C] = mc_learner.evaluate(Xtr);
            a_train = sum(Ytr==C) / numel(Ytr);
            [F H C] = mc_learner.evaluate(Xte);
            a_test = sum(Yte==C) / numel(Yte);
            cv_accs(cv_round) = a_test;
            fprintf('Round: %d, train_loss: %.4f, train_acc: %.4f, test_acc: %.4f\n',...
                i,L,a_train, a_test);
        end
        mc_learner.set_codewords(Xtr,Ytr,5);
    end
    mc_learners{cv_round} = mc_learner;
    save('test_flower17_res.mat','mc_learners','im_classes','class_names');
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
