clear;

load('datasets/caltech101_mean_compact.mat');

X_train = X_mean;
Y_train = Y;

clear('X_mean','X_geomean','one_idx','train_ones','Y');

mc_opts = struct();
mc_opts.nu = 1.0;
mc_opts.loss_func = @loss_bindev;
mc_opts.l_count = 20;
mc_opts.l_const = @StumpLearner;
mc_opts.lam_nuc = 0e-3;
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
    for r=1:20,
        fprintf('==================================================\n');
        fprintf('META ROUND %d...\n',r);
        for i=1:5,
            tidx = randsample(1:size(Xtr,1), round(size(Xtr,1) * 0.66));
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
    save('test_caltech_res.mat','mc_learners','image_classes','image_names');
    try
        process_caltech101_results();
    catch
        fprintf('PROCESSING RESULTS FAILED\n');
    end
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
