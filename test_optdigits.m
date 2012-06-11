% Run tests with UCI optdigit data

clear;

load('datasets/optdigits.mat');
stream = RandStream.getDefaultStream();
c = clock();
reset(stream,round(1000*c(6)));

X_train = ZMUV(X_train);
X_test = ZMUV(X_test);
Y_train = Y_train + 1;
Y_test = Y_test + 1;

mc_opts = struct();
mc_opts.nu = 0.33;
mc_opts.loss_func = @loss_huberhinge;
mc_opts.l_count = round(2.0 * numel(unique(Y_train)));
mc_opts.l_const = @StumpLearner;
mc_opts.lam_l1 = 5e-2;
mc_opts.lam_l2 = 1e-4;

cv_rounds = 30;
cv_test_frac = 0.2;
cv_accs = zeros(cv_rounds,1);
for cv_round=1:cv_rounds,
    Xtr = X_train;
    Ytr = Y_train;
    Xte = X_test;
    Yte = Y_test;
    mc_learner = MultiClassLearner(X_train,Y_train,mc_opts);
    for r=1:15,
        fprintf('==================================================\n');
        fprintf('META ROUND %d...\n',r);
        for i=1:5,
            tidx = randsample(1:size(Xtr,1), round(size(Xtr,1) * 0.66));
            Xn = Xtr(tidx,:);
            Xn = Xn + bsxfun(@times, randn(size(Xn)), 0.1 * std(Xn));
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
    end
    save('results_optdigits.mat');
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
