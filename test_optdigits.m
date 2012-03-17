% test optdigit data from statlib/UCI repository
%
%
clear;
warning off all;
load('optdigits.mat');
stream = RandStream.getDefaultStream();
c = clock();
reset(stream,round(1000*c(6)));

X_train = ZMUV(X_train);
X_test = ZMUV(X_test);

mc_opts = struct();
mc_opts.nu = 0.1;
mc_opts.loss_func = @loss_bindev;
mc_opts.lam_l2 = 0.5;
mc_opts.lam_l1 = 0.2;
mc_opts.l_count = 20;
mc_opts.l_const = @RegLearner;
mc_opts.extend_all = 1;
mc_opts.l_opts.l_const = @StumpLearner;
mc_opts.l_opts.l_count = 2;

mc_learners = {};
round_count = 10;
test_accs = zeros(round_count,1);
for t=1:round_count,
    mc_learner = SparseClassLearner(X_train,Y_train,mc_opts);
    for r=1:30,
        fprintf('==================================================\n');
        fprintf('META ROUND %d...\n',r);
        for i=1:5,
            tidx = randsample(size(X_train,1), round(size(X_train,1)/1.5));
            ntidx = setdiff(1:size(X_train,1),tidx);
            L = mc_learner.extend(X_train(tidx,:),Y_train(tidx,:));
            [F H C] = mc_learner.evaluate(X_train(tidx,:));
            a_train = sum(Y_train(tidx)==C) / numel(Y_train(tidx));
            [F H C] = mc_learner.evaluate(X_test);
            a_test = sum(Y_test==C) / numel(Y_test);
            if (a_test > test_accs(t))
                test_accs(t) = a_test;
            end
            fprintf('Round: %d, train_loss: %.4f, train_acc: %.4f, test_acc: %.4f\n',...
                i,L,a_train, a_test);
        end
        mc_learner.set_codewords(X_train,Y_train);
        mc_learner.lam_l1 = mc_learner.lam_l1 * 1.1;
    end
    mc_learners{t} = mc_learner;
end

% [F H C] = mc_learners{1}.evaluate(X_test);
% H_meta = zeros(size(H));
% for t=1:round_count,
%     [F H C] = mc_learners{t}.evaluate(X_test);
%     H_meta = H_meta + H;
% end
% [max_vals max_idx] = max(H_meta,[],2);
% C = reshape(mc_learners{1}.c_labels(max_idx),numel(max_idx),1);
% a_test = sum(Y_test==C) / numel(Y_test);
% fprintf('META TEST ACC: %.4f\n',a_test);