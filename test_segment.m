% test image segmentation data from statlib/UCI repository
%
clear;
warning off all;
load('datasets/segment.mat');
X = ZMUV(X);
stream = RandStream.getDefaultStream();
c = clock();
reset(stream,round(1000*c(6)));

mc_opts = struct();
mc_opts.nu = 0.33;
mc_opts.loss_func = @loss_huberhinge;
mc_opts.lam_l1 = 5e-2;
mc_opts.l_count = round(2.0 * numel(unique(Y)));
mc_opts.l_const = @StumpLearner;
mc_opts.extend_all = 1;

obs_count = size(X,1);
test_size = round(obs_count/5);

mc_learners = {};
round_count = 3;
test_accs = zeros(round_count,1);
for t=1:round_count,
    test_idx = randsample(1:obs_count,test_size);
    train_idx = setdiff(1:obs_count,test_idx);
    X_train = X(train_idx,:);
    Y_train = Y(train_idx);
    X_test = X(test_idx,:);
    Y_test = Y(test_idx);
    mc_learner = MultiClassLearner(X_train,Y_train,mc_opts);
    for r=1:8,
        fprintf('==================================================\n');
        fprintf('META ROUND %d...\n',r);
        for i=1:10,
            tidx = randsample(size(X_train,1), round(size(X_train,1)*0.66));
            Xtr = X_train(tidx,:);
            %Xtr = Xtr + bsxfun(@times, randn(size(Xtr)), 0.1 * std(Xtr));
            L = mc_learner.extend(Xtr,Y_train(tidx,:));
            [F H C] = mc_learner.evaluate(X_train);
            a_train = sum(Y_train==C) / numel(Y_train);
            [F H C] = mc_learner.evaluate(X_test);
            a_test = sum(Y_test==C) / numel(Y_test);
            test_accs(t) = a_test;
            fprintf('Round: %d, train_loss: %.4f, train_acc: %.4f, test_acc: %.4f\n',...
                i,L,a_train, a_test);
        end
        mc_learner.set_codewords(X_train,Y_train, 5);
    end
    save('results_segment.mat');
end
