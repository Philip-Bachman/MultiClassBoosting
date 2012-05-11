% test MNIST digit data
%

clear;
warning off all;
% Load MNIST data
load('mnist_data.mat');
X = ZMUV(double(X_train));
Y = Y_train;
% % Remove data, to only learn for a subset of digits
tidx = (Y == 2) | (Y == 3) | (Y == 4) | (Y == 5);
X = X(tidx,:);
Y = Y(tidx);
% % Compute granule features for the remaining data
% [Xg g_coords] = patch_granules(X);
% save('mnist_data_grans.mat');

% load('mnist_data_grans.mat');
stream = RandStream.getDefaultStream();
c = clock();
reset(stream,round(1000*c(6)));

mc_opts = struct();
mc_opts.nu = 0.25;
mc_opts.loss_func = @loss_bindev;
mc_opts.lam_l1 = 0.00;
mc_opts.l_count = 5;
mc_opts.l_const = @TreeLearner;
mc_opts.extend_all = 1;
mc_opts.l_opts.l_const = @StumpLearner;
mc_opts.l_opts.l_count = 3;
mc_opts.l_opts.nz_count = 10;

obs_count = size(X,1);
test_size = round(obs_count/4);

mc_learners = {};
round_count = 10;
test_accs = zeros(round_count,1);
for t=1:round_count,
    test_idx = randsample(1:obs_count,test_size);
    train_idx = setdiff(1:obs_count,test_idx);
    X_train = X(train_idx,:);
    Y_train = Y(train_idx);
%     shuf_idx = randsample(1:numel(train_idx),round(numel(train_idx)/10));
%     Y_train(shuf_idx) = Y_train(shuf_idx(randperm(numel(shuf_idx))));
    X_test = X(test_idx,:);
    Y_test = Y(test_idx);
    mc_learner = SparseClassLearner(X_train,Y_train,mc_opts);
    for r=1:20,
        fprintf('==================================================\n');
        fprintf('META ROUND %d...\n',r);
        for i=1:5,
            tidx = randsample(size(X_train,1), round(size(X_train,1)/2.0));
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
        mc_learner.lam_l1 = mc_learner.lam_l1 * 1.25;
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