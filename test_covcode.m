% test MNIST digit data
%

clear;
warning off all;
stream = RandStream.getDefaultStream();
c = clock();
reset(stream,round(1000*c(6)));

% Load MNIST covcode feature data
load('mnist_covcode.mat');
X = Xt_cov;
Y = Yt;
clear('Xt_cov','Xt_omp','Yt');

mc_opts = struct();
mc_opts.nu = 1.0;
mc_opts.loss_func = @loss_hinge;
mc_opts.l_count = 6;
mc_opts.l_const = @StumpLearner;
% mc_opts.l_dim = 10;
% mc_opts.l_const = @VecStumpLearner;
mc_opts.extend_all = 1;

obs_count = size(X,1);
test_size = round(obs_count/5);

round_count = 1;
test_accs = zeros(round_count,1);
for t=1:round_count,
    test_idx = randsample(1:obs_count,test_size);
    train_idx = setdiff(1:obs_count,test_idx);
    X_train = X(train_idx,:);
    Y_train = Y(train_idx);
    X_test = X(test_idx,:);
    Y_test = Y(test_idx);
    clear('X');
    mc_learner = MultiClassLearner(X_train,Y_train,mc_opts);
    %mc_learner.lrnr.p = 0.0;
    for r=1:25,
        fprintf('==================================================\n');
        fprintf('META ROUND %d...\n',r);
        for i=1:10,
            % Do an update of weak learners on subsample of training set
            tidx = randsample(size(X_train,1), round(size(X_train,1)*0.66));
            L = mc_learner.extend(X_train(tidx,:),Y_train(tidx,:));
            % Evaluate loss/acc on the subsample
            [F H C] = mc_learner.evaluate(X_train(tidx,:));
            a_train = sum(Y_train(tidx)==C) / numel(Y_train(tidx));
            % Evaluate loss/acc on the test set
            [F H C] = mc_learner.evaluate(X_test);
            a_test = sum(Y_test==C) / numel(Y_test);
            if (a_test > test_accs(t))
                test_accs(t) = a_test;
            end
            fprintf('  Round: %d, train_loss: %.4f, train_acc: %.4f, test_acc: %.4f\n',...
                i,L,a_train, a_test);
        end
        mc_learner.set_codewords(X_train,Y_train);
    end
end