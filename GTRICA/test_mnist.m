% test multiclass MNIST digit classification using a GTRICA base learner
% CHANGE
% Good settings:
%   ????
%

clear;
warning off all;
load('mnist.mat');
stream = RandStream.getDefaultStream();
c = clock();
reset(stream,round(1000*c(6)));

mc_opts = struct();
mc_opts.loss_func = @loss_hinge;
mc_opts.l_count = 30;
mc_opts.l_const = @GtricaLearner;
mc_opts.l_opts.alpha = 1.0;
mc_opts.l_opts.use_sosm = 1;
mc_opts.l_opts.group_size = 4;
mc_opts.l_opts.group_count = 6;
mc_opts.l_opts.l_logreg = 1e-3;
mc_opts.l_opts.l_class = 75.0;
mc_opts.l_opts.l_smooth = 1e-3;
mc_opts.l_opts.l_spars = 0.0;
mc_opts.l_opts.ab_iters = 8;

obs_count = size(X_all,1);
train_size = 40000;
test_size = 5000;

for t=1:1,
    train_idx = randsample(1:obs_count, train_size);
    test_idx = setdiff(1:obs_count, train_idx);
    test_idx = randsample(test_idx, test_size);
    X_train = ZMUN(double(X_all(train_idx,:)));
    Y_train = Y_all(train_idx);
    X_test = ZMUN(double(X_all(test_idx,:)));
    Y_test = Y_all(test_idx);
    mc_learner = MultiClassLearner(X_train,Y_train,mc_opts);
    for j=1:100,
        fprintf('RESAMPLING TRAINING SET\n');
        clear('X_train');
        train_idx = setdiff(1:obs_count, test_idx);
        train_idx = randsample(train_idx, train_size);
        X_train = ZMUN(double(X_all(train_idx,:)));
        Y_train = Y_all(train_idx);
        L = mc_learner.extend(X_train,Y_train);
        [F Cf] = mc_learner.evaluate(X_train);
        a_train = sum(Y_train==Cf) / numel(Y_train);
        [F Cf] = mc_learner.evaluate(X_test);
        a_test = sum(Y_test==Cf) / numel(Y_test);
        fprintf('==========================================================\n');
        fprintf('==========================================================\n');
        fprintf('Rnd: %d, loss: %.4f, tr_acc: %.4f, te_acc: %.4f\n',...
            j,L,a_train, a_test);
        fprintf('==========================================================\n');
        fprintf('==========================================================\n');
        mc_learner.set_codewords(X_train,Y_train);
    end
end
% fprintf('==================================================\n');
% fprintf('Building meta learner...\n');
% fprintf('==================================================\n');
% % Build a learner on top of the output of previous learner
% F_train = mc_learner.evaluate(X_train);
% F_test = mc_learner.evaluate(X_test);
% mc_opts = rmfield(mc_opts,'l_count');
% mc_opts.l_const = @StumpLearner;
% meta_learner = MultiClassLearner(F_train,Y_train,mc_opts);
% for i=1:50,
%     tidx = randsample(1:size(F_train,1), 4000);
%     L = meta_learner.extend(F_train(tidx,:),Y_train(tidx));
%     [F Cf] = meta_learner.evaluate(F_train);
%     a_train = sum(Y_train==Cf) / numel(Y_train);
%     [F Cf] = meta_learner.evaluate(F_test);
%     a_test = sum(Y_test==Cf) / numel(Y_test);
%     fprintf('Round: %d, train_loss: %.4f, train_acc: %.4f, test_acc: %.4f\n',...
%         i,L,a_train, a_test);
% end