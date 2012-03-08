% test image segmentation data from statlib/UCI repository
%
% Good settings:
%   nu: 0.33
%   loss_func: loss_bindev
%   l_count: 8
%   l_const: @MCLearner
%   l_opts.alpha: 0.5
%   l_opts.extend_all: 1
%   l_opts.l_count: 2
%   l_opts.l_const: @StumpLearner
% 5 updates per meta-round (i.e. between code updates), and ~12 meta rounds
%
clear;
warning off all;
load('segment.mat');
stream = RandStream.getDefaultStream();
c = clock();
reset(stream,round(1000*c(6)));

mc_opts = struct();
mc_opts.nu = 0.1;
mc_opts.loss_func = @loss_bindev;
mc_opts.lambda = 1e-1;
mc_opts.l_count = 8;
mc_opts.l_const = @InfluenceLearner;
mc_opts.extend_all = 1;
mc_opts.l_opts.l_const = @StumpLearner;
mc_opts.l_opts.l_count = 2;

obs_count = size(X,1);
test_size = round(obs_count/5);

test_idx = randsample(1:obs_count,test_size);
train_idx = setdiff(1:obs_count,test_idx);
X_train = X(train_idx,:);
Y_train = Y(train_idx);
X_test = X(test_idx,:);
Y_test = Y(test_idx);

mc_learners = {};
round_count = 5;
for t=1:round_count,
    mc_learner = FlexiClassLearner(X_train,Y_train,mc_opts);
    for r=1:10,
        fprintf('==================================================\n');
        fprintf('META ROUND %d...\n',r);
        for i=1:5,
            tidx = 1:size(X_train,1);
            ntidx = setdiff(1:size(X_train,1),tidx);
            L = mc_learner.extend(X_train(tidx,:),Y_train(tidx,:));
            [F H C] = mc_learner.evaluate(X_train(tidx,:));
            a_train = sum(Y_train(tidx)==C) / numel(Y_train(tidx));
            [F H C] = mc_learner.evaluate(X_test);
            a_test = sum(Y_test==C) / numel(Y_test);
            fprintf('Round: %d, train_loss: %.4f, train_acc: %.4f, test_acc: %.4f\n',...
                i,L,a_train, a_test);
        end
        mc_learner.set_codewords(X_train,Y_train);
    end
    mc_learners{t} = mc_learner;
end
[F H C] = mc_learners{1}.evaluate(X_test);
H_meta = zeros(size(H));
for t=1:round_count,
    [F H C] = mc_learners{t}.evaluate(X_test);
    H_meta = H_meta + H;
end
[max_vals max_idx] = max(H_meta,[],2);
C = reshape(mc_learners{1}.c_labels(max_idx),numel(max_idx),1);
a_test = sum(Y_test==C) / numel(Y_test);
fprintf('META TEST ACC: %.4f\n',a_test);

% for t=1:3,
%     test_idx = randsample(1:obs_count,test_size);
%     train_idx = setdiff(1:obs_count,test_idx);
%     X_train = X(train_idx,:);
%     Y_train = Y(train_idx);
%     X_test = X(test_idx,:);
%     Y_test = Y(test_idx);
%     mc_learner = MultiClassLearner(X_train,Y_train,mc_opts);
%     for r=1:20,
%         fprintf('STARTING META ROUND %d...\n',r);
%         tidx = randsample(1:numel(train_idx),round(0.75*numel(train_idx)));
%         Xt = X_train(tidx,:);
%         Yt = Y_train(tidx);
%         F = mc_learner.evaluate(Xt);
%         [L dLdFa] = mc_learner.compute_loss_grad(F, Yt);
%         for i=1:length(mc_learner.l_objs),
%             lrnr=mc_learner.l_objs{i};
%             lrnr.set_possi_bumps(Xt,Yt, abs(dLdFa(:,i)), bump_count);
%         end
%         for i=1:5,
%             L = mc_learner.extend(Xt,Yt);
%             [F H C] = mc_learner.evaluate(X_train);
%             a_train = sum(Y_train==C) / numel(Y_train);
%             [F H C] = mc_learner.evaluate(X_test);
%             a_test = sum(Y_test==C) / numel(Y_test);
%             fprintf('Round: %d, train_loss: %.4f, train_acc: %.4f, test_acc: %.4f\n',...
%                 i,L,a_train, a_test);
%         end
%         mc_learner.set_codewords(X_train,Y_train);
%     end
% end