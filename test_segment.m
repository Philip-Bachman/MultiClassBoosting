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

bump_count = 250;
mc_opts = struct();
mc_opts.nu = 0.33;
mc_opts.loss_func = @loss_bindev;
mc_opts.l_count = 6;
mc_opts.l_const = @VorLearner;
mc_opts.l_opts.vor_count = 2;
mc_opts.l_opts.vor_samples = 500;

obs_count = size(X,1);
test_size = round(obs_count/5);

for t=1:10,
    test_idx = randsample(1:obs_count,test_size);
    train_idx = setdiff(1:obs_count,test_idx);
    X_train = X(train_idx,:);
    Y_train = Y(train_idx);
    X_test = X(test_idx,:);
    Y_test = Y(test_idx);
    mc_learner = MultiClassLearner(X_train,Y_train,mc_opts);
    for r=1:10,
        fprintf('==================================================\n');
        fprintf('META ROUND %d...\n',r);
        for i=1:5,
            tidx = 1:size(X_train,1); %randsample(size(X_train,1),round(0.66*size(X_train,1)));
            ntidx = setdiff(1:size(X_train,1),tidx);
            L = mc_learner.extend(X_train(tidx,:),Y_train(tidx,:));
            [F Cf] = mc_learner.evaluate(X_train(tidx,:));
            a_train = sum(Y_train(tidx)==Cf) / numel(Y_train(tidx));
            [F Cf] = mc_learner.evaluate(X_test);
            a_test = sum(Y_test==Cf) / numel(Y_test);
            fprintf('Round: %d, train_loss: %.4f, train_acc: %.4f, test_acc: %.4f\n',...
                i,L,a_train, a_test);
        end
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
%             [F Cf] = mc_learner.evaluate(X_train);
%             a_train = sum(Y_train==Cf) / numel(Y_train);
%             [F Cf] = mc_learner.evaluate(X_test);
%             a_test = sum(Y_test==Cf) / numel(Y_test);
%             fprintf('Round: %d, train_loss: %.4f, train_acc: %.4f, test_acc: %.4f\n',...
%                 i,L,a_train, a_test);
%         end
%         mc_learner.set_codewords(X_train,Y_train);
%     end
% end