% Run tests with UCI satimage (a.k.a. landsat) data

clear;

load('datasets/satimage.mat');
stream = RandStream.getDefaultStream();
c = clock();
reset(stream,round(1000*c(6)));

X_train = ZMUV(X_train);
X_test = ZMUV(X_test);
    
mc_opts = struct();
mc_opts.nu = 0.33;
mc_opts.loss_func = @loss_huberhinge;
mc_opts.l_count = round(2.0 * numel(unique(Y_train)));
mc_opts.l_const = @StumpLearner;
mc_opts.lam_l1 = 5e-2;
mc_opts.lam_nuc = 1e1;
mc_opts.lam_dif = 0e-5;

test_count = 30;
test_accs = zeros(test_count,1);
for t_num=1:test_count,
    mc_learner = NukeClassLearner(X_train,Y_train,mc_opts);
    for r=1:15,
        fprintf('==================================================\n');
        fprintf('META ROUND %d...\n',r);
        for i=1:5,
            tidx = randsample(1:size(X_train,1), round(size(X_train,1) * 0.66));
            Xt = X_train(tidx,:);
            Xt = Xt + bsxfun(@times, randn(size(Xt)), 0.1 * std(Xt));
            L = mc_learner.extend(Xt,Y_train(tidx));
            [F H C] = mc_learner.evaluate(X_train);
            a_train = sum(Y_train==C) / numel(Y_train);
            [F H C] = mc_learner.evaluate(X_test);
            a_test = sum(Y_test==C) / numel(Y_test);
            test_accs(t_num) = a_test;
            fprintf('Round: %d, train_loss: %.4f, train_acc: %.4f, test_acc: %.4f\n',...
                i,L,a_train, a_test);
        end
        mc_learner.set_codewords(X_train,Y_train, 5);
    end
    save('results_satimage.mat');
end

