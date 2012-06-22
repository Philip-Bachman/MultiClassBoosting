% Run tests with UCI satimage (a.k.a. landsat) data

clear;

load('datasets/satimage.mat');
stream = RandStream.getDefaultStream();
c = clock();
reset(stream,round(1000*c(6)));

train_size = size(X_train,1);
X_all = ZMUV([X_train; X_test]);
X_train = X_all(1:train_size,:);
X_test = X_all(train_size+1:end,:);
    
mc_opts = struct();
mc_opts.nu = 1.0;
mc_opts.loss_func = @loss_hingesq;
mc_opts.l_count = 3;
mc_opts.l_const = @RBFLearner;
mc_opts.lam_l1 = 0e-2;
mc_opts.lam_nuc = 0e-1;
mc_opts.lam_dif = 1e-2;

test_count = 1;
test_accs = zeros(test_count,1);
for t_num=1:test_count,
    mc_learner = NukeClassLearner(X_train,Y_train,mc_opts);
    for r=1:10,
        fprintf('==================================================\n');
        fprintf('META ROUND %d...\n',r);
        for i=1:2,
            % Compute the margins for all training examples and sort training
            % examples by margin from least->greatest
            M = mc_learner.margins(X_train,Y_train);
            [Ms Ms_idx] = sort(M,'ascend');
            plot(Ms(1:1000));
            drawnow;
            for j=1:numel(mc_learner.l_objs),
                lrnr = mc_learner.l_objs{j};
                centers = [];
                gammas = [];
                for g=5:5:30,
                    idx = randsample(Ms_idx(1:1000),150);
                    gamma = g * (1 / size(X_train,2));
                    centers = [centers; X_train(idx,:)];
                    gammas = [gammas; gamma * ones(150,1)];
                end
                %gammas = gammas .* sign(randn(size(gammas)));
                lrnr.set_rbf_centers(centers,gammas);
                lrnr.rbf_type = 2;
                lrnr.nz_count = 50;
                lrnr.lam_l2 = 1e-2;
            end
            L = mc_learner.extend(X_train,Y_train);
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

% %%%%%%%%%%%%%%%%%%%
% % SPARE SCRIPTAGE %
% %%%%%%%%%%%%%%%%%%%
% gammas = (1 / size(X_train,2)) * 0.2:0.2:4.0;
% costs = logspace(-1,4.0,30);
% gamma_cost_accs = zeros(numel(gammas)*numel(costs),4);
% idx = 1;
% for i=1:numel(gammas),
%     fprintf('============================================================\n');
%     fprintf('TESTING GAMMA: %.4f\n',gammas(i));
%     fprintf('============================================================\n');
%     for j=1:numel(costs),
%         svm_str = sprintf('-s 0 -t 2 -g %.4f -c %.4f -e 0.01 -q',...
%             gammas(i),costs(j));
%         model = svmtrain(Y_train,X_train,svm_str);
%         [label train_acc preds] = svmpredict(Y_train, X_train, model,'-q');
%         [label test_acc preds] = svmpredict(Y_test, X_test, model,'-q');
%         fprintf('  cost = %.4f, tr_acc = %.4f, te_acc: %.4f, sv_count = %d\n',...
%             costs(j), train_acc(1), test_acc(1), model.totalSV);
%         gamma_cost_accs(idx,:) = [gammas(i) costs(j) train_acc(1) test_acc(1)];
%         idx = idx + 1;
%     end 
% end
