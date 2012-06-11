% test the "theorem" regarding degradation of boosting performance when the
% target concept has significant "symmetry" relative to the weak learner's
% hypothesis space

% The basic hypothesis we work with is I[sin(s*x) > 0], which permits a simple
% analytical solution. In generating training/test sets, we add a bit of noise,
% to make the +/- transistions of sin(s*x) "fuzzy". The parameter s indicates a
% scaling of x, with higher values indicating more oscillations between +/- for
% a given range of x values.
class_func_clean = @(x, s) sign(sin(s*x));
class_func_noisy = @(x, s) sign(sin(s*x) + (0.0*randn(size(x))));

s_vals = [4 8 16 32 64];
s_count = numel(s_vals);

test_rounds = 50;
boost_rounds = 500;
train_size = 10000;
test_size = 10000;
test_boost_loss = ones(s_count, test_rounds, boost_rounds+1) .* 0.7;
test_boost_err = ones(s_count, test_rounds, boost_rounds+1) .* 0.5;
test_chain_loss = ones(s_count, test_rounds, boost_rounds+1) .* 0.7;
test_chain_err = ones(s_count, test_rounds, boost_rounds+1) .* 0.5;
test_true_err = zeros(s_count, test_rounds);

% Setup options for the basic learner to be used during testing
l_opts = struct();
l_opts.nu = 1.0;
l_opts.loss_func = @loss_bindev;
l_opts.update_rounds = 5;

for t_num=1:test_rounds,
    fprintf('TEST ROUND %d:\n',t_num);
    for s_num=1:s_count,
        s_val = s_vals(s_num);
        fprintf('  Testing s_val=%.4f:\n',s_val);
        % Generate training and test sets
        X_train = randn(train_size,1);
        X_test = randn(test_size,1);
        Y_train = class_func_noisy(X_train(:,1),s_val);
        Y_test = class_func_noisy(X_test(:,1),s_val);
        % Check error on test set using true classifier
        Y_test_true = class_func_clean(X_test(:,1),s_val);
        test_true_err(s_num,t_num) = sum(Y_test ~= Y_test_true) / numel(Y_test);
        fprintf('    oracle error: %.4f\n',test_true_err(s_num,t_num));
        % Create a learner to use for boosting
        fprintf('    Boosting:\n');
        lrnr = FastStumpLearner(X_train, Y_train, l_opts);
        lrnr_c = FastChainLearner(X_train, Y_train, l_opts);
        % Initialize online computation of test hypothesis outputs
        Ft_stumps = lrnr.evaluate(X_test);
        Ft_chains = lrnr_c.evaluate(X_test);
        for b_num=1:boost_rounds,
            L = lrnr.extend(X_train, Y_train);
            Ft_stumps = Ft_stumps + lrnr.evaluate(X_test,-1);
            E = sum(Y_test ~= sign(Ft_stumps)) / numel(Y_test);
            test_boost_loss(s_num,t_num,b_num+1) = L;
            test_boost_err(s_num,t_num,b_num+1) = E;
            if (mod(b_num, round(boost_rounds/50)) == 0)
                fprintf('      round %d: loss=%.4f, err=%.4f\n',b_num,L,E);
            end
            L = lrnr_c.extend(X_train, Y_train);
            Ft_chains = Ft_chains + lrnr_c.evaluate(X_test,-1);
            E = sum(Y_test ~= sign(Ft_chains)) / numel(Y_test);
            test_chain_loss(s_num,t_num,b_num+1) = L;
            test_chain_err(s_num,t_num,b_num+1) = E;
        end
    end
    if (mod(t_num,5) == 0)
        mean_boost_loss = squeeze(mean(test_boost_loss(:,1:t_num,:),2));
        mean_boost_err = squeeze(mean(test_boost_err(:,1:t_num,:),2));
        mean_chain_err = squeeze(mean(test_chain_err(:,1:t_num,:),2));
        mean_joint_err = [mean_boost_err(3:5,:); mean_chain_err(3:5,:)];
        % Draw loss convergence figure
        loss_fig = plot_train_loss(mean_boost_loss');
        hgsave(loss_fig, 'converge_thm_loss.fig');
        % Draw error convergence figure
        err_fig = plot_test_error(mean_boost_err');
        hgsave(err_fig, 'converge_thm_error.fig');
        % Draw comparison of stumps vs. stump chains
        joint_fig = plot_joint_error(mean_joint_err');
        hgsave(joint_fig, 'converge_thm_joint.fig');
        drawnow();
        pause(10);
        close(loss_fig);
        close(err_fig);
        close(joint_fig);
    end
    save('test_converge_thm_results.mat');
end
mean_boost_loss = squeeze(mean(test_boost_loss,2));
mean_boost_err = squeeze(mean(test_boost_err,2));
mean_chain_err = squeeze(mean(test_chain_err,2));
mean_joint_err = [mean_boost_err(3:5,:); mean_chain_err(3:5,:)];
% Draw loss convergence figure
loss_fig = plot_train_loss(mean_boost_loss'); 
hgsave(loss_fig, 'converge_thm_loss.fig');
% Draw error convergence figure
err_fig = plot_test_error(mean_boost_err');
hgsave(err_fig, 'converge_thm_error.fig');
% Draw comparison of stumps vs. stump chains
joint_fig = plot_joint_error(mean_joint_err');
hgsave(joint_fig, 'converge_thm_joint.fig');

save('test_converge_thm_results.mat');
