classdef FastChainLearner < Learner
    % A class for boosting chains of stumps for real-valued observations. A
    % stump chain is a decision tree in which all nodes at a given depth are
    % split on the same feature, hence producing a homogeneous sequence (chain)
    % of comparisons for every input. Unlike a product learner, the chain
    % learner is allowed to independently weight its leaves.
    % 
    % The constructor for this class accepts a set of observations X, a set of
    % target (possibly weighted) classes Y, and an options structure.
    %
    % Accepted options:
    %   opts.nu: shrinkage/regularization term for boosting
    %   opts.loss_func: Loss function handle to a function that can be wrapped
    %                   around hypothesis outputs F as @(F)loss_func(F,Y)
    %   opts.chain_len: The length of each stump chain weak learner to be used
    %                   during extension via boosting
    %   opts.update_rounds: The number of update cycles to perform when
    %                       computing each stump chain
    %
    
    properties
        % stump_chains is a cell array, in which each cell stores a struct
        % containing a stump chain and an array of leaf weights for the
        % decision tree induced by the stump chain.
        stump_chains
        % chain_len is the number of stumps to include in each stump chain
        chain_len
        % update_rounds is the number of update cycles to perform while
        % optimizing ech stump chain
        update_rounds
        % Ft and Xt are fixed training sets, used to avoid redundant
        % evaluations
        Ft
        Xt
        Yt
    end
    
    methods
        function [ self ] = FastChainLearner(X, Y, opts)
            % Simple constructor for boosted stumps, which computes a constant
            % valued stump to initialize the set of stumps.
            if ~exist('opts','var')
                opts = struct();
            end
            if ~isfield(opts,'nu')
                self.nu = 1.0;
            else
                self.nu = opts.nu;
            end
            if ~isfield(opts,'loss_func')
                self.loss_func = @loss_bindev;
            else
                self.loss_func = opts.loss_func;
            end
            if ~isfield(opts,'chain_len')
                self.chain_len = 2;
            else
                self.chain_len = opts.chain_len;
            end
            if ~isfield(opts,'update_rounds')
                self.update_rounds = 2;
            else
                self.update_rounds = opts.update_rounds;
            end
            idx = 1:size(X,1);
            F = zeros(size(X,1),1);
            [ s ] = FastChainLearner.find_step(F, @(f)self.loss_func(f,Y,idx));
            stump_chain = struct();
            stump_chain.stumps = [0 0];
            stump_chain.leaf_weights = [s];
            self.stump_chains = {stump_chain};
            % Set up structures for training optimization
            self.Xt = X;
            self.Yt = Y;
            self.Ft = self.evaluate(self.Xt);
            return
        end
        
        function [ L ] = extend(self, X, Y, keep_it, method)
            % Dispatcher, to select extension via either least-squares gradient
            % matching or maximum difference of weighted means gradient
            % matching.
            if ~exist('keep_it','var')
                keep_it = 1;
            end
            if ~exist('method','var')
                method = 2;
            end
            if (method ~= 1 && method ~= 2)
                error('SELECT METHOD 1 OR 2 FOR EXTENSION.\n');
            end
            if (method == 1)
                %L = self.extend_lsq(X, Y, keep_it);
                % NOTE: least-squares splits not yet implemented
                L = self.extend_mdm(X, Y, keep_it);
            else
                L = self.extend_mdm(X, Y, keep_it);
            end 
            return
        end
        
        function [ L ] = extend_mdm(self, X, Y, keep_it)
            % Extend the current set of stumps, based on the observations in X
            % and the loss/grad function loss_func. Return the post-update loss
            F = self.Ft;
            X = self.Xt;
            Y = self.Yt;
            obs_count = size(X,1);
            [L dL] = self.loss_func(F, Y, 1:obs_count);
            stump_chain = struct();
            stump_chain.stumps = zeros(self.chain_len, 2);
            stump_chain.leaf_weights = zeros(2^(self.chain_len),1);
            % For each update round, cycle through the current set of stumps,
            % updating each one in turn, based on the tree induced by the
            % current state of the remaining stumps.
            stump_idx = 1:size(stump_chain.stumps,1);
            for r_num=1:self.update_rounds,
                update_order = 1:numel(stump_idx);
                for u_num=1:numel(stump_idx),
                    this_idx = update_order(u_num);
                    other_stumps = stump_chain.stumps(stump_idx ~= this_idx,:);
                    % Compute a set of indices describing the way the "other"
                    % stumps subdivide the observations
                    leaf_idx = FastChainLearner.get_leaf_idx(X, other_stumps);
                    leaf_dL = repmat(dL, 1, max(leaf_idx));
                    for l=1:max(leaf_idx),
                        leaf_dL(leaf_idx ~= l, l) = 0;
                    end
                    feat_count = size(X,2);
                    feat_info = zeros(feat_count, 2);
                    for f_num=1:feat_count,
                        [f_vals f_idx] = sort(X(:,f_num),'ascend');
                        f_grad = leaf_dL(f_idx,:);
                        f_sum = 0;
                        f_val = 0;
                        cs_l = cumsum(f_grad,1);
                        cs_r = bsxfun(@plus, -cs_l, cs_l(end,:));
                        cs_lr = sum(abs(cs_l),2) + sum(abs(cs_r),2);
                        [cs_vals cs_idx] = sort(cs_lr,'descend');
                        % For the current feature, check all possible splits, 
                        % tracking best split point and its corresponding gap
                        for s_num=1:obs_count,
                            idx = cs_idx(s_num);
                            if ((idx == obs_count) || ...
                                    (f_vals(idx) < f_vals(idx+1)))
                                f_sum = cs_vals(s_num);
                                if (idx < obs_count)                       
                                    f_val = f_vals(idx) + ...
                                          (rand()*(f_vals(idx+1)-f_vals(idx)));
                                else
                                    f_val = 1e10;
                                end
                                break
                            end
                        end
                        % Record the best "sum" and corresponding split
                        feat_info(f_num,1) = f_sum;
                        feat_info(f_num,2) = f_val;
                    end
                    % Find the feature/split-point pairing producing the best
                    % "sum"
                    [best_sum best_feat] = max(feat_info(:,1));
                    best_thresh = feat_info(best_feat,2);
                    stump_chain.stumps(this_idx,1) = best_feat;
                    stump_chain.stumps(this_idx,2) = best_thresh;
                end
            end
            % For the computed stump chain, get the indices of the leaves into
            % which the observations are subdivided
            leaf_idx = FastChainLearner.get_leaf_idx(X, stump_chain.stumps);
            for l=1:max(leaf_idx),
                idx = find(leaf_idx == l);
                if (numel(idx) > 0)
                    w = FastChainLearner.find_step(...
                        F, @( f ) self.loss_func(f, Y, idx));
                else
                    w = 0;
                end
                stump_chain.leaf_weights(l) = w * self.nu;
                self.Ft(idx) = self.Ft(idx) + (w * self.nu);
            end
            % Append the computed stump chain to the current set of computed
            % stump chains
            self.stump_chains{end+1} = stump_chain;
            L_new = self.loss_func(self.Ft, Y, 1:obs_count);
            if (L_new - L > 1e-3)
                error('oops: loss increase in stump chain extension\n');
            end
            L = L_new;
            % Undo addition of stump chain if keep_it ~= 1
            if (keep_it ~= 1)
                self.stump_chains(end) = [];
            end
            return 
        end
        
        function [ F ] = evaluate(self, X, idx)
            % Evaluate the current set of stumps associated with this learner
            if ~exist('idx','var')
                idx = 1:numel(self.stump_chains);
            end
            if (idx == -1)
                idx = numel(self.stump_chains);
            end
            F = zeros(size(X,1), 1);
            for s_num=1:numel(idx),
                % Evaluate each stump chain referenced by idx
                stump_chain = self.stump_chains{idx(s_num)};
                l_idx = FastChainLearner.get_leaf_idx(X, stump_chain.stumps);
                for l_num=1:max(l_idx),
                    F(l_idx == l_num) = F(l_idx == l_num) + ...
                        stump_chain.leaf_weights(l_num);
                end
            end
            return
        end
    end
    
    methods (Static = true)

        function [ step ] = find_step(F, step_func)
            % Use Matlab unconstrained optimization to find a step length that
            % minimizes: loss_func(F + Fs.*step)
            step_opts = optimset('MaxFunEvals',30,'TolX',1e-3,'TolFun',...
                1e-3,'Display','off');
            [L dL] = step_func(F);
            if (sum(dL) > 0)
                step = fminbnd(@( s ) step_func(F + s), -1, 0, step_opts);
            else
                step = fminbnd(@( s ) step_func(F + s), 0, 1, step_opts);
            end
            return
        end
        
        function [ leaf_idx ] = get_leaf_idx(X, stumps)
            % Get the leaf index for each observation in X according to the
            % stump chain in stump_chain
            %
            % Parameters:
            %   X: observations to pass through stump chain
            %   stumps: (stump_count x 2) array describing a stump chain, with
            %           each row being a stump, i.e. (split_feat, split_val)
            % Outputs:
            %   leaf_idx: the leaf index for each observation in X
            %
            obs_count = size(X,1);
            % Filter out stumps splitting on the 0'th feature (they're "empty")
            stumps = stumps(stumps(:,1) > 0,:);
            stump_count = size(stumps, 1);
            leaf_idx = zeros(obs_count,1);
            for s=1:stump_count,
                % Get indices of all observations that go left on this split,
                % with the implication being r_idx = ~l_idx.
                l_idx = X(:,stumps(s,1)) <= stumps(s,2);
                leaf_idx(l_idx) = leaf_idx(l_idx) .* 2;
                leaf_idx(~l_idx) = (leaf_idx(~l_idx) .* 2) + 1;
            end
            leaf_idx = leaf_idx + 1;
            return
        end
            
    end
    
end

