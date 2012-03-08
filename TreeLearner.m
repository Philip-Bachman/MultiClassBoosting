classdef TreeLearner < Learner
    % A simple class for boosting short trees for real-valued observations.
    % Trees are currently learned greedily, and with a full set of leaves. The
    % constructor expects a set of initial observations X, (possibly weighted)
    % classes Y, and an options structure.
    %
    % Accepted options:
    %   opts.nu: shrinkage/regularization term for boosting
    %   opts.loss_func: Loss function handle to a function that can be wrapped
    %                   around hypothesis outputs F as @(F)loss_func(F,Y).
    %   opts.do_opt: This indicates whether to use fast training optimization.
    %                This should only be set to 1 if all training rounds will
    %                use the same training set of observations/classes.
    %   opts.max_depth: The depth to which each tree will be extended. A max
    %                   depth of 1 corresponds to boosting stumps. All leaves at
    %                   each depth will be split. (i.e. trees are full)
    %
    
    properties
        % trees is a cell array of the trees making up this learner
        trees
        % max_depth gives the depth to which each tree will be grown
        max_depth
        % Xt is an optional fixed training set, used if opt_train==1
        Xt
        % Ft is the current output of this learner for each row in Xt
        Ft
        % opt_train indicates if to use fast training optimization
        opt_train
    end
    
    methods
        function [ self ] = TreeLearner(X, Y, opts)
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
            if ~isfield(opts,'do_opt')
                self.opt_train = 0;
                self.Xt = [];
                self.Ft = [];
            else
                self.opt_train = opts.do_opt;
                self.Xt = X;
                self.Ft = zeros(size(X,1),1);
            end
            if ~isfield(opts,'max_depth')
                self.max_depth = 2;
            else
                self.max_depth = opts.max_depth;
            end
            % Init with a constant tree
            depth = self.max_depth;
            nude = self.nu;
            self.nu = 1.0;
            self.max_depth = 0;
            self.extend(X, Y, 1);
            self.nu = nude;
            self.max_depth = depth;
            return
        end
        
        function [ L ] = extend(self, X, Y, keep_it)
            % Extend the current set of trees, based on the observations in X
            % and the loss/grad function loss_func. Return the post-update loss
            if ~exist('keep_it','var')
                keep_it = 1;
            end
            % First, evaluate learner and compute loss/gradient
            if (self.opt_train ~= 1)
                F = self.evaluate(X);
            else
                F = self.Ft;
                X = self.Xt;
            end
            obs_count = size(X,1);
            [L dL] = self.loss_func(F, Y, 1:obs_count);
            % Iteratively split all leaves at each current tree depth, creating
            % a full binary tree of depth self.max_depth, where a stump is
            % considered as having depth 1.
            root = TreeNode();
            root.sample_idx = 1:size(X,1);
            new_leaves = {root};
            for d=1:self.max_depth,
                old_leaves = {new_leaves{1:end}};
                new_leaves = {};
                leaf_count = length(old_leaves);
                % Split each leaf spawned by the previous round of splits
                for l_num=1:leaf_count,
                    leaf = old_leaves{l_num};
                    leaf.has_children = true;
                    leaf_idx = leaf.sample_idx;
                    if (numel(leaf_idx) > 0)
                        % Greedily split this leaf
                        leaf_X = X(leaf_idx,:);
                        leaf_dL = dL(leaf_idx);
                        [split_f split_t] = TreeLearner.find_split(leaf_X, leaf_dL);

                    else
                        % This leaf contains none of the training samples
                        split_f = 1;
                        split_t = 0;
                    end
                    % Set split info in the split leaf
                    leaf.split_feat = split_f;
                    leaf.split_val = split_t;
                    % Create right/left children, and set their split indices
                    leaf.left_child = TreeNode();
                    leaf.right_child = TreeNode();
                    l_idx = leaf_idx(X(leaf_idx,split_f) <= split_t);
                    r_idx = leaf_idx(X(leaf_idx,split_f) > split_t);
                    leaf.left_child.sample_idx = l_idx;
                    leaf.right_child.sample_idx = r_idx;
                    % Add the newly generated leaves/children to the leaf list
                    new_leaves{end+1} = leaf.left_child;
                    new_leaves{end+1} = leaf.right_child;
                end
            end
            % Set weight in each leaf of the generated tree
            for l_num=1:length(new_leaves),
                leaf = new_leaves{l_num};
                if (numel(leaf.sample_idx) > 0)
                    step_func = @( f ) self.loss_func(f, Y, leaf.sample_idx);
                    weight = TreeLearner.find_step(F, step_func);
                    leaf.weight = weight * self.nu;
                else
                    leaf.weight = 0;
                end
            end
            % Append the generated tree to the set of trees from which this
            % learner is composed.
            self.trees{end+1} = root;
            if (self.opt_train == 1)
                % Use fast training optimization via incremental evaluation
                Ft_new = self.evaluate(self.Xt, length(self.trees));
                self.Ft = self.Ft + Ft_new;
                F = self.Ft;
            else
                F = self.evaluate(X);
            end
            L = self.loss_func(F, Y, 1:obs_count);
            % Undo addition of stump if keep_it ~= 1
            if (keep_it ~= 1)
                self.trees = {self.trees{1:end-1}};
                if (self.opt_train == 1)
                    self.Ft = self.Ft - Ft_new;
                end
            end
            return 
        end
        
        function [ F ] = evaluate(self, X, idx)
            % Evaluate the prediction made for each observation in X by the
            % trees from which this learner is composed.
            %
            % Parameters:
            %   X: input observations
            %   tree_idx: the list of trees to evaluate (optional)
            %
            % Output:
            %   F: the predictions for X given self.trees and idx
            %
            if ~exist('idx','var')
                idx = 1:length(self.trees);
            end
            if (idx == -1)
                idx = length(self.trees);
            end
            F = zeros(size(X,1),1);
            for t=1:length(idx),
                tree = self.trees{idx(t)};
                weights = tree.get_weight(X);
                F = F + weights;
            end
            return
        end
    end
    methods (Static = true)
        function [ step ] = find_step(F, step_func)
            % Use Matlab unconstrained optimization to find a step length that
            % minimizes: loss_func(F + Fs.*step)
            options = optimset('MaxFunEvals',30,'TolX',1e-3,'TolFun',1e-3,...
                'Display','off');
            [L dL] = step_func(F);
            if (sum(dL) > 0)
                step = fminbnd(@( s ) step_func(F + s), -1, 0, options);
            else
                step = fminbnd(@( s ) step_func(F + s), 0, 1, options);
            end
            return
        end
        
        function [best_feat best_thresh] = find_split(X, dL)
            % Compute a split of the given set of values that maximizes the
            % weighted difference of means for dL
            %
            % Parameters:
            %   X: set of observations to split
            %   dL: loss gradient with respect to each observation
            % Output:
            %   best_feat: feature on which split occurred
            %   best_thresh: threshold for split
            %
            obs_dim = size(X,2);
            obs_count = size(X,1);
            best_feat = 0;
            best_thresh = 0;
            best_sum = 0;
            % Compute the best split point for each feature, tracking best
            % feat/split pair
            for f_num=1:size(X,2),
               [f_vals f_idx] = sort(X(:,f_num),'ascend');
               f_grad = dL(f_idx);
               f_sum = 0;
               f_val = 0;
               cs_l = cumsum(f_grad);
               cs_r = -cs_l + cs_l(end);
               cs_lr = abs(cs_l) + abs(cs_r);
               [cs_vals cs_idx] = sort(cs_lr,'descend');
               % For the current feature, check all possible split points, 
               % tracking best split point and its corresponding gap
               for s_num=1:obs_count,
                   idx = cs_idx(s_num);
                   if ((idx == obs_count) || (f_vals(idx) < f_vals(idx+1)))
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
               % Check if the best split point found for this feature is better
               % than any split point found for previously examined features
               if (f_sum > best_sum)
                   best_sum = f_sum;
                   best_feat = f_num;
                   best_thresh = f_val;
               end
            end
            % What to do if no good split was found
            if (best_feat == 0)
                best_feat = 1;
                best_thresh = 1e10;
            end
            return
        end
    end
    
end

