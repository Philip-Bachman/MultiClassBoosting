classdef VorLearner < Learner
    % A simple class for boosting depth limited trees for real-valued
    % observations. Trees are currently learned greedily, and with a full set
    % of leaves. The constructor expects a set of initial observations X,
    % (possibly weighted) classes Y, and an options structure.
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
    %   opts.vor_count: Number of points with which to induce each voronoi
    %                   partition of the input space
    %   opts.vor_samples: Number of sets of vor_count points to sample before
    %                     selecting the best seen voronoi partition thus far
    %
    
    properties
        % trees is a cell array of the trees making up this learner
        trees
        % max_depth gives the depth to which each tree will be grown
        max_depth
        % vor_count is explained above
        vor_count
        % vor_samples is explained above
        vor_samples
        % Xt is an optional fixed training set, used if opt_train==1
        Xt
        % Ft is the current output of this learner for each row in Xt
        Ft
        % opt_train indicates if to use fast training optimization
        opt_train
        % step_opts contains options for self.find_step
        step_opts
    end
    
    methods
        function [ self ] = VorLearner(X, Y, opts)
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
                self.max_depth = 1;
            else
                self.max_depth = opts.max_depth;
            end
            if ~isfield(opts,'vor_count')
                self.vor_count = 2;
            else
                self.vor_count = opts.vor_count;
            end
            if ~isfield(opts,'vor_samples')
                self.vor_samples = 100;
            else
                self.vor_samples = opts.vor_samples;
            end 
            % Init with a constant tree
            F = zeros(size(X,1),1);
            w = self.find_step(F, @( f ) self.loss_func(f, Y, 1:size(Y,1)));
            self.trees = {VorNode(w)};
            self.step_opts = optimset('MaxFunEvals',40,'TolX',1e-3,'TolFun',...
                1e-3, 'Display','off');
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
            [L dL] = self.loss_func(F, Y, 1:size(F,1));
            % Iteratively split all leaves at each current tree depth, creating
            % a full voronoi tree of depth self.max_depth. Each voronoi node
            % splits on self.vor_count points. Splitting points are selected as
            % the best among self.vor_samples randomly sampled sets.
            root = VorNode();
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
                    % Greedily split this leaf
                    leaf_X = X(leaf_idx,:);
                    leaf_dL = dL(leaf_idx);
                    [d_vals min_idx] = self.set_split(leaf_X, leaf_dL, leaf);
                    % Create children for each vor point, and set their indices
                    for c_num=1:size(leaf.vor_points,1),
                        leaf.children{c_num} = VorNode();
                        leaf.children{c_num}.sample_idx = ...
                            leaf_idx(min_idx == c_num);
                        new_leaves{end+1} = leaf.children{c_num};
                    end
                end
            end
            % Set weight in each leaf of the generated tree
            for l_num=1:length(new_leaves),
                leaf = new_leaves{l_num};
                step_func = @( f ) self.loss_func(f, Y, leaf.sample_idx);
                weight = self.find_step(F, step_func);
                leaf.weight = weight * self.nu;
            end
            % Append the generated tree to the set of trees from which this
            % learner is composed.
            self.trees{end+1} = root;
            if (self.opt_train == 1)
                % Use fast training optimization via incremental evaluation
                Ft_new = self.evaluate(self.Xt, -1);
                self.Ft = self.Ft + Ft_new;
                F = self.Ft;
            else
                F = self.evaluate(X);
            end
            L = self.loss_func(F, Y, 1:size(F,1));
            % Undo addition of stump if keep_it ~= 1
            if (keep_it ~= 1)
                self.trees = {self.trees{1:end-1}};
                if (self.opt_train == 1)
                    self.Ft = self.Ft - Ft_new;
                end
            end
            return 
        end
        
        function [best_dv best_midx] = set_split(self, X, dL, vnode)
            % Compute a split of the given set of values that maximizes the
            % weighted difference of means for dL
            %
            % Parameters:
            %   X: set of observations to split
            %   dL: loss gradient with respect to each observation
            %   vnode: the VorNode whose distance function to use
            % Output:
            %   best_dv: distances from points in X to the nearest of the best
            %            found voronoi points
            %   best_midx: index of the nearest best found voronoi point for
            %              each point in X
            %
            best_vps = [];
            best_sum = 0;
            best_dv = [];
            best_midx = [];
            s_idx = zeros(1,self.vor_count);
            for s_num=1:self.vor_samples,
                % select self.vor_count random points from X
                %w = abs(dL);
                %for i=1:self.vor_count,
                %    s_idx(i) = randsample(size(X,1),1,true,w);
                %    w(s_idx(i)) = 0;
                %end
                s_idx = randsample(size(X,1),self.vor_count);
                % set these as the vor_points in vnode
                vnode.vor_points = X(s_idx,:);
                % compute voronoi partition of X using these points
                [dv midx] = vnode.compute_dists(X);
                % compute absolute sums of gradients in each voronoi region
                s_sum = 0;
                for vnum=1:self.vor_count,
                    s_sum = s_sum + abs(sum(dL(midx == vnum)));
                end
                if (s_sum > best_sum)
                    % if the voronoi partition induced by these points was the
                    % best match of the gradient thus far, store their info
                    best_sum = s_sum;
                    best_vps = vnode.vor_points;
                    best_dv = dv;
                    best_midx = midx;
                end
            end
            vnode.vor_points = best_vps;
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

        function [ step ] = find_step(self, F, step_func)
            % Use Matlab unconstrained optimization to find a step length that
            % minimizes: step_func(F + Fs.*step)
            [L dL] = step_func(F);
            if (sum(dL) > 0)
                step = fminbnd(@( s ) step_func(F + s), -2, 0, self.step_opts);
            else
                step = fminbnd(@( s ) step_func(F + s), 0, 2, self.step_opts);
            end
            return
        end
    end
    
end

