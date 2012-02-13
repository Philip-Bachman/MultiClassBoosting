classdef StumpLearner < Learner
    % A simple class for boosting stumps for real-valued observations. The
    % constructor for this class accepts a set of observations X, a set of
    % target (possibly weighted) classes Y, and an options structure.
    %
    % Accepted options:
    %   opts.nu: shrinkage/regularization term for boosting
    %   opts.loss_func: Loss function handle to a function that can be wrapped
    %                   around hypothesis outputs F as @(F)loss_func(F,Y).
    %   opts.do_opt: This indicates whether to use fast training optimization.
    %                This should only be set to 1 if all training rounds will
    %                use the same training set of observations/classes.
    %
    
    properties
        % Stumps stores the stumps that make up this learner. Each row describes
        % a stump, with row(1) being the feature on which to split, row(2) being
        % the value at which to split, row(3) being the value left of the split,
        % and row(4) being the value right of the split.
        stumps
        % Xt is an optional fixed training set, used if opt_train==1
        Xt
        % Ft is the current output of this learner for each row in Xt
        Ft
        % opt_train indicates if to use fast training optimization
        opt_train
        % step_opts stores options for self.find_step()
        step_opts
    end
    
    methods
        function [ self ] = StumpLearner(X, Y, opts)
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
            self.step_opts = optimset('MaxFunEvals',30,'TolX',1e-3,'TolFun',...
                1e-3,'Display','off');
            idx = 1:size(X,1);
            F = zeros(size(X,1),1);
            [ s ] = self.find_step(F, @(f)self.loss_func(f,Y,idx));
            self.stumps = [1 0 s s];
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
                L = self.extend_lsq(X, Y, keep_it);
            else
                L = self.extend_mdm(X, Y, keep_it);
            end 
            return
        end
        
        function [ L ] = extend_lsq(self, X, Y, keep_it)
            % Extend the current set of stumps, based on the observations in X
            % and the loss/grad function loss_func. Return the post-update loss
            if (self.opt_train ~= 1)
                F = self.evaluate(X);
            else
                F = self.Ft;
                X = self.Xt;
            end
            obs_count = size(X,1);
            [L dL] = self.loss_func(F, Y, 1:obs_count);
            % Compute a split point based on the computed gradient
            feat_count = size(X,2);
            best_feat = 0;
            best_thresh = 0;
            best_err = sum(dL.^2);
            r_sz = [(obs_count-1):-1:1 1];
            % Compute the best split point for each feature, tracking best
            % feat/split pair
            for f_num=1:feat_count,
               [f_vals f_idx] = sort(X(:,f_num),'ascend');
               f_grad = dL(f_idx);
               f_err = sum(f_grad.^2);
               f_val = 0;
               csums = cumsum(f_grad);
               cssq = cumsum(f_grad.^2);
               % For the current feature, check all possible split points, 
               % tracking best split point and its corresponding error
               for s_num=1:obs_count,
                   % Compute the error for points left of the split
                   l_err = cssq(s_num) - (csums(s_num)^2 / s_num);
                   % Compute the error for points right of the split
                   r_err = (cssq(end) - cssq(s_num)) - ...
                       ((csums(end)-csums(s_num))^2 / r_sz(s_num));
                   % Compute the joint left/right error and check if it is the
                   % best so far
                   if ((l_err + r_err) < f_err)
                       if (s_num==obs_count || f_vals(s_num)<f_vals(s_num+1))
                           f_err = l_err + r_err;
                           % Compute a partially randomized split point
                           if (s_num < obs_count)
                               f_val = f_vals(s_num) + ...
                                   (rand()*(f_vals(s_num+1)-f_vals(s_num)));
                           else
                               f_val = f_vals(obs_count) + 1;
                           end
                       end
                   end
               end
               % Check if the best split point found for this feature is better
               % than any split point found for previously examined features
               if (f_err < best_err)
                   best_err = f_err;
                   best_feat = f_num;
                   best_thresh = f_val;
               end
            end
            % For the best split point, compute left and right weights
            idx = find(X(:,best_feat) <= best_thresh);
            w_l = self.find_step(F, @( f ) self.loss_func(f, Y, idx));
            idx = find(X(:,best_feat) > best_thresh);
            w_r = self.find_step(F, @( f ) self.loss_func(f, Y, idx));
            % Append the best split found as a new (reweighted) stump
            stump = [best_feat best_thresh (w_l*self.nu) (w_r*self.nu)];
            self.stumps = [self.stumps; stump];
            if (self.opt_train == 1)
                % Use fast training optimization via incremental evaluation
                Ft_new = self.evaluate(self.Xt, size(self.stumps,1));
                self.Ft = self.Ft + Ft_new;
                F = self.Ft;
            else
                F = self.evaluate(X);
            end
            L = self.loss_func(F, Y, 1:obs_count);
            % Undo addition of stump if keep_it ~= 1
            if (keep_it ~= 1)
                self.stumps = self.stumps(1:end-1,:);
                if (self.opt_train == 1)
                    self.Ft = self.Ft - Ft_new;
                end
            end
            return 
        end
        
        function [ L ] = extend_mdm(self, X, Y, keep_it)
            % Extend the current set of stumps, based on the observations in X
            % and the loss/grad function loss_func. Return the post-update loss
            if (self.opt_train ~= 1)
                F = self.evaluate(X);
            else
                F = self.Ft;
                X = self.Xt;
            end
            obs_count = size(X,1);
            [L dL] = self.loss_func(F, Y, 1:obs_count);
            % Compute a split point based on the computed gradient
            feat_count = size(X,2);
            best_feat = 0;
            best_thresh = 0;
            best_sum = 0;
            % Compute the best split point for each feature, tracking best
            % feat/split pair
            for f_num=1:feat_count,
               [f_vals f_idx] = sort(X(:,f_num),'ascend');
               f_grad = dL(f_idx);
               f_sum = 0;
               f_val = 0;
               cs_l = cumsum(f_grad);
               cs_r = -cs_l + cs_l(end);
               % For the current feature, check all possible split points, 
               % tracking best split point and its corresponding gap
               for s_num=1:obs_count,
                   % Compute the left->right sums and check if it is best yet
                   if ((abs(cs_l(s_num)) + abs(cs_r(s_num))) > f_sum)
                       if (s_num==obs_count || f_vals(s_num)<f_vals(s_num+1))
                           f_sum = abs(cs_l(s_num)) + abs(cs_r(s_num));
                           % Compute a partially randomized split point
                           if (s_num < obs_count)
                               f_val = f_vals(s_num) + ...
                                   (rand()*(f_vals(s_num+1)-f_vals(s_num)));
                           else
                               f_val = f_vals(obs_count) + 1;
                           end
                       end
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
            if (best_feat == 0)
                best_feat = 1;
                best_thresh = 1e10;
            end
            % For the best split point, compute left and right weights
            idx = find(X(:,best_feat) <= best_thresh);
            w_l = self.find_step(F, @( f ) self.loss_func(f, Y, idx));
            idx = find(X(:,best_feat) > best_thresh);
            w_r = self.find_step(F, @( f ) self.loss_func(f, Y, idx));
            % Append the best split found as a new (reweighted) stump
            stump = [best_feat best_thresh (w_l*self.nu) (w_r*self.nu)];
            self.stumps = [self.stumps; stump];
            if (self.opt_train == 1)
                % Use fast training optimization via incremental evaluation
                Ft_new = self.evaluate(self.Xt, size(self.stumps,1));
                self.Ft = self.Ft + Ft_new;
                F = self.Ft;
            else
                F = self.evaluate(X);
            end
            L_new = self.loss_func(F, Y, 1:obs_count);
            if (L_new > L)
                error('oops: loss increase in stump extension\n');
            end
            L = L_new;
            % Undo addition of stump if keep_it ~= 1
            if (keep_it ~= 1)
                self.stumps = self.stumps(1:end-1,:);
                if (self.opt_train == 1)
                    self.Ft = self.Ft - Ft_new;
                end
            end
            return 
        end
        
        function [ F ] = evaluate(self, X, idx)
            % Evaluate the current set of stumps associated with this learner
            if ~exist('idx','var')
                idx = 1:size(self.stumps,1);
            end
            if (idx == -1)
                idx = size(self.stumps,1);
            end
            obs_count = size(X,1);
            stump_count = numel(idx);
            F = zeros(obs_count, 1);
            for s_num=1:stump_count,
                s_idx = idx(s_num);
                s_feat = self.stumps(s_idx,1);
                s_thresh = self.stumps(s_idx,2);
                l_idx = X(:,s_feat) <= s_thresh;
                F(l_idx) = F(l_idx) + self.stumps(s_idx,3);
                F(~l_idx) = F(~l_idx) + self.stumps(s_idx,4);
            end
            return
        end       

        function [ step ] = find_step(self, F, step_func)
            % Use Matlab unconstrained optimization to find a step length that
            % minimizes: loss_func(F + Fs.*step)
            [L dL] = step_func(F);
            if (sum(dL) > 0)
                step = fminbnd(@( s ) step_func(F + s), -1, 0, self.step_opts);
            else
                step = fminbnd(@( s ) step_func(F + s), 0, 1, self.step_opts);
            end
%             function [obj grad] = step_loss( s )
%                 [obj grad] = loss_func(F + Fs.*s);
%                 grad = sum(grad.*Fs);
%             end
%             options = optimset('MaxFunEvals',30,'TolFun',1e-3,'GradObj','on',...
%                 'Display','off');
%             step = fminunc(@step_loss, 0.0, options);
            return
        end
    end
    
end

