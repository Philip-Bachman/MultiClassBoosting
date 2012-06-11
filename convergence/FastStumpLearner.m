classdef FastStumpLearner < Learner
    % A simple class for boosting stumps for real-valued observations. The
    % constructor for this class accepts a set of observations X, a set of
    % target (possibly weighted) classes Y, and an options structure.
    %
    % Accepted options:
    %   opts.nu: shrinkage/regularization term for boosting
    %   opts.loss_func: Loss function handle to a function that can be wrapped
    %                   around hypothesis outputs F as @(F)loss_func(F,Y).
    %
    
    properties
        % Stumps stores the stumps that make up this learner. Each row describes
        % a stump, with row(1) being the feature on which to split, row(2) being
        % the value at which to split, row(3) being the value left of the split,
        % and row(4) being the value right of the split.
        stumps
        % Ft and Xt are fixed training sets, used to avoid redundant
        % evaluations
        Ft
        Xt
        Yt
    end
    
    methods
        function [ self ] = FastStumpLearner(X, Y, opts)
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
            idx = 1:size(X,1);
            F = zeros(size(X,1),1);
            [ s ] = FastStumpLearner.find_step(F, @(f)self.loss_func(f,Y,idx));
            self.stumps = [1 0 s s];
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
            % Compute the best split point for each feature, tracking best
            % feat/split pair. Array feat_info(i,1) contains the best found
            % "sum" for the i'th feature and feat_info(i,2) contains the split 
            % point which produced that sum.
            feat_count = size(X,2);
            feat_info = zeros(feat_count, 2);
            for f_num=1:feat_count,
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
               % Record the best "sum" and corresponding split for this feat
               feat_info(f_num,1) = f_sum;
               feat_info(f_num,2) = f_val;
            end
            % Find the best feature/split-point pair
            [best_err best_feat] = max(feat_info(:,1));
            best_thresh = feat_info(best_feat,2);
            % For the best split point, compute left and right weights
            l_idx = find(X(:,best_feat) <= best_thresh);
            w_l = self.find_step(F, @( f ) self.loss_func(f, Y, l_idx));
            r_idx = find(X(:,best_feat) > best_thresh);
            w_r = self.find_step(F, @( f ) self.loss_func(f, Y, r_idx));
            % Append the best split found as a new (reweighted) stump
            stump = [best_feat best_thresh (w_l*self.nu) (w_r*self.nu)];
            self.stumps = [self.stumps; stump];
            self.Ft(l_idx) = self.Ft(l_idx) + stump(3);
            self.Ft(r_idx) = self.Ft(r_idx) + stump(4);
            L_new = self.loss_func(self.Ft, Y, 1:obs_count);
            if (L_new - L > 1e-5)
                error('oops: loss increase in stump extension\n');
            end
            L = L_new;
            % Undo addition of stump if keep_it ~= 1
            if (keep_it ~= 1)
                self.stumps = self.stumps(1:end-1,:);
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
    end
    
end

