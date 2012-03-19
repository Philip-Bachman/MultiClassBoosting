classdef VecStumpLearner < Learner
    % A simple class for boosting stumps for real-valued observations. The
    % constructor for this class accepts a set of observations X, a set of
    % target (possibly weighted) classes Y, and an options structure. This stump
    % learner is designed for vector-valued output and thus requires a loss
    % function capable of handling vector-valued inputs.
    %
    % Accepted options:
    %   opts.nu: shrinkage/regularization term for boosting
    %   opts.l_dim: dimension of the vector-valued output for this learner
    %   opts.loss_func: Loss function handle to a function that can be wrapped
    %                   around hypothesis outputs F as @(F)loss_func(F,Y).
    %   opts.do_opt: This indicates whether to use fast training optimization.
    %                This should only be set to 1 if all training rounds will
    %                use the same training set of observations/classes.
    %
    
    properties
        % stumps is a cell array of stump structs, each with these fields:
        %   stump.feat: feature (idx) on which to split
        %   stump.thresh: threshold value for the split
        %   stump.w_l: weight for points <= split
        %   stump.w_r: weight for points > split
        %   stump.v_l: vector-valued output for points <= split
        %   stump.v_r: vector-valued output for points > split
        stumps
        % l_dim is the dimension of the vector-valued output for this learner
        l_dim
        % p is a scaling factor contolling "selectivity" of split choice
        p
        % Xt is an optional fixed training set, used if opt_train==1
        Xt
        % Ft is the current output of this learner for each row in Xt
        Ft
        % opt_train indicates if to use fast training optimization
        opt_train
    end
    
    methods
        function [ self ] = VecStumpLearner(X, Y, opts)
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
            if ~isfield(opts,'l_dim')
                self.l_dim = 1;
            else
                self.l_dim = opts.l_dim;
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
                self.Ft = zeros(size(X,1),self.l_dim);
            end
            self.p = 0.5;
            % TODO: devise and implement a reasonable way of intializing with a
            % "constant" output.
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
            loss_func = @( f ) self.loss_func(f, Y);
            % First, evaluate learner and compute loss/gradient
            if (self.opt_train ~= 1)
                F = self.evaluate(X);
            else
                F = self.Ft;
                X = self.Xt;
            end
            [L dLdF] = loss_func(F);
            % Compute a split point based on the computed gradient
            feat_count = size(X,2);
            obs_count = size(X,1);
            best_feat = 0;
            best_thresh = 0;
            best_err = sum(sum(dLdF.^2,2));
            best_v_l = zeros(1,size(dLdF,2));
            best_v_r = zeros(1,size(dLdF,2));
            r_sz = [(obs_count-1):-1:1 1];
            % Compute the best split point for each feature, tracking best
            % feat/split pair
            for f_num=1:feat_count,
               [f_vals f_idx] = sort(X(:,f_num),'ascend');
               f_grad = dLdF(f_idx,:);
               f_err = best_err + 1;
               f_val = 0;
               f_v_l = zeros(1,size(dLdF,2));
               f_v_r = zeros(1,size(dLdF,2));
               csums = cumsum(f_grad,1);
               cssq = cumsum(f_grad.^2,1);
               % For the current feature, check all possible split points, 
               % tracking best split point and its corresponding error
               for s_num=1:obs_count,
                   %if (s_num==obs_count || f_vals(s_num)<f_vals(s_num+1))
                       % Compute the error for points left of the split
                       l_err = cssq(s_num,:) - (csums(s_num,:).^2 ./ s_num);
                       % Compute the error for points right of the split
                       r_err = (cssq(end,:) - cssq(s_num,:)) - ...
                           ((csums(end,:)-csums(s_num,:)).^2 ./ r_sz(s_num));
                       % Compute the joint left/right error and check if best
                       if ((sum(l_err) + sum(r_err)) < f_err)
                           % For best error yet, record: error, left/right mean
                           % vectors, and a splitting threshold.
                           f_err = sum(l_err) + sum(r_err);
                           f_v_l = csums(s_num,:);
                           f_v_vl = f_v_l ./ norm(f_v_l);
                           f_v_r = (csums(end,:) - csums(s_num,:));
                           f_v_r = f_v_r ./ norm(f_v_r);
                           % Compute a partially randomized split point
                           if (s_num < obs_count)
                               f_val = f_vals(s_num) + ...
                                   (rand()*(f_vals(s_num+1)-f_vals(s_num)));
                           else
                               f_val = f_vals(obs_count) + 1;
                           end
                       end
                   %end
               end
               % Check if the best split point found for this feature is better
               % than any split point found for previously examined features
               if (f_err < best_err)
                   best_err = f_err;
                   best_feat = f_num;
                   best_thresh = f_val;
                   best_v_l = f_v_l;
                   best_v_r = f_v_r;
               end
            end
            % For the best split point, compute left and right weights
            Fs = zeros(size(F));
            l_idx = X(:,best_feat) <= best_thresh;
            Fs(l_idx,:) = repmat(best_v_l,sum(l_idx),1);
            w_l = StumpLearner.find_step(F, Fs, loss_func);
            Fs = zeros(size(F));
            Fs(~l_idx,:) = repmat(best_v_r,sum(~l_idx),1);
            w_r = StumpLearner.find_step(F, Fs, loss_func);
            % Append the best split found as a new (reweighted) stump
            stump = struct();
            stump.feat = best_feat;
            stump.thresh = best_thresh;
            stump.v_l = best_v_l;
            stump.v_r = best_v_r;
            stump.w_l = w_l * self.nu;
            stump.w_r = w_r * self.nu;
            self.stumps{end+1} = stump;
            if (self.opt_train == 1)
                % Use fast training optimization via incremental evaluation
                Ft_new = self.evaluate(self.Xt, size(self.stumps,1));
                self.Ft = self.Ft + Ft_new;
                F = self.Ft;
            else
                F = self.evaluate(X);
            end
            L = loss_func(F);
            % Undo addition of stump if keep_it ~= 1
            if (keep_it ~= 1)
                self.stumps = {self.stumps{1:end-1,:}};
                if (self.opt_train == 1)
                    self.Ft = self.Ft - Ft_new;
                end
            end
            return 
        end
        
        function [ L ] = extend_mdm(self, X, Y, keep_it)
            % Extend the current set of stumps, based on the observations in X
            % and the loss/grad function loss_func. Return the post-update loss
            loss_func = @( f ) self.loss_func(f, Y);
            % First, evaluate learner and compute loss/gradient
            if (self.opt_train ~= 1)
                F = self.evaluate(X);
            else
                F = self.Ft;
                X = self.Xt;
            end
            [L dLdF] = loss_func(F);
            % Compute a split point based on the computed gradient
            feat_count = size(X,2);
            obs_count = size(X,1);
            best_feat = 0;
            best_thresh = 0;
            best_sum = 0;
            best_v_l = zeros(1,size(dLdF,2));
            best_v_r = zeros(1,size(dLdF,2));
            % Compute the best split point for each feature, tracking best
            % feat/split pair
            for f_num=1:feat_count,
               [f_vals f_idx] = sort(X(:,f_num),'ascend');
               f_grad = dLdF(f_idx,:);
               f_sum = best_sum - 1;
               f_val = 0;
               f_v_l = zeros(1,size(dLdF,2));
               f_v_r = zeros(1,size(dLdF,2));
               l_sums = cumsum(f_grad,1);
               r_sums = bsxfun(@plus, -l_sums, l_sums(end));
               % Compute a nonlinear, large value emphasizing transform of
               % l_sums and r_sums
               l_sums_sa = sqrt(l_sums.^2 + 1e-8).^self.p;
               r_sums_sa = sqrt(r_sums.^2 + 1e-8).^self.p;
               l_sums_sa = bsxfun(@rdivide,(l_sums.*l_sums_sa),sum(l_sums_sa,2));
               r_sums_sa = bsxfun(@rdivide,(r_sums.*r_sums_sa),sum(r_sums_sa,2));
               % For the current feature, check all possible split points, 
               % tracking best split point and its corresponding error
               for s_num=1:obs_count,
                   % Compute the joint left/right sums and check if it is best
                   l_vec = l_sums_sa(s_num,:);
                   r_vec = r_sums_sa(s_num,:);
                   if (s_num == obs_count || f_vals(s_num) < f_vals(s_num+1))
                       if ( ( sum(abs(l_vec)) + sum(abs(r_vec)) ) > f_sum )
                           % For best sum yet, record: sum, left and right mean
                           % vectors, and a splitting threshold.
                           f_sum = sum(abs(l_vec)) + sum(abs(r_vec));
                           f_v_l = l_vec ./ norm(l_vec);
                           f_v_r = r_vec ./ norm(r_vec);
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
                   best_v_l = f_v_l;
                   best_v_r = f_v_r;
               end
            end
            % For the best split point, compute left and right weights
            Fs = zeros(size(F));
            l_idx = X(:,best_feat) <= best_thresh;
            Fs(l_idx,:) = repmat(best_v_l,sum(l_idx),1);
            w_l = VecStumpLearner.find_step(F, Fs, loss_func);
            Fs = zeros(size(F));
            Fs(~l_idx,:) = repmat(best_v_r,sum(~l_idx),1);
            w_r = VecStumpLearner.find_step(F, Fs, loss_func);
            % Append the best split found as a new (reweighted) stump
            stump = struct();
            stump.feat = best_feat;
            stump.thresh = best_thresh;
            stump.v_l = best_v_l;
            stump.v_r = best_v_r;
            stump.w_l = w_l * self.nu;
            stump.w_r = w_r * self.nu;
            self.stumps{end+1} = stump;
            if (self.opt_train == 1)
                % Use fast training optimization via incremental evaluation
                Ft_new = self.evaluate(self.Xt, size(self.stumps,1));
                self.Ft = self.Ft + Ft_new;
                F = self.Ft;
            else
                F = self.evaluate(X);
            end
            L = loss_func(F);
            % Undo addition of stump if keep_it ~= 1
            if (keep_it ~= 1)
                self.stumps = {self.stumps{1:end-1,:}};
                if (self.opt_train == 1)
                    self.Ft = self.Ft - Ft_new;
                end
            end
            return 
        end
        
        function [ F ] = evaluate(self, X, idx)
            % Evaluate the current set of stumps associated with this learner
            if ~exist('idx','var')
                idx = 1:length(self.stumps);
            end
            if (idx == -1)
                idx = length(self.stumps);
            end
            obs_count = size(X,1);
            stump_count = numel(idx);
            F = zeros(obs_count, self.l_dim);
            for s_num=1:stump_count,
                s_idx = idx(s_num);
                s_feat = self.stumps{s_idx}.feat;
                s_thresh = self.stumps{s_idx}.thresh;
                l_idx = X(:,s_feat) <= s_thresh;
                F(l_idx,:) = bsxfun(@plus, F(l_idx,:),...
                    (self.stumps{s_idx}.v_l .* self.stumps{s_idx}.w_l));
                F(~l_idx,:) = bsxfun(@plus, F(~l_idx,:),...
                    (self.stumps{s_idx}.v_r .* self.stumps{s_idx}.w_r));
            end
            return
        end       
    end
    methods (Static = true)
        function [ step ] = find_step(F, Fs, loss_func)
            % Use Matlab unconstrained optimization to find a step length that
            % minimizes: loss_func(F + Fs.*step)
            options = optimset('MaxFunEvals',30,'TolX',1e-3,'TolFun',1e-3,...
                'Display','off');
            [L dL] = loss_func(F);
            if (sum(sum(Fs.*dL)) > 0)
                step = fminbnd(@( s ) loss_func(F + Fs.*s), -1, 0, options);
            else
                step = fminbnd(@( s ) loss_func(F + Fs.*s), 0, 1, options);
            end
            return
        end
    end
    
end

