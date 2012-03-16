classdef SparseLearner < Learner
    % A simple class for performing boosted sparse regressions. Regressions in
    % this learner are first performed as elastic-net logistic regressions
    % targeted at the current loss gradients, with the set of non-zero
    % coefficients produced by this regression then updated using an 
    % l2-regularized hyperbolic tangent regression. The number of non-zero
    % coefficients to select in the first regression is a parameter.
    %
    % Accepted options:
    %   opts.nu: shrinkage/regularization term for boosting
    %   opts.loss_func: Loss function handle to a function that can be wrapped
    %                   around hypothesis outputs F as @(F)loss_func(F,Y).
    %   opts.do_opt: This indicates whether to use fast training optimization.
    %                This should only be set to 1 if all training rounds will
    %                use the same training set of observations/classes.
    %   opts.nz_count: Number of non-zero coefficients to produce in each
    %                  elastic-net logistic regression.
    %
    
    properties
        % coeffs stores the coefficients (including a bias term) for each
        % regression from which this learner is composed
        coeffs
        % weights stores the left and right weights for the regressions from
        % which this learner is composed
        weights
        % nz_count determines the number of non-zeros each set of coefficients
        nz_count
        % Xt is an optional fixed training set, used if opt_train==1
        Xt
        % Ft is the current output of this learner for each row in Xt
        Ft
        % opt_train indicates if to use fast training optimization
        opt_train
    end
    
    methods
        function [ self ] = SparseLearner(X, Y, opts)
            % Simple constructor for boosted regressions, which initializes the
            % set of regressions with a "constant" split
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
            if ~isfield(opts,'nz_count')
                self.nz_count = round(size(X,2)/2);
            else
                self.nz_count = opts.nz_count;
            end
            % Compute a constant step to apply to all observations
            F = zeros(size(X,1),1);
            step_func = @( f ) self.loss_func(f, Y, 1:size(Y,1));
            [ s ] = SparseLearner.find_step(F, step_func);
            self.coeffs = zeros(1,size(X,2)+1);
            self.weights = [s s];
            return
        end
        
        function [ L ] = extend(self, X, Y, keep_it)
            % Extend the current set of regressions, based on the observations
            % in X, using the classes in Y, and keeping changes if keep_it==1.
            if ~exist('keep_it','var')
                keep_it = 1;
            end
            if (self.opt_train ~= 1)
                F = self.evaluate(X);
            else
                F = self.Ft;
                X = self.Xt;
            end
            obs_count = size(X,1);
            [L dL] = self.loss_func(F, Y, 1:obs_count);
            % Do a sparsifying regression to find a good set of coefficients
            w = SparseLearner.do_reg(X, dL, self.nz_count);
            % For the best splitting regression, compute left and right weights
            X = [X ones(size(X,1),1)];
            step_func = @( f ) self.loss_func(f, Y, find(X*w <= 0));
            w_l = SparseLearner.find_step(F, step_func);
            step_func = @( f ) self.loss_func(f, Y, find(X*w > 0));
            w_r = SparseLearner.find_step(F, step_func);
            % Append the best split found as a new (reweighted) regression
            self.coeffs = [self.coeffs; w'];
            self.weights = [self.weights; w_l w_r]; 
            if (self.opt_train == 1)
                % Use fast training optimization via incremental evaluation
                Ft_new = self.evaluate(self.Xt, size(self.coeffs,1));
                self.Ft = self.Ft + Ft_new;
                F = self.Ft;
            else
                F = self.evaluate(X(:,1:end-1));
            end
            L = self.loss_func(F, Y, 1:obs_count);
            % Undo addition of regression if keep_it ~= 1
            if (keep_it ~= 1)
                self.coeffs = self.coeffs(1:end-1,:);
                self.weights = self.weights(1:end-1,:);
                if (self.opt_train == 1)
                    self.Ft = self.Ft - Ft_new;
                end
            end
            return 
        end
        
        function [ F ] = evaluate(self, X, idx)
            % Evaluate the regression splits underlying this learner.
            %
            % Parameters:
            %   X: observations to evaluate (a bias column will be added)
            %   idx: indices of regression splits to evaluate (optional, with
            %        the default behavior being evaluation of all splits).
            % Output:
            %   F: joint hypothesis output for each observation in X
            %
            if ~exist('idx','var')
                idx = 1:size(self.coeffs,1);
            end 
            if (idx == -1)
                idx = size(self.coeffs,1);
            end
            reg_count = length(idx);
            F = zeros(size(X,1), 1);
            % Augment X to accomodate bias term
            X = [X ones(size(X,1),1)];
            for i=1:reg_count,
                r_num = idx(i);
                b = self.coeffs(r_num,:)';
                r = X*b;
                F(r <= 0) = F(r <= 0) + self.weights(r_num,1);
                F(r > 0) = F(r > 0) + self.weights(r_num,2);
            end
            return
        end       
    end
    methods (Static = true)
        function [ step ] = find_step(F, step_func)
            % Use Matlab unconstrained optimization to find a step length that
            % minimizes: step_func(F + step)
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
        
        function [ w ] = do_reg( X, Y, nz_count )
            % Compute a sparsifying elastic-net logistic regression, followed by
            % weight tweaking using an l2-regularized hyperbolic tangent
            % regression.
            %
            % Parameters:
            %   X: the input observations (obs_count x obs_dim)
            %   Y: the weighted classifications (obs_count x 1)
            %   nz_count: the number of non-zero coefficients to produce
            %
            % Outputs:
            %   w: the weights of the regression (obs_dim(+1) x 1)
            %
            glm_opts = glmnetSet();
            glm_opts.dfmax = ceil(3.0 * nz_count);
            glm_opts.alpha = 0.75;
            glm_opts.nlambda = 300;
            glm_opts.thresh = 1e-3;
            % Decompose the target gradients into "classes" and "weights" for the lr
            Ys = round((sign(Y)+3)./2);
            Yw = abs(Y);
            glm_opts.weights = Yw;
            % Fit a weighted elastic-net-regularized logistic regression
            %     fit.beta(:,i): coefficients of ith regression
            %     fit.a0(i): bias of ith regression
            fit = glmnet(X, Ys, 'binomial', glm_opts);
            fit_count = size(fit.beta,2);
            % Scan the fit to find the result with most non-zero coefficients <= nz_count
            for f_num=1:fit_count,
                f_coeffs = fit.beta(:,f_num);
                if (nnz(f_coeffs) > nz_count || f_num == fit_count)
                    best_coeffs = fit.beta(:,f_num-1);
                    best_bias = fit.a0(f_num-1);
                    break;
                end
            end
            try
                best_coeffs = reshape(best_coeffs,numel(best_coeffs),1);
            catch
                error('SparseLearner.do_reg(): no best_coeffs\n');
            end
            % Setup options for minFunc
            mf_opts = struct();
            mf_opts.Display = 'off';
            mf_opts.Method = 'lbfgs';
            mf_opts.Corr = 5;
            mf_opts.LS = 1;
            mf_opts.LS_init = 3;
            mf_opts.MaxIter = 200;
            mf_opts.MaxFunEvals = 500;
            mf_opts.TolX = 1e-8;
            % Do a hyper tangent regression update of the coefficients for non-linearity
            nz_idx = find(best_coeffs);
            Xnz = [X(:,nz_idx) ones(size(X,1),1)];
            wi = [best_coeffs(nz_idx); best_bias];
            funObj = @( b ) SparseLearner.l2_hyptan(b, Xnz, Y, 1e-3, 0);
            wi = minFunc(funObj, wi, mf_opts);
            % Reload best coefficients and bias using tweaked values
            w = zeros(numel(best_coeffs)+1,1);
            w(nz_idx) = wi(1:numel(nz_idx));
            w(end) = wi(end);
            return
        end
        
        function [ L dLdW ] = l2_hyptan(w, X, Y, lam, pen_last)
            % Compute the objective function value for L2-regularized weighted
            % hyperbolic tangent regression.
            %
            % Parameters:
            %   w: parameter values for which to compute objective and gradient
            %   X: input observations
            %   Y: weighted target classes
            %   lam: regularization weight
            %   pen_last: whether to regularize last element of w. To be used
            %             when this element represents a bias term.
            %
            % Output:
            %   L: total loss over observations in X
            %   dLdW: gradient of objective with respect to elements of w
            %
            if ~exist('pen_last','var')
                pen_last = 1;
            end
            a = 1.0; % Smoothing/scale for hyperbolic tangent
            obs_count = size(X,1);
            obs_dim = size(X,2);
            lam = lam / obs_dim;
            % Decompose Y into sign and magnitude components
            Ys = sign(Y);
            Ym = abs(Y);
            % Compute objective function value
            F = X*w;
            if (pen_last == 1)
                loss_reg = sum(w.^2);
            else
                loss_reg = sum(w(1:end-1).^2);
            end
            loss_class = sum(Ym .* (1 - tanh((Ys .* a) .* F))) / obs_count;
            L = loss_class + ((lam/2) * loss_reg);
            if (nargout > 1)
                % Compute objective function gradients
                dR = w;
                if (pen_last == 0)
                    dR(end) = 0;
                end
                % Loss gradient with respect to output at each input
                dL = (Ym .* ((tanh((Ys .* a) .* F).^2 - 1) .* (Ys .* a)));
                % Backpropagate through input observations
                dLdW = sum(bsxfun(@times, X .* a, dL));
                dLdW = dLdW' + (lam * dR);
            end
            return
        end
        
    end
    
end

