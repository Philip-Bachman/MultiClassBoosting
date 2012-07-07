classdef RBFLearner < Learner
    % RBFLearner is a class for learning boosted RBF classifiers/regressors.
    % At each round of boosting, a new "weak learner" is produced either by
    % fitting an RBF classifier, via a pair of regularized weighted
    % classification problems, or an RBF regressor (fit analogously to the
    % classifier) to the current functional gradient. The classifier or
    % regressor then constitutes a step direction in function space for which a
    % step size is selected via line-search.
    %
    % Accepted options:
    %   opts.nu: shrinkage/regularization term for boosting
    %   opts.loss_func: Loss function handle to a function that can be wrapped
    %                   around hypothesis outputs F as @(F)loss_func(F,Y).
    %   opts.nz_count: Number of non-zero coefficients to produce in each RBF
    %                  classifier and/or regressor.
    %   opts.rbf_type: 1 for classifier and 2 for regressor
    %   opts.rbf_gamma: rbf_gamma dtermines the bandwidth of RBFs
    %
    
    properties
        % rbf_funcs stores the set of learned RBF-based weak learners, each
        % described by a struct containing the RBF centers and coefficients for
        % a weak learner, the type of the learner (i.e. class/regress), and
        % either one or two step weights, for regress/class respectively.
        %
        % IF rbf_func = rbf_funcs{i}, rbf_func contains:
        %   rbf_func.centers: matrix containing RBF centers
        %   rbf_func.coeffs: array of coefficients for the centers
        %   rbf_func.gammas: width of the RBF for each center
        %   rbf_func.weights: the one (two) weight(s) for regress/class func
        %   rbf_func.type: 1/2, for classifier/regressor, respectively
        %
        rbf_funcs
        % nz_count determines the number of non-zeros each set of coefficients
        nz_count
        % rbf_type determines the type of weak learner to produce in each
        % boosting round.
        rbf_type
        % rbf_gamma determines bandwidth of the RBFs used in a boosting round
        rbf_gammas
        % rbf_centers contains the potential RBF centers for a boostong round
        rbf_centers
        % lam_l2 is a regularization weight used during "post-processing"
        lam_l2
    end
    
    methods
        function [ self ] = RBFLearner(X, Y, opts)
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
            if ~isfield(opts,'nz_count')
                self.nz_count = 10;
            else
                self.nz_count = opts.nz_count;
            end
            if ~isfield(opts,'rbf_type')
                self.rbf_type = 1;
            else
                self.rbf_type = opts.rbf_type;
                self.check_rbf_type();
            end
            if ~isfield(opts,'rbf_gamma')
                % Default to gamma = 1/num_feat (same as for libsvm)
                self.rbf_gammas = 1 / size(X,2);
            else
                self.rbf_gammas = opts.rbf_gammas;
            end
            self.rbf_centers = [];
            self.rbf_funcs = {};
            % Compute a constant step to apply to all observations
            F = zeros(size(X,1),1);
            Fs = ones(size(F));
            step_func = @( f ) self.loss_func(f, Y, 1:size(Y,1));
            [ s ] = self.find_step(F, Fs, step_func);
            % Create an rbf_func structure to hold the "contant" rbf step
            rbf_func = struct();
            rbf_func.centers = X(1,:);
            rbf_func.coeffs = 1;
            rbf_func.gammas = 1;
            rbf_func.type = 1;
            rbf_func.weights = [s s];
            self.rbf_funcs{1} = rbf_func;
            self.lam_l2 = 1e-1;
            return
        end
        
        function [ L F ] = extend(self, X, Y, keep_it)
            % Extend the current set of regressions, based on the observations
            % in X, using the classes in Y, and keeping changes if keep_it==1.
            self.check_rbf_type();
            if ~exist('keep_it','var')
                keep_it = 1;
            end
            if (numel(self.rbf_centers) == 0)
                error('RBFLearner.extend(): unset rbf centers.\n');
            end
            F = self.evaluate(X);
            obs_count = size(X,1);
            [L dL] = self.loss_func(F, Y, 1:obs_count);
            % Compute an RBF representation of X using current RBF centers
            [X_rbf centers gammas] = ...
                self.compute_rbfs(X, self.rbf_centers, self.rbf_gammas);
            % Do a sparsifying regression to find a good set of coefficients
            w = self.do_reg(X_rbf, dL, self.nz_count);
            % Compute the output of the learned RBF classifier/regressor for
            % each observation in X
            if (numel(w) > size(X_rbf,2))
                % There is a bias in w
                F_rbf = (X_rbf * w(1:end-1)) + w(end);
                nz_idx = find(w(1:end-1));
                w = [w(nz_idx); w(end)];
            else
                % There is no bias in w
                F_rbf = X_rbf * w;
                nz_idx = find(w);
                w = w(nz_idx);
            end
            % Build an rbf_func structure around the learned RBF
            rbf_func = struct();
            if (numel(nz_idx) > 0)
                rbf_func.centers = centers(nz_idx,:);
                rbf_func.gammas = gammas(nz_idx);
                rbf_func.coeffs = w;
            else
                rbf_func.centers = centers(1,:);
                rbf_func.gammas = 1;
                rbf_func.coeffs = 0;
            end
            rbf_func.type = self.rbf_type;
            % Set the rbf_func weights, dependent on rbf_type
            if (self.rbf_type == 1)
                % Find a good step size for a dichotomizing direction
                Fs = ones(size(F));
                step_func = @( f ) self.loss_func(f, Y, find(F_rbf <= 0));
                w_l = self.find_step(F, Fs, step_func);
                step_func = @( f ) self.loss_func(f, Y, find(F_rbf > 0));
                w_r = self.find_step(F, Fs, step_func);
                rbf_func.weights = [w_l w_r] .* self.nu;
            else
                % Find a good step size for a regression direction
                step_func = @( f ) self.loss_func(f, Y, 1:numel(Y));
                w = self.find_step(F, F_rbf, step_func);
                rbf_func.weights = w * self.nu;
            end
            % Append the learned RBF classifier/regressor to self.rbf_funcs
            self.rbf_funcs{end+1} = rbf_func;
            F = F + self.evaluate_rbf_func(X, rbf_func);
            L = self.loss_func(F, Y, 1:obs_count);
            % Undo addition of regression if keep_it ~= 1
            if (keep_it ~= 1)
                self.rbf_funcs(end) = [];
            end
            return 
        end
        
        function [ F ] = evaluate(self, X, idx)
            % Evaluate the RBF classifiers/regressors underlying this learner.
            %
            % Parameters:
            %   X: observations to evaluate (a bias column will be added)
            %   idx: indices of regression splits to evaluate (optional, with
            %        the default behavior being evaluation of all splits).
            % Output:
            %   F: joint hypothesis output for each observation in X
            %
            if ~exist('idx','var')
                idx = 1:length(self.rbf_funcs);
            end 
            if (idx == -1)
                idx = length(self.rbf_funcs);
            end
            self.check_rbf_type();
            rbf_count = length(idx);
            F = zeros(size(X,1), 1);
            for i=1:rbf_count,
                rbf_num = idx(i);
                Frbf = self.evaluate_rbf_func(X, self.rbf_funcs{rbf_num});
                F = F + Frbf;
            end
            return
        end

        function [ F ] = evaluate_rbf_func(self, X, rbf_func)
            % Evaluate the RBF function described by the struct rbf_func for
            % each observation in X
            %
            X_rbf = self.compute_rbfs(X, rbf_func.centers, rbf_func.gammas);
            if (numel(rbf_func.coeffs) > size(rbf_func.centers,1))
                % Compute when there is a bias
                F = (X_rbf * rbf_func.coeffs(1:end-1)) + rbf_func.coeffs(end);
            else
                % Compute when there is no bias
                F = X_rbf * rbf_func.coeffs;
            end
            if (self.rbf_type == 1)
                % Compute final output for classifiers/dichotomizers
                l_idx = F <= 0;
                F(l_idx) = rbf_func.weights(1);
                F(~l_idx) = rbf_func.weights(2);
            else
                % Compute final output for regressors
                F = F .* rbf_func.weights(1);
            end
            return
        end

        function [X_rbf centers gammas] = compute_rbfs(self, X, centers, gammas)
            % Use the current self.rbf_centers and self.rbf_gammas to compute
            % the RBF representation of the observations in X
            %
            if ~exist('centers','var')
                centers = self.rbf_centers;
            end
            if ~exist('gammas','var')
                gammas = self.rbf_gammas;
            end
            if (numel(gammas) < size(centers,1))
                gammas = repmat(gammas(1),size(centers,1),1);
            end
            rbf_count = size(centers,1);
            X_rbf = zeros(size(X,1),rbf_count);
            for i=1:rbf_count,
                if (gammas(i) < 0)
                    X_rbf(:,i) = log(1 + exp(X * centers(i,:)'));
                else
                    X_rbf(:,i) = exp(...
                        -sum(bsxfun(@minus,X,centers(i,:)).^2,2) .* gammas(i));
                end
            end
            return
        end

        function [res] = set_rbf_centers(self, centers, gammas, center_count)
            % set self.rbf_centers
            if (numel(gammas) < size(centers,1))
                gammas = repmat(gammas(1),size(centers,1),1);
            end
            if exist('center_count','var')
                % subsample the set of proposed centers, if so desired
                idx = randsample(size(centers,1),center_count);
                centers = centers(idx,:);
                gammas = gammas(idx);
            end
            self.rbf_centers = centers;
            self.rbf_gammas = gammas;
            res = 0;
            return
        end
        
        function [res] = check_rbf_type(self)
            % Check if self.rbf_type is valid (i.e. 1 or 2)
            if (self.rbf_type == 1 || self.rbf_type == 2)
                res = 1;
            else
                error('RBFLearner.check_rbf_type(): type should be 1 or 2.\n');
            end
            return
        end
        
        %%%%%%%%%%%%%%%%%%%%
        % REGRESSION STUFF %
        %%%%%%%%%%%%%%%%%%%%
        
        function [w] = do_reg(self, X, Y, nz_count)
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
            glm_opts.dfmax = ceil(1.5 * nz_count);
            glm_opts.pmax = 5 * glm_opts.dfmax;
            glm_opts.alpha = 0.9;
            glm_opts.nlambda = 500;
            glm_opts.thresh = 1e-4;
            if (self.rbf_type == 1)
                % Decompose the target gradients into "classes" and "weights"
                % for the logistic regression
                Ys = round((sign(Y)+3)./2);
                Yw = abs(Y);
                % Fit a weighted elastic-net-regularized logistic regression
                %     fit.beta(:,i): coefficients of ith regression
                %     fit.a0(i): bias of ith regression
                glm_opts.weights = Yw;
                fit = glmnet(X, Ys, 'binomial', glm_opts);
            else
                fit = glmnet(X, Y, 'gaussian', glm_opts);
            end
            fit_count = size(fit.beta,2);
            % Scan the fits to maximize non-zero coefficients <= nz_count
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
                error('RBFLearner.do_reg(): no best_coeffs\n');
            end
            % Setup options for minFunc
            %warning off all;
            mf_opts = struct();
            mf_opts.Display = 'off';
            mf_opts.Method = 'cg';
            mf_opts.Corr = 5;
            mf_opts.LS = 3;
            mf_opts.LS_init = 1;
            mf_opts.MaxIter = 50;
            mf_opts.MaxFunEvals = 250;
            mf_opts.TolX = 1e-6;
            mf_opts.TolFun = 1e-6;
            % Do a hyper tangent regression update of the coefficients for
            % non-linearity
            nz_idx = find(best_coeffs);
            if (numel(nz_idx) > 0)
                Xnz = [X(:,nz_idx) ones(size(X,1),1)];
                wi = [best_coeffs(nz_idx); best_bias];
                if (self.rbf_type == 1)
                    funObj = @( b ) self.l2_hyptan(b, Xnz, Y, self.lam_l2, 0);
                else
                    funObj = @( b ) self.l2_lsq(b, Xnz, Y, self.lam_l2, 0);
                end
                wi = minFunc(funObj, wi, mf_opts);
                % Reload best coefficients and bias using tweaked values
                w = zeros(numel(best_coeffs)+1,1);
                w(nz_idx) = wi(1:numel(nz_idx));
                w(end) = wi(end);
            else
                w = zeros(numel(best_coeffs)+1,1);
            end
            return
        end
        
        function [ L dLdW ] = l2_hyptan( self, w, X, Y, lam, pen_last )
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
            % Smoothing/scale for hyperbolic tangent
            a = 1.0;
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
            loss_class = sum(Ym .* (1 - tanh((Ys .* a) .* F)));
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
        
        function [ L dLdW ] = l2_lsq(self, w, X, Y, lam, pen_last)
            % Compute the objective value and gradient for least squares.
            %
            % Parameters:
            %   w: coefficients
            %   X: observations
            %   Y: target outputs
            %   lam: l2 regularization weight for coefficients
            %   pen_last: whether to penalize last (bias) coefficient
            %
            % Output:
            %   L: objective value
            %   dLdF: gradient of objective with respect to values in F
            %
            if ~exist('pen_last','var')
                pen_last = 1;
            end
            F = X * w;
            R = F - Y;
            % Compute least-squares and regularization loss
            L_res = sum(R.^2);
            L_reg = sum(w.^2);
            if (pen_last == 0)
                L_reg = L_reg - w(end).^2;
            end
            L = L_res + L_reg;
            if (nargout > 1)
                % Loss gradient with respect to residual
                dLdR = 2 * R;
                % Loss gradient with respect to coefficients
                dLdW = sum(bsxfun(@times,X,dLdR));
                dLdW = dLdW' + (2 * lam * w);
                if (pen_last == 0)
                   dLdW(end) = dLdW(end) - (2 * lam * w(end));
                end
                dLdW = dLdW ./ norm(dLdW);
            end
            return
        end
    end
end

