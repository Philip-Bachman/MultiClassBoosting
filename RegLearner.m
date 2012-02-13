classdef RegLearner < Learner
    % A simple class for performing boosted regressions. Regressions in this
    % learner are first performed as l2-regularized logistic regressions
    % (targeted at current loss gradients), which is used to "warm start"
    % l2-regularized hyperbolic tangent regression. The second regression
    % corresponds more closely to the objective sum_i(h(x_i)*g_i), where x_i is
    % an input and g_i is the current loss gradient with respect to x_i.
    %
    % Accepted options:
    %   opts.nu: shrinkage/regularization term for boosting
    %   opts.loss_func: Loss function handle to a function that can be wrapped
    %                   around hypothesis outputs F as @(F)loss_func(F,Y).
    %   opts.do_opt: This indicates whether to use fast training optimization.
    %                This should only be set to 1 if all training rounds will
    %                use the same training set of observations/classes.
    %   opts.lambda: l2 regularization weight for regressions
    %
    
    properties
        % coeffs stores the coefficients (including a bias term) for each
        % regression from which this learner is composed
        coeffs
        % weights stores the left and right weights for the regressions from
        % which this learner is composed
        weights
        % lambda is the (L2) regularization weight to use during regressions
        lambda
        % Xt is an optional fixed training set, used if opt_train==1
        Xt
        % Ft is the current output of this learner for each row in Xt
        Ft
        % opt_train indicates if to use fast training optimization
        opt_train
    end
    
    methods
        function [ self ] = RegLearner(X, Y, opts)
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
            if ~isfield(opts,'lambda')
                self.lambda = 1e-2;
            else
                self.lambda = opts.lambda;
            end
            % Compute a constant step to apply to all observations
            F = zeros(size(X,1),1);
            step_func = @( f ) self.loss_func(f, Y, 1:size(Y,1));
            [ s ] = RegLearner.find_step(F, step_func);
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
            % First, do a weighted logistic regression
            X = [X ones(size(X,1),1)];
            w = RegLearner.do_reg(X, dL, self.lambda, 250, 1, 0);
            % Second, do a warm-started hyperbolic tangent regression
            w = RegLearner.do_reg(X, dL, self.lambda, 250, 2, 0, w);
            % For the best splitting regression, compute left and right weights
            step_func = @( f ) self.loss_func(f, Y, find(X*w <= 0));
            w_l = RegLearner.find_step(F, step_func);
            step_func = @( f ) self.loss_func(f, Y, find(X*w > 0));
            w_r = RegLearner.find_step(F, step_func);
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
        
        function [ w ] = do_reg( X, Y, lam, max_iter, loss_type, pen_last, wi )
            % Compute a weighted l2-regularized regression using either
            % hyperbolic tangent loss or binomial deviance loss.
            %
            % Parameters:
            %   X: the input observations (obs_count x obs_dim)
            %   Y: the weighted classifications (obs_count x 1)
            %   lam: the regularization weight for l2 penalty
            %   max_iter: maximum number of minFunc iterations
            %   loss_type: loss type to use (1: bin dev, 2: hyp tan)
            %   pen_last: whether to regularize the final element of w
            %   wi: optional initialization vector for w
            %
            % Outputs:
            %   beta: the weights of the regression (obs_dim(+1) x 1)
            %
            if ~exist('max_iter','var')
                max_iter = 500;
            end
            if ~exist('loss_type','var')
                loss_type = 1;
            end
            if ~exist('pen_last','var')
                pen_last = 1;
            end
            if ~exist('wi','var')
                w = zeros(size(X,2),1);
            else
                w = wi; 
            end
            % Setup options for minFunc
            options = struct();
            options.Display = 'off';
            options.Method = 'lbfgs';
            options.Corr = 5;
            options.LS = 1;
            options.LS_init = 3;
            options.MaxIter = max_iter;
            options.MaxFunEvals = 2*max_iter;
            options.TolX = 1e-8;
            % Select the loss function to use, based on "loss_type"
            if (loss_type == 1)
                funObj = @( b ) RegLearner.l2_logreg(b, X, Y, lam, pen_last);
            else
                funObj = @( b ) RegLearner.l2_hyptan(b, X, Y, lam, pen_last);
            end
            w = minFunc(funObj, w, options);
            return
        end

        function [ L dLdW ] = l2_logreg(w, X, Y, lam, pen_last)
            % Compute the objective function value for L2-regularized weighted
            % logistic regression. This uses a weighted binomial deviance loss.
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
            loss_class = sum(Ym .* log(exp(-Ys.*F) + 1)) / obs_count;
            L = loss_class + ((lam/2) * loss_reg);
            if (nargout > 1)
                % Compute objective function gradients
                dR = w;
                if (pen_last == 0)
                    dR(end) = 0;
                end
                % Loss gradient with respect to output at each input
                dL = Ym .* (-Ys .* (exp(-Ys.*F) ./ (exp(-Ys.*F) + 1)));
                % Backpropagate through input observations
                dLdW = sum(bsxfun(@times, X, dL));
                dLdW = dLdW' + (lam * dR);
            end
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

