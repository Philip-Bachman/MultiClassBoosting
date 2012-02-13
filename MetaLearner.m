classdef MetaLearner < Learner
    % A simple class for performing MetaBoost a.k.a Boosters, Boosters, Golly!
    %
    % Accepted options:
    %   opts.nu: shrinkage/regularization term for boosting
    %   opts.loss_func: Loss function handle to a function that can be wrapped
    %                   around hypothesis outputs F as @(F)loss_func(F,Y).
    %   opts.do_opt: This indicates whether to use fast training optimization.
    %                This should only be set to 1 if all training rounds will
    %                use the same training set of observations/classes.
    %   opts.l_const: constructor handle for base learner
    %   opts.l_rounds: number of rounds to update base learner
    %   opts.l_opts: options struct to pass to opts.l_const
    %
    properties
        % l_objs is a cell array containing the learners from which the meta
        % learner has been composed. For h = l_objs{i}, h.lrnr gives the
        % learner object handle, h.w_l gives the weight for x for which
        % h.lrnr.evaluate(x) <= 0, and h.w_r gives the weight for x for which
        % h.lrnr.evaluate(x) > 0.
        l_objs
        % l_const is a function handle for constructing base learners
        l_const
        % l_rounds gives the number of rounds to train each base learner
        l_rounds
        % l_opts gives the options structure to pass to self.l_const
        l_opts
        % Xt is an optional fixed training set, used if opt_train==1
        Xt
        % Ft is the current output of this learner for each row in Xt
        Ft
        % opt_train indicates if to use fast training optimization
        opt_train
    end % END PROPERTIES
    
    methods
        function [ self ] = MetaLearner(X, Y, opts)
            % Simple constructor for MetaBoost, which initializes with a
            % constant stump.
            %
            if ~exist('opts','var')
                opts = struct();
            end
            if ~isfield(opts,'nu')
                self.nu = 1.0;
            else
                self.nu = opts.nu;
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
            if ~isfield(opts, 'loss_func')
                self.loss_func = @loss_bindev;
            else
                self.loss_func = opts.loss_func;
            end
            if ~isfield(opts,'l_opts')
                opts.l_opts = struct();
            end
            if ~isfield(opts,'l_const')
                opts.l_const = @StumpLearner;
            end
            if ~isfield(opts,'l_rounds')
                opts.l_rounds = 10;
            end
            % Store the specified learner constructor, round count, and nu
            self.l_const = opts.l_const;
            self.l_rounds = opts.l_rounds;
            self.l_opts = opts.l_opts;
            self.l_opts.do_opt = 1;
            % Create a constant stump and add to the hypothesis set
            h = struct();
            h.lrnr = StumpLearner(X, Y, self.l_opts);
            h.w_l = h.lrnr.evaluate(X(1,:));
            h.w_r = h.lrnr.evaluate(X(1,:));
            self.l_objs = {h};
            return
        end
        
        function [ L ] = extend(self, X, Y, keep_it)
            % Extend the current set of base learners. Construct a new base
            % learner, and extend it as many times as desired, then add a
            % binarized, reweighted version of the learner to the set h.
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
            [L dLdF] = self.loss_func(F, Y, 1:obs_count);
            %self.l_opts.loss_func = @loss_hyptan;
            self.l_opts.loss_func = @loss_lsq;
            %self.l_opts.loss_func = @loss_huberreg;
            lrnr = self.l_const(X, dLdF, self.l_opts);
            for r=1:self.l_rounds,
                lrnr.extend(X, dLdF, 1);
            end
            % Evaluate the learned learner for each observation
            Fl = lrnr.evaluate(X);
            % Given the induced data split, compute left and right weights
            step_func = @( f ) self.loss_func(f, Y, find(Fl <= 0));
            w_l = MetaLearner.find_step(F, step_func);
            step_func = @( f ) self.loss_func(f, Y, find(Fl > 0));
            w_r = MetaLearner.find_step(F, step_func);
            % create the struct describing this base learner
            h = struct();
            h.lrnr = lrnr;
            h.w_l = w_l * self.nu;
            h.w_r = w_r * self.nu;
            self.l_objs{end+1} = h;
            % Evaluate the updated learner
            if (self.opt_train == 1)
                % Use fast training optimization via incremental evaluation
                Ft_new = self.evaluate(self.Xt, length(self.l_objs));
                self.Ft = self.Ft + Ft_new;
                F = self.Ft;
            else
                F = self.evaluate(X);
            end
            L = self.loss_func(F, Y, 1:obs_count);
            % Undo addition of base learner if keep_it ~= 1
            if (keep_it ~= 1)
                self.l_objs = {self.l_objs{1:end-1}};
                if (self.opt_train == 1)
                    self.Ft = self.Ft - Ft_new;
                end
            end
            return
        end
        
        function [ F ] = evaluate(self, X, idx)
            % Evaluate the current set of base learners from which this meta
            % learner is composed.
            if ~exist('idx','var')
                idx = 1:length(self.l_objs);
            end;
            if (idx == -1)
                idx = length(self.l_objs);
            end
            F = zeros(size(X,1),1);
            for l_num=1:length(idx),
                h = self.l_objs{idx(l_num)};
                Fh = h.lrnr.evaluate(X);
                F(Fh <= 0) = F(Fh <= 0) + h.w_l;
                F(Fh > 0) = F(Fh > 0) + h.w_r;
            end
            return
        end       
    end % END METHODS
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
    end % END METHODS (STATIC)
    
end % END CLASSDEF

