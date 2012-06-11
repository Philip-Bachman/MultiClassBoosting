classdef MCLearner < Learner
    % A simple class for performing multiple classifier boosting. The multiple
    % classifier boosting implementation in this class uses the sum-of-softmax
    % loss for hypothesis unification.
    %
    % Accepted options:
    %   opts.nu: shrinkage/regularization term for boosting
    %   opts.loss_func: Loss function handle to a function that can be wrapped
    %                   around hypothesis outputs F as @(F)loss_func(F,Y).
    %   opts.extend_all: if this is set, all hypotheses will be updated on each
    %                    cycle, in a random order
    %   opts.l_const: constructor handle for base learner
    %   opts.l_count: the number of base learners to use
    %   opts.l_opts: options struct to pass to opts.l_const
    %   opts.alpha: smoothing factor for sum-of-softmax loss
    %
    % Note: opts.l_opts.nu is overridden by opts.nu.
    %

    properties
        % l_objs is a cell array containing the learners from which the meta,
        % multiple classifier hypothesis (i.e. this class instance) is formed.
        l_objs
        % l_count is the number of base learners
        l_count
        % extend_all determines whether to extend all or only best
        extend_all
        % alpha is a scaling factor for sum-of-softmax
        alpha
    end % END PROPERTIES
    
    methods
        
        function [ self ] = MCLearner(X, Y, opts)
            % Simple constructor for Multiple Classifier boosting via
            % sum-of-softmax.
            %
            if ~exist('opts','var')
                opts = struct();
            end
            if ~isfield(opts,'nu')
                self.nu = 1.0;
            else
                self.nu = opts.nu;
            end
            if ~isfield(opts,'alpha')
                self.alpha = 1.0;
            else
                self.alpha = opts.alpha;
            end
            if ~isfield(opts,'loss_func')
                self.loss_func = @loss_bindev;
            else
                self.loss_func = opts.loss_func;
            end
            if ~isfield(opts,'l_const')
                opts.l_const = @StumpLearner;
            end
            if ~isfield(opts,'l_count')
                self.l_count = 4;
            else
                self.l_count = opts.l_count;
            end
            if ~isfield(opts,'l_opts')
                opts.l_opts = struct();
            end
            if ~isfield(opts,'extend_all')
                self.extend_all = 0;
            else
                self.extend_all = opts.extend_all;
            end
            % Create initial base learners from which the joint hypothesis will
            % be composed.
            opts.l_opts.nu = self.nu;
            self.l_objs = {};
            for i=1:self.l_count,
                self.l_objs{i} = opts.l_const(X, Y, opts.l_opts);
            end
            return
        end
        
        function [ L ] = extend(self, X, Y, keep_it)
            % Dispatcher, to select extend_best or extend_all.
            if ~exist('keep_it','var')
                keep_it = 1;
            end
            if (self.extend_all == 1)
                L = self.extend_tout(X, Y, keep_it);
            else
                L = self.extend_best(X, Y, keep_it);
            end
            return
        end   
        
        function [ L ] = extend_best(self, X, Y, keep_it)
            % Extend the current set of base learners. Find the base learner
            % that most improves the multiple hypothesis loss for the
            % observations in X and classes in Y.
            if ~exist('keep_it','var')
                keep_it = 1;
            end
            [F Fh] = self.evaluate(X);
            obs_count = size(X,1);
            % Find the learner that, if updated, will most reduce the loss
            L_min = self.sosm_loss(Fh, Y, self.alpha, 1:obs_count);
            l_min = 1;
            for l_num=1:self.l_count,
                lrnr = self.l_objs{l_num};
                lrnr.loss_func = @( Fc, Yc, idx ) self.loss_wrapper(...
                    Fc, Fh, Yc, l_num, self.alpha, idx);
                Ll = lrnr.extend(X, Y, 0);
                if (Ll < L_min)
                    L_min = Ll;
                    l_min = l_num;
                end
            end
            % Permanently update the best found base learner
            l_loss = @( Fc, Yc, idx ) self.loss_wrapper(...
                    Fc, Fh, Yc, l_min, self.alpha, idx);
            self.l_objs{l_min}.loss_func = l_loss;
            L = self.l_objs{l_min}.extend(X, Y, 1);
            return
        end
        
        function [ L ] = extend_tout(self, X, Y, keep_it)
            % Extend the current set of base learners. Using the sum-of-softmax 
            % activation for the observations in X and classes in Y={-1,+1},
            % cycle through all current base learners, using a randomized cycle
            % order, updating each in turn.
            %
            % Note: This is kind of incomplete for now.
            %
            if ~exist('keep_it','var')
                keep_it = 1;
            end
            % Iteratively update all learners
            l_order = randperm(self.l_count);
            for i=1:self.l_count,
                l_num = l_order(i);
                [F Fh] = self.evaluate(X);
                lrnr = self.l_objs{l_num};
                lrnr.loss_func = @( Fc, Yc, idx ) self.loss_wrapper(...
                    Fc, Fh, Yc, l_num, self.alpha, idx);
                L = lrnr.extend(X, Y, 1);
            end
            return
        end
        
        function [ F H ] = evaluate(self, X, idx)
            % Evaluate the current set of base learners from which this meta
            % learner is composed.
            if ~exist('idx','var')
                idx = 1;
            else
                if (idx == -1)
                    error('-1 eval index not yet supported for MCLearner\n');
                end
            end
            H = zeros(size(X,1),self.l_count);
            % Compute output of each base learner for each 
            for l_num=1:self.l_count,
                h = self.l_objs{l_num};
                H(:,l_num) = h.evaluate(X);
            end
            % Compute the sum-of-softmax for the values in F
            F = self.sosm_eval(H, self.alpha);
            return
        end
        
        function [ L dLdFc ] = loss_wrapper(self, Fc, Fa, Y, h_num, alpha, idx)
            % Create a wrapper function that computes loss and gradient of loss
            % with respect to a single column of H.
            %
            % Parameters:
            %   Fc: the output of the hypothesis under consideration
            %   Fa: the output of all hypotheses, including column h_num, which
            %       will be overwritten by Fc
            %   Y: target classes for each observation
            %   h_num: column index of hypothesis with respect to which to
            %          compute loss and gradient of loss
            %   alpha: smoothing factor for multiple hypothesis combination
            %   idx: indices at which to evaluate loss and gradients
            % Output:
            %   L: loss of the full joint hypothesis
            %   dLdFc: gradient of L with respect to column h_num of Fa.
            %
            Fa(:,h_num) = Fc(:);
            if (nargout > 1)
                [L dLdFa] = self.sosm_loss(Fa, Y, alpha, idx);
                dLdFc = dLdFa(:,h_num);
            else
                L = self.sosm_loss(Fa, Y, alpha, idx);
            end
            return
        end
        
        function [ L dLdH ] = sosm_loss(self, H, Y, alpha, idx)
            % Compute binomial deviance loss for each row of H, under the
            % assumption that each row of H comprises the output of multiple
            % boosted sets of hypotheses. Each set of hypotheses (i.e. a row of
            % H) is then combined via softmax into a joint hypothesis for the
            % observation from which that row of H was computed. The target
            % class of each observation is given in Y {-1,+1}.
            %
            % Parameters:
            %   H: the output of each sub hypothesis for each observation
            %   Y: the targets for sum-of-softmax outputs
            %   alpha: a smoothing factor for the softmax function
            %   idx: indices at which to evaluate loss and gradients
            %
            % Output:
            %   L: the total loss over the set of given observation outputs
            %   dLdH: loss gradient with respect to each entry of H
            %
            if ~exist('alpha','var')
                alpha = self.alpha;
            end
            % Always compute loss
            Hs = sum(exp(alpha*H),2);
            Hn = bsxfun(@rdivide, exp(alpha*H), Hs);
            F = sum(H .* Hn, 2);
            L = self.loss_func(F, Y, idx);
            % Compute gradients if they are requested
            % d L(F(H))     d L   d F
            % ---------  =  --- * ---   (i.e. chain rule of differentiation)
            %    d H        d F   d H
            if (nargout > 1)
                Hd = bsxfun(@minus, H, F);
                dFdH = Hn .* (1 + alpha*Hd);
                [L dLdF] = self.loss_func(F, Y, idx);
                dLdH = bsxfun(@times, dFdH(idx,:), dLdF);
            end
            return
        end
        
        function [ F ] = sosm_eval(self, H, alpha)
            % Compute the sum-of-softmax activation for the set of hypothesis
            % outputs in H.
            if ~exist('alpha','var')
                alpha = self.alpha;
            end
            Hs = sum(exp(alpha*H),2);
            Hn = bsxfun(@rdivide, exp(alpha*H), Hs);
            F = sum(H .* Hn, 2);
            return
        end
        
    end % END METHODS
    
end % END CLASSDEF

