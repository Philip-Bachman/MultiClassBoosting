classdef InfluenceLearner < Learner
    % A simple class for performing multiple classifier boosting. The multiple
    % classifier boosting implementation in this class uses boosted pairs of
    % hypothesis/influence functions. In each pair, the hypothesis computes a
    % proposed classifier output, while the influence function controls the
    % relative influence of this proposed output on the joint classifier.
    %
    % The influence functions in this implementation can be interpreted as
    % estimating a distribution over the relative merits of each sub hypothesis
    % using multinomial logistic regression (i.e. softmax).
    %
    % Accepted options:
    %   opts.nu: shrinkage/regularization term for boosting
    %   opts.loss_func: Loss function handle to a function that can be wrapped
    %                   around hypothesis outputs F as @(F)loss_func(F,Y).
    %   opts.extend_all: if this is set, all hypotheses will be updated on each
    %                    cycle, in a random order
    %   opts.l_const: constructor handle for base learners
    %   opts.l_count: the number of hypothesis/influence pairs to learn
    %   opts.l_opts: options struct to pass to opts.l_const
    %
    % Note: opts.l_opts.nu is overridden by opts.nu.
    %

    properties
        % h_objs is a cell array containing the hypothesis learner for each
        % hypothesis/influence learner pair
        h_objs
        % g_objs is a cell array containing the influence learner for each
        % hypothesis/influence learner pair
        g_objs
        % l_count is the number of hypothesis/influence pairs
        l_count
        % extend_all determines whether to extend all or only best
        extend_all
    end % END PROPERTIES
    
    methods
        
        function [ self ] = InfluenceLearner(X, Y, opts)
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
                self.extend_all = 1;
            else
                self.extend_all = opts.extend_all;
            end
            % Create initial base learners from which the joint hypothesis will
            % be composed.
            opts.l_opts.nu = self.nu;
            self.h_objs = {};
            self.g_objs = {};
            for i=1:self.l_count,
                self.h_objs{i} = opts.l_const(X, Y, opts.l_opts);
                self.g_objs{i} = opts.l_const(...
                    zeros(2,size(X,2)), [1; -1], opts.l_opts);
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
            [F Ha Ga] = self.evaluate(X);
            for i=1:self.l_count,
                l_num = l_order(i);
                % Get the hypothesis to update and set its loss/grad function
                lrnr = self.h_objs{l_num};
                lrnr.loss_func = @( Hc, Yc, idx ) self.hypo_wrapper(...
                    Hc, Ha, Ga, Yc, l_num, idx);
                % Update the hypothesis
                lrnr.extend(X, Y, 1);
                % Update set of hypothesis values to account for lrnr update
                Ha(:,l_num) = lrnr.evaluate(X);
                % Get the influence to update and set its loss/grad function
                lrnr = self.g_objs{l_num};
                lrnr.loss_func = @( Gc, Yc, idx ) self.infl_wrapper(...
                    Gc, Ha, Ga, Yc, l_num, idx);
                % Update the influence
                L = lrnr.extend(X, Y, 1);
                % Update set of influence values to account for lrnr update
                Ga(:,l_num) = lrnr.evaluate(X);
            end
            return
        end
        
        function [ F H G ] = evaluate(self, X, idx)
            % Evaluate the current set of base learners from which this meta
            % learner is composed.
            if ~exist('idx','var')
                idx = 1;
            else
                if (idx == -1)
                    error('-1 eval index not for InfluenceLearner\n');
                end
            end
            H = zeros(size(X,1),self.l_count);
            G = zeros(size(X,1),self.l_count);
            % Compute output of each hypothesis/influence learner pair
            for l_num=1:self.l_count,
                h = self.h_objs{l_num};
                g = self.g_objs{l_num};
                H(:,l_num) = h.evaluate(X);
                G(:,l_num) = g.evaluate(X);
            end
            % Compute the influence-weighted sum of hypothesis outputs
            Gn = bsxfun(@rdivide, exp(G), sum(exp(G),2));
            F = sum(H .* Gn, 2);
            return
        end
        
        function [ L dLdHc ] = hypo_wrapper(self, Hc, Ha, Ga, Y, h_num, idx)
            % Create a wrapper function that computes loss and gradient of loss
            % with respect to a single column of H.
            %
            % Parameters:
            %   Hc: the output of hypothesis under consideration
            %   Ha: the output of all hypotheses, including column h_num, which
            %       will be overwritten by Hc
            %   Ga: the output of all influences
            %   Y: target classes for each observation
            %   h_num: column index of hypothesis with respect to which to
            %          compute loss and gradient of loss
            %   idx: indices at which to evaluate loss and gradients
            % Output:
            %   L: loss of the full joint hypothesis
            %   dLdFc: gradient of L with respect to column h_num of Fa.
            %
            Ha(:,h_num) = Hc(:);
            if (nargout > 1)
                [L dLdHa] = self.hypo_loss(Ha, Ga, Y, idx);
                dLdHc = dLdHa(:,h_num);
            else
                L = self.hypo_loss(Ha, Ga, Y, idx);
            end
            return
        end
        
        function [ L dLdGc ] = infl_wrapper(self, Gc, Ha, Ga, Y, g_num, idx)
            % Create a wrapper function that computes loss and gradient of loss
            % with respect to a single column of G.
            %
            % Parameters:
            %   Gc: the output of influence under consideration
            %   Ha: the output of all hypotheses, including column h_num, which
            %       will be overwritten by Hc
            %   Ga: the output of all influences
            %   Y: target classes for each observation
            %   g_num: column index of influence with respect to which to
            %          compute loss and gradient of loss
            %   idx: indices at which to evaluate loss and gradients
            % Output:
            %   L: loss of the full joint hypothesis
            %   dLdFc: gradient of L with respect to column h_num of Fa.
            %
            Ga(:,g_num) = Gc(:);
            if (nargout > 1)
                [L dLdGa] = self.infl_loss(Ha, Ga, Y, idx);
                dLdGc = dLdGa(:,g_num);
            else
                L = self.infl_loss(Ha, Ga, Y, idx);
            end
            return
        end
        
        function [ L dLdH ] = hypo_loss(self, H, G, Y, idx)
            % Compute loss with respect to the current set of hypothesis
            % functions and compute partial gradients of this loss with respect
            % to the output of each hypothesis function.
            %
            % Parameters:
            %   H: the output of each sub hypothesis for each observation
            %   G: the output of each sub influence for each observation
            %   Y: the targets outputs
            %   idx: indices at which to evaluate loss and gradients
            %
            % Output:
            %   L: the total loss over the set of given observation outputs
            %   dLdH: loss gradient with respect to each entry of H
            %
            Gn = bsxfun(@rdivide, exp(G), sum(exp(G),2));
            F = sum(H .* Gn, 2);
            if (nargout == 1)
                L = self.loss_func(F, Y, idx);
            else
                % Compute gradients if they are requested
                % d L(F(H,G))     d L   d F
                % -----------  =  --- * ---   (i.e. derivative chain rule)
                %     d H         d F   d H
                [L dLdF] = self.loss_func(F, Y, idx);
                dLdH = bsxfun(@times, Gn(idx,:), dLdF);
            end
            return
        end
        
        function [ L dLdG ] = infl_loss(self, H, G, Y, idx)
            % Compute loss with respect to the current set of influence
            % functions and compute partial gradients of this loss with respect
            % to the output of each influence function.
            %
            % Parameters:
            %   H: the output of each sub hypothesis for each observation
            %   G: the output of each sub influence for each observation
            %   Y: the targets outputs
            %   idx: indices at which to evaluate loss and gradients
            %
            % Output:
            %   L: the total loss over the set of given observation outputs
            %   dLdG: loss gradient with respect to each entry of G
            %
            Gn = bsxfun(@rdivide, exp(G), sum(exp(G),2));
            F = sum(H .* Gn, 2);
            if (nargout == 1)
                L = self.loss_func(F, Y, idx);
            else
                % Compute gradients if they are requested
                % d L(F(H,G))     d L   d F
                % -----------  =  --- * ---   (i.e. derivative chain rule)
                %     d G         d F   d G
                [L dLdF] = self.loss_func(F, Y, idx);
                dFdG = (Gn .* H) - bsxfun(@times, Gn, F);
                dLdG = bsxfun(@times, dFdG(idx,:), dLdF);
            end
            return
        end
        
    end % END METHODS
    
end % END CLASSDEF