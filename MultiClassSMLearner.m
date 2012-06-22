classdef MultiClassSMLearner < Learner
    % A simple class for learning a boosted classifier for data coming from
    % multiple classes. All (numeric) class labels should be represented in the
    % class vector Y passed to the constructor. Class labels across training
    % sets must be consistent for training to be useful. This learner uses the
    % standard multiclass logistic regression loss.
    %
    %   opts.nu: shrinkage/regularization term for boosting
    %   opts.l_const: constructor handle for base learner
    %   opts.l_count: This determines the dimensionality of the space into which
    %                 observations will be projected for classification (and
    %                 hence the number of boosted learners to train).
    %   opts.l_opts: options struct to pass to opts.l_const
    %   opts.lam_l1: L1 regularization weight for class codes
    %
    % Note: opts.l_opts.nu is overridden by opts.nu.
    %

    properties
        % l_objs is a cell array containing the learners from which the meta,
        % multiple classifier hypothesis (i.e. this class instance) is formed.
        l_objs
        % l_count is the number of base learners
        l_count
        % c_labels contains the labels for each class
        c_labels
        % c_codes contains the codeword for each class, with the i'th row of
        % c_codes containing the codeword currently assigned to the i'th class
        % label in c_labels
        c_codes
        % lam_l1 is a regularization weight on code words
        lam_l1
    end % END PROPERTIES
    
    methods
        
        function [ self ] = MultiClassSMLearner(X, Y, opts)
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
            if ~isfield(opts,'lam_l1')
                self.lam_l1 = 0.0;
            else
                self.lam_l1 = opts.lam_l1;
            end
            if ~isfield(opts,'l_const')
                opts.l_const = @StumpLearner;
            end
            if ~isfield(opts,'l_count')
                self.l_count = numel(unique(Y));
            else
                self.l_count = opts.l_count;
            end
            if ~isfield(opts,'l_opts')
                opts.l_opts = struct();
            end
            % Find the classes present in Y and assign them codewords
            Cy = sort(unique(Y),'ascend');
            self.c_labels = Cy(:);
            self.c_codes = MultiClassSMLearner.init_codes(...
                self.c_labels, self.l_count);
            % Create initial base learners from which the joint hypothesis will
            % be composed. We use a "dummy" Y, because all learners default to
            % setting a constant, and expect Y = {-1,+1}.
            Y = [ones(ceil(size(X,1)/2),1); -ones(floor(size(X,1)/2),1)];
            opts.l_opts.nu = self.nu;
            self.l_objs = {};
            for i=1:self.l_count,
                self.l_objs{i} = opts.l_const(X, Y, opts.l_opts);
            end
            return
        end
        
        function [ L ] = extend(self, X, Y, keep_it)
            % Extend the current set of base learners. All learners will be
            % updated, in a randomly selected order.
            %
            % Parameters:
            %   X: the observations to use in the updating process
            %   Y: the classes to which each observation in X belongs
            %   keep_it: binary indicator, selecting whteher to keep update
            %
            % Output:
            %   L: scalar-valued post-update loss
            %
            if ~exist('keep_it','var')
                keep_it = 1;
            end
            Fa = self.evaluate(X);
            l_perm = randperm(self.l_count);
            for i=1:self.l_count,
                l_num = l_perm(i);
                lrnr = self.l_objs{l_num};
                lrnr.loss_func = @( fc, yc, ic )...
                    self.loss_wrapper(fc, Fa, yc, l_num, ic);
                L = lrnr.extend(X, Y, keep_it);
                Fa(:,l_num) = lrnr.evaluate(X);
                lrnr.loss_func = [];
            end
            return
        end
        
        function [ F H C ] = evaluate(self, X)
            % Evaluate the current set of base learners from which this meta
            % learner is composed.
            %
            % Parameters:
            %   X: the observations for which to evaluate the classifier
            %
            % Outputs:
            %   F: matrix of hypothesis outputs for each input in X
            %   H: matrix of class outputs for each input in X
            %   C: inferred class labels for each input in X
            %
            F = zeros(size(X,1),self.l_count);
            % Compute output of each base learner for each 
            for l_num=1:self.l_count,
                lrnr = self.l_objs{l_num};
                F(:,l_num) = lrnr.evaluate(X);
            end
            % Compute the class labels given the hypothesis outputs
            H = F * self.c_codes';
            [max_vals max_idx] = max(H,[],2);
            C = reshape(self.c_labels(max_idx),size(H,1),1);
            return
        end
        
        function [ M ] = margins(self, X, Y)
            % Compute margins for the observations in X, given the classes in Y
            %
            obs_count = size(X,1);
            c_count = numel(self.c_labels);
            F = self.evaluate(X);
            % Get the index into self.c_labels and self.c_codes for each class
            % membership, and put these in a binary masking matrix
            c_idx = zeros(obs_count,1);
            c_mask = zeros(obs_count,c_count);
            for c_num=1:c_count,
                c_mask(Y == self.c_labels(c_num), c_num) = 1;
                c_idx(Y == self.c_labels(c_num)) = c_num;
            end
            H = F * self.c_codes';
            M = sum(H .* c_mask,2) - max(H .* (1 - c_mask),[],2);
            return
        end
        
        function [ L dLdFc ] = loss_wrapper(self, Fc, Fa, Y, c_num, idx)
            % Wrapper for evaluating multiclass loss and gradient, with respect
            % to outputs of a single learner.
            %
            % Parameters:
            %   Fc: the output of the hypothesis for which gradients will be
            %       produced
            %   Fa: the output of all hypothese (the column describing the
            %       hypothesis under examination will be overwritten)
            %   Y: target classes for each row/observation in F
            %   c_num: the column number corresponding to the hypothesis under
            %          examination
            %   idx: indices into Fc/Fa/Y at which to evaluate loss and grads
            % Outputs:
            %   L: loss over the full set of hypotheses
            %   dLdFc: single column of gradients of loss, with respect to the
            %         hypothesis under examination
            %
            Fa(:,c_num) = Fc(:);
            if (nargout > 1)
                [L dLdFa] = self.compute_loss_grad(Fa, Y, idx);
                dLdFc = dLdFa(:,c_num);
            else
                L = self.compute_loss_grad(Fa, Y, idx);
            end
            return
        end
        
        function [ L dLdF ] = compute_loss_grad(self, F, Y, idx)
            % Compute multiclass loss and gradient.
            %
            % Parameters:
            %   F: vector-valued output for each of the examples in X.
            %   Y: target class for each of the observations for which F was
            %      computed. This should match labels in self.c_labels
            %   idx: indices into F/Y at which to evaluate loss and grads
            % Outputs:
            %   L: average multiclass loss over the values in F and Y
            %   dLdF: gradients of L with respect to each element of F
            %
            if ~exist('idx','var')
                idx = 1:size(F,1);
            end
            F = F(idx,:);
            Y = Y(idx);
            obs_count = size(F,1);
            c_count = length(self.c_labels);
            % Get the index into self.c_labels and self.c_codes for each class
            % membership, and put these in a binary masking matrix
            c_idx = zeros(obs_count,1);
            c_mask = zeros(obs_count,c_count);
            for c_num=1:c_count,
                c_mask(Y == self.c_labels(c_num), c_num) = 1;
                c_idx(Y == self.c_labels(c_num)) = c_num;
            end
            % Compute the function outputs relative to each class codeword
            Fc = F * self.c_codes';
            % Compute softmax probability estimates
            Fn = bsxfun(@rdivide, exp(Fc), sum(exp(Fc),2));
            % Compute loss, i.e. log-likelihood of observations
            L = -sum(sum(log(Fn) .* c_mask));
            % Compute gradients if they are requested
            if (nargout > 1)
                % dLc is gradient of loss with respect to the output for each
                % class codeword
                dLc = -(c_mask - Fn);
                % dLdF is the gradient of the loss with respect to the outputs
                % of the learners from which the joint classifier is composed.
                dLdF = dLc * self.c_codes;
            end
            return
        end
        
        function [ L dLdC ] = code_loss_grad(self, codes, F, Y)
            % Compute multiclass loss and graadient, with respect to the current
            % class codewords.
            %
            % Parameters:
            %   codes: linearized vector containing codewords for each
            %          observation represented in F and Y
            %   F: function outputs for each observation
            %   Y: target class for each observation
            %
            % Outputs:
            %   L: average multiclass loss over the values in F and Y
            %   dLdC: gradients of L with respect to each element of F
            %
            obs_count = size(F,1);
            obs_dim = size(F,2);
            c_count = numel(codes) / obs_dim;
            codes = reshape(codes,c_count,obs_dim);
            codes_scales = sqrt(sum(codes.^2,2) + 1e-8);
            codes_normed = bsxfun(@rdivide, codes, codes_scales);
            codes_nabs = sqrt(codes_normed.^2 + 1e-6);
            % Get the index into self.c_labels and self.c_codes for each class
            % membership, and put these in a binary masking matrix
            c_idx = zeros(obs_count,1);
            c_mask = zeros(obs_count,c_count);
            for c_num=1:c_count,
                c_mask(Y == self.c_labels(c_num), c_num) = 1;
                c_idx(Y == self.c_labels(c_num)) = c_num;
            end
            % Compute the function outputs relative to each class codeword
            Fc = F * self.c_codes';
            % Compute softmax probability estimates
            Fn = bsxfun(@rdivide, exp(Fc), sum(exp(Fc),2));
            % Compute loss, i.e. log-likelihood of observations
            L = (-sum(sum(log(Fn) .* c_mask)) / obs_count) + ...
                (self.lam_l1 / numel(codes)) * sum(sum(codes_nabs));
            % Compute gradients if they are requested
            if (nargout > 1)
                % dLc is gradient of loss with respect to the output for each
                % class codeword
                dLc = c_mask - Fn;
                % dLdC is the gradient of the loss with respect to the elements
                % of the codewords for each class
                dLdC = ((dLc' * F) ./ obs_count) + ...
                    (self.lam_l1 / numel(codes)) * (codes_normed ./ codes_nabs);
                dLdC = bsxfun(@rdivide, dLdC, codes_scales) -...
                       bsxfun(@times, codes_normed, sum(dLdC.*codes, 2) ./...
                       (codes_scales.^2));
                dLdC = dLdC(:);
            end
            return
        end
        
        function [ codes ]  = set_codewords(self, X, Y, rounds)
            % Reset the current class codewords based on the observations in X
            % and Y. For now, set them to the means of each class, rescaled to
            % unit norm.
            %
            if ~exist('rounds','var')
                rounds = 10;
            end
            % Setup options for minFunc
            options = struct();
            options.Display = 'off';
            options.Method = 'cg';
            options.Corr = 5;
            options.LS = 0;
            options.LS_init = 3;
            options.MaxIter = rounds;
            options.MaxFunEvals = 75;
            options.TolX = 1e-8;
            % Set the loss function for code optimization
            F = self.evaluate(X);
            funObj = @( c ) self.code_loss_grad(c, F, Y);
            codes = minFunc(funObj, self.c_codes(:), options);
            codes = reshape(codes,length(self.c_labels),self.l_count);
            codes = bsxfun(@rdivide, codes, sqrt(sum(codes.^2,2)));
            self.c_codes = codes;
            return
        end
        
    end % END METHODS
    methods (Static = true)
        
        function [ codes ] = init_codes(c_labels, l_count)
            % Generate an initial set of codewords for the classes whose labels
            % are contained in c_labels. The label really just determines the
            % number of codes to generate.
            if (length(c_labels) == l_count)
                codes = eye(l_count);
            else
                codes = randn(length(c_labels),l_count);
                codes = bsxfun(@rdivide, codes, sqrt(sum(codes.^2,2)));
            end
            return
        end
        
    end % END METHODS (STATIC)
    
end % END CLASSDEF

