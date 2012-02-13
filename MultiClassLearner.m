classdef MultiClassLearner < Learner
    % A simple class for learning a boosted classifier for data coming from
    % multiple classes. All (numeric) class labels should be represented in the
    % class vector Y passed to the constructor. Class labels across training
    % sets must be consistent for training to be useful. 
    %
    %   opts.nu: shrinkage/regularization term for boosting
    %   opts.loss_func: This is the loss function applied to differences between
    %                   the dot products: <f(x_i), c(x_i)> - <f(x_i), ~c(x_i)>,
    %                   where c(x_i) is the codeword for x_i's class, and
    %                   ~c(x_i) is a codeword for any other class.
    %   opts.do_opt: This indicates whether to use fast training optimization.
    %                This should only be set to 1 if all training rounds will
    %                use the same training set of observations/classes.
    %   opts.l_const: constructor handle for base learner
    %   opts.l_count: This determines the dimensionality of the space into which
    %                 observations will be projected for classification (and
    %                 hence the number of boosted learners to train).
    %   opts.l_opts: options struct to pass to opts.l_const
    %   opts.alpha: This is currently ignored. It will determine the smoothing
    %               factor for the softmax used in multiclass loss.
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
        % alpha is a scaling factor for sum-of-softmax
        alpha
        % Xt is an optional fixed training set, used if opt_train==1
        Xt
        % Ft is the current output of this learner for each row in Xt
        Ft
        % opt_train indicates if to use fast training optimization
        opt_train
    end % END PROPERTIES
    
    methods
        
        function [ self ] = MultiClassLearner(X, Y, opts)
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
            if ~isfield(opts,'alpha')
                self.alpha = 1.0;
            else
                self.alpha = opts.alpha;
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
            if ~isfield(opts,'do_opt')
                self.opt_train = 0;
                self.Xt = [];
                self.Ft = [];
            else
                self.opt_train = opts.do_opt;
                self.Xt = X;
                self.Ft = zeros(size(X,1),self.l_count);
            end
            % Find the classes present in Y and assign them codewords
            Cy = unique(Y);
            self.c_labels = Cy(:);
            self.c_codes = MultiClassLearner.init_codes(...
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
                    self.loss_wrapper(fc, Fa, yc, self.loss_func, l_num, ic);
                L = lrnr.extend(X, Y, keep_it);
                Fa(:,l_num) = lrnr.evaluate(X);
            end
            return
        end
        
        function [ F C ] = evaluate(self, X)
            % Evaluate the current set of base learners from which this meta
            % learner is composed.
            %
            % Parameters:
            %   X: the observations for which to evaluate the classifier
            %
            % Outputs:
            %   F: matrix of vector outputs for each input in X
            %   C: class labels for each input in X
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
        
        function [ L dLdFc ] = loss_wrapper(self, Fc, Fa, Y, lfun, c_num, idx)
            % Wrapper for evaluating multiclass loss and gradient, with respect
            % to outputs of a single learner.
            %
            % Parameters:
            %   Fc: the output of the hypothesis for which gradients will be
            %       produced
            %   Fa: the output of all hypothese (the column describing the
            %       hypothesis under examination will be overwritten)
            %   Y: target classes for each row/observation in F
            %   lfun: the loss function to apply to inter-class gaps
            %   c_num: the column number corresponding to the hypothesis under
            %          examination
            %   idx: indices into Fc/Fa/Y at which to evaluate loss and grads
            % Outputs:
            %   L: loss over the full set of hypotheses
            %   dLdFc: single column of gradients of loss, with respect to the
            %         hypothesis under examination
            %
            if ~exist('lfun','var')
                lfun = self.loss_func;
            end
            Fa(:,c_num) = Fc(:);
            if (nargout > 1)
                [L dLdFa] = self.compute_loss_grad(Fa, Y, lfun, idx);
                dLdFc = dLdFa(:,c_num);
            else
                L = self.compute_loss_grad(Fa, Y, lfun, idx);
            end
            return
        end
        
        function [ L dLdF ] = compute_loss_grad(self, F, Y, loss_func, idx)
            % Compute multiclass loss and gradient.
            %
            % Parameters:
            %   F: vector-valued output for each of the examples in X.
            %   Y: target class for each of the observations for which F was
            %      computed. This should match labels in self.c_labels
            %   loss_func:loss function to use in computing gradients
            %   idx: indices into F/Y at which to evaluate loss and grads
            % Outputs:
            %   L: average multiclass loss over the values in F and Y
            %   dLdF: gradients of L with respect to each element of F
            %
            if ~exist('loss_func','var')
                loss_func = self.loss_func;
            end
            if ~exist('idx','var')
                idx = 1:size(F,1);
            end
            F = F(idx,:);
            Y = Y(idx);
            idx = 1:size(F,1);
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
            % Extract the output for each observation's target class
            Fp = sum(Fc .* c_mask, 2);
            % Compute differences between output for each observation's target
            % class and output for all other classes
            Fd = bsxfun(@plus, -Fc, Fp);
            Lc = zeros(size(Fd));
            uno = ones(obs_count,1);
            for c_num=1:c_count,
                lc = loss_func(Fd(:,c_num), uno, idx, 1);
                Lc(:,c_num) = lc(:);
            end
            % Zero-out the entries in Lc and dLc corresponding to the class to
            % which each observation belongs.
            Lc = -(Lc .* (c_mask - 1));
            L = sum(sum(Lc)) / obs_count;
            % Compute gradients if they are requested
            if (nargout > 1)
                % dLc is gradient of loss with respect to the differences
                % between dot products. dLc(i,j) is the derivative of the loss
                % with respect to <f(x_i), c_{x_i}> - <f(x), c_j>, where c_{x_i}
                % is the codeword for the class of x_i and c_j is the codeword
                % for the j'th class
                dLc = zeros(size(Fd));
                for c_num=1:c_count,
                    [lc dlc] = loss_func(Fd(:,c_num), uno, idx, 1);
                    dLc(:,c_num) = dlc(:);
                end
                dLc = dLc .* -(c_mask - 1);
                dLc_c = -sum(dLc,2);
                %dLc_c = -min(dLc,[],2);
                dLc(sub2ind(size(dLc),1:obs_count,c_idx')) = dLc_c(:);
                % dLdF is the gradient of the loss with respect to the outputs
                % of the learners from which the joint classifier is composed.
                dLdF = -dLc * self.c_codes;
            end
            return
        end
        
        function [ L dLdC ] = code_loss_grad(self, codes, F, Y, loss_func)
            % Compute multiclass loss and graadient, with respect to the current
            % class codewords.
            %
            % Parameters:
            %   codes: linearized vector containing codewords for each
            %          observation represented in F and Y
            %   F: function outputs for each observation
            %   Y: target class for each observation
            %   loss_func:loss function to use in computing gradients
            %
            % Outputs:
            %   L: average multiclass loss over the values in F and Y
            %   dLdC: gradients of L with respect to each element of F
            %
            if ~exist('loss_func','var')
                loss_func = self.loss_func;
            end
            obs_count = size(F,1);
            idx = 1:obs_count;
            obs_dim = size(F,2);
            c_count = numel(codes) / obs_dim;
            codes = reshape(codes,c_count,obs_dim);
            codes_scales = sqrt(sum(codes.^2,2) + 1e-8);
            codes_normed = bsxfun(@rdivide, codes, codes_scales);
            % Get the index into self.c_labels and self.c_codes for each class
            % membership, and put these in a binary masking matrix
            c_idx = zeros(obs_count,1);
            c_mask = zeros(obs_count,c_count);
            for c_num=1:c_count,
                c_mask(Y == self.c_labels(c_num), c_num) = 1;
                c_idx(Y == self.c_labels(c_num)) = c_num;
            end
            % Compute the function outputs relative to each class codeword
            Fc = F * codes_normed';
            % Extract the output for each observation's target class
            Fp = sum(Fc .* c_mask, 2);
            % Compute differences between output for each observation's target
            % class and output for all other classes
            Fd = bsxfun(@plus, -Fc, Fp);
            Lc = zeros(size(Fd));
            uno = ones(obs_count,1);
            for c_num=1:c_count,
                lc = loss_func(Fd(:,c_num), uno, idx, 1);
                Lc(:,c_num) = lc(:);
            end
            % Zero-out the entries in Lc and dLc corresponding to the class to
            % which each observation belongs.
            Lc = -(Lc .* (c_mask - 1));
            L = sum(sum(Lc)) / obs_count;
            % Compute gradients if they are requested
            if (nargout > 1)
                % dLc is gradient of loss with respect to the differences
                % between dot products. dLc(i,j) is the derivative of the loss
                % with respect to <f(x_i), c_{x_i}> - <f(x_i), c_j>, where 
                % c_{x_i} is codeword for class of x_i and c_j is the codeword
                % for the j'th class
                dLc = zeros(size(Fd));
                for c_num=1:c_count,
                    [lc dlc] = loss_func(Fd(:,c_num), uno, idx, 1);
                    dLc(:,c_num) = dlc(:);
                end
                dLc = dLc .* -(c_mask - 1);
                dLc_c = -sum(dLc,2);
                %dLc_c = -min(dLc,[],2);
                dLc(sub2ind(size(dLc),1:obs_count,c_idx')) = dLc_c(:);
                % dLdF is the gradient of the loss with respect to the outputs
                % of the learners from which the joint classifier is composed.
                dLdC = -dLc' * F;
                dLdC = bsxfun(@rdivide, dLdC, codes_scales) -...
                       bsxfun(@times, codes_normed, sum(dLdC.*codes, 2) ./...
                       (codes_scales.^2));
                dLdC = dLdC(:);
            end
            return
        end
        
        function [ codes ]  = set_codewords(self, X, Y)
            % Reset the current class codewords based on the observations in X
            % and Y. For now, set them to the means of each class, rescaled to
            % unit norm.
            %
            % Setup options for minFunc
            options = struct();
            options.Display = 'off';
            options.Method = 'lbfgs';
            options.Corr = 5;
            options.LS = 1;
            options.LS_init = 3;
            options.MaxIter = 75;
            options.MaxFunEvals = 100;
            options.TolX = 1e-8;
            % Set the loss function for code optimization
            F = self.evaluate(X);
            funObj = @( c ) self.code_loss_grad(c, F, Y, self.loss_func);
            codes = minFunc(funObj, self.c_codes(:), options);
            codes = reshape(codes,length(self.c_labels),self.l_count);
            codes = bsxfun(@rdivide, codes, sqrt(sum(codes.^2,2)));
            self.c_codes = codes;
            return
        end
        
    end % END METHODS
    methods (Static = true)
        
        function [ step ] = find_step(F, Fs, step_func)
            % Use Matlab unconstrained optimization to find a step length that
            % minimizes: step_func(F + Fs.*step)
            options = optimset('MaxFunEvals',30,'TolX',1e-3,'TolFun',1e-3,...
                'Display','off');
            [L dL] = step_func(F);
            if (sum(Fs.*dL) > 0)
                step = fminbnd(@( s ) step_func(F + Fs.*s), -1, 0, options);
            else
                step = fminbnd(@( s ) step_func(F + Fs.*s), 0, 1, options);
            end
            return
        end
        
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

