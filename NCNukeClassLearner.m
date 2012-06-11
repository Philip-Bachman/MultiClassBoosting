classdef NCNukeClassLearner < Learner
    % A simple class for learning a boosted classifier for data coming from
    % multiple classes. All (numeric) class labels should be represented in the
    % class vector Y passed to the constructor. Class labels across training
    % sets must be consistent for training to be useful. Two forms of
    % regularization are used in this learner: the first is a nuclear norm
    % penalty on the matrix of class code vectors, and the second is an L2
    % penalty on diffusion of the obsevations about their class centesrs/codes.
    %
    % Note: A smooth approximation to the nuclear norm is used, due to its
    %       non-differentiability.
    %
    %   opts.nu: shrinkage/regularization term for boosting
    %   opts.loss_func: This is the loss function applied to differences in the
    %                   distances to various class centers
    %   opts.l_const: constructor handle for base learner
    %   opts.l_count: This determines the dimensionality of the space into which
    %                 observations will be projected for classification (and
    %                 hence the number of boosted learners to train).
    %   opts.l_opts: options struct to pass to opts.l_const
    %   opts.lam_nuc: This weight controls the strength of nulcear norm
    %                 regularization of the class code words.
    %   opts.lam_dif: this weight controls penalization of diffusion of
    %                 observations about their assigned class centers
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
        % lam_nuc is a regularization weight (see above)
        lam_nuc
        % lam_dif is a regularization weight (see above)
        lam_dif
    end % END PROPERTIES
    
    methods
        
        function [ self ] = NCNukeClassLearner(X, Y, opts)
            % Simple constructor for Multiclass Boosting using projections into
            % a voronoi-decomposed space, with nuclear norm regularization
            % applied to the voronoi centers.
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
            if ~isfield(opts,'lam_nuc')
                self.lam_nuc = 1e-3;
            else
                self.lam_nuc = opts.lam_nuc;
            end
            if ~isfield(opts,'lam_dif')
                self.lam_dif = 1e-3;
            else
                self.lam_dif = opts.lam_dif;
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
            Cy = unique(Y);
            self.c_labels = Cy(:);
            self.c_codes = NCNukeClassLearner.init_codes(...
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
            %   F: matrix of vector outputs for each input in X
            %   H: matrix of distances to each class center
            %   C: class labels for each input in X
            %
            F = zeros(size(X,1),self.l_count);
            % Compute output of each base learner for each 
            for l_num=1:self.l_count,
                lrnr = self.l_objs{l_num};
                F(:,l_num) = lrnr.evaluate(X);
            end
            % Compute the class labels given the hypothesis outputs
            code_norms = diag(self.c_codes * self.c_codes');
            code_norms = reshape(code_norms, 1, numel(code_norms));
            H = repmat(code_norms, size(X,1), 1) - (2 * (F * self.c_codes'));
            [min_vals min_idx] = min(H,[],2);
            C = reshape(self.c_labels(min_idx),size(H,1),1);
            return
        end
        
        function [ L dLdFc ] = loss_wrapper(self, Fc, Fa, Y, lfun, c_num, idx)
            % Wrapper for evaluating multiclass loss and gradient, with respect
            % to outputs of a single learner.
            %
            % Parameters:
            %   Fc: the output of the hypothesis for which gradients will be
            %       produced
            %   Fa: the output of all hypotheses (the column describing the
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
            % Compute multiclass loss and gradient with respect to vector of
            % hypothesis outputs for each example represented in F/Y
            %
            % Parameters:
            %   F: vector-valued output for each of the examples in X.
            %   Y: target class for each of the observations for which F was
            %      computed. This should match labels in self.c_labels
            %   loss_func: loss function to use in computing gradients
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
            % Compute the squared distance between each vector of hypothesis
            % outputs and each class center.
            code_norms = sum(self.c_codes'.^2);
            F_norms = sum(F.^2,2);
            Fc = F * self.c_codes';
            Da = bsxfun(@plus, -2*Fc, code_norms);
            Da = bsxfun(@plus, Da, F_norms);
            % Extract the distance for each observation's target class
            Dp = sum(Da .* c_mask, 2);
            % Compute difference between distance for each observation's target
            % class and distance for each other class
            Dd = bsxfun(@minus, Da, Dp);
            Lc = zeros(size(Dd));
            uno = ones(obs_count,1);
            for c_num=1:c_count,
                lc = loss_func(Dd(:,c_num), uno, idx, 1);
                Lc(:,c_num) = lc(:);
            end
            % Zero-out the entries in Lc corresponding to the class to which
            % each observation belongs.
            Lc = Lc .* (1 - c_mask);
            L = (sum(sum(Lc)) + (self.lam_dif * sum(Dp))) / obs_count;
            % Compute gradients if they are requested
            if (nargout > 1)  
                % dLc is gradient of loss with respect to the differences
                % between distances. dLc(i,j) is the derivative of the loss
                % with respect to ||f(x)-c_j||^2 - ||f(x_i)-c_{x_i}||^2, where
                % c_{x_i} is the codeword for the class of x_i and c_j is the
                % codeword for the j'th class
                dLc = zeros(size(Dd));
                for c_num=1:c_count,
                    [lc dlc] = loss_func(Dd(:,c_num), uno, idx, 1);
                    dLc(:,c_num) = dlc(:);
                end
                dLc = dLc .* -(c_mask - 1);
                dLc_c = -sum(dLc,2);
                dLc(sub2ind(size(dLc),1:obs_count,c_idx')) = dLc_c(:);
                dGrads = zeros(size(F));
                for c_num=1:c_count,
                    dGrads(c_idx==c_num,:) = self.lam_dif * ...
                        bsxfun(@minus,F(c_idx==c_num,:),self.c_codes(c_num,:));
                end
                % dLdF is the gradient of the loss with respect to the outputs
                % of the learners from which the joint classifier is composed.
                dLdF = 2 * (dGrads - (dLc * self.c_codes));
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
            %   loss_func: loss function to use in computing gradients
            %
            % Outputs:
            %   L: average multiclass loss over the values in F and Y
            %   dLdC: gradients of L with respect to each element of F
            %
            if ~exist('loss_func','var')
                loss_func = self.loss_func;
            end
            obs_count = size(F,1);
            obs_dim = size(F,2);
            idx = 1:obs_count;
            c_count = numel(codes) / obs_dim;
            codes = reshape(codes,c_count,obs_dim);
            % Get the index into self.c_labels and self.c_codes for each class
            % membership, and put these in a binary masking matrix
            c_idx = zeros(obs_count,1);
            c_mask = zeros(obs_count,c_count);
            for c_num=1:c_count,
                c_mask(Y == self.c_labels(c_num), c_num) = 1;
                c_idx(Y == self.c_labels(c_num)) = c_num;
            end
            % Compute the squared distance between each vector of hypothesis
            % outputs and each class center.
            code_norms = sum(codes'.^2);
            F_norms = sum(F.^2,2);
            Fc = F * codes';
            Da = bsxfun(@plus, -2*Fc, code_norms);
            Da = bsxfun(@plus, Da, F_norms);
            % Extract the distance for each observation's target class
            Dp = sum(Da .* c_mask, 2);
            % Compute difference between distance for each observation's target
            % class and distance for each other class
            Dd = bsxfun(@minus, Da, Dp);
            Lc = zeros(size(Dd));
            uno = ones(obs_count,1);
            for c_num=1:c_count,
                lc = loss_func(Dd(:,c_num), uno, idx, 1);
                Lc(:,c_num) = lc(:);
            end
            % Zero-out the entries in Lc corresponding to the class to which
            % each observation belongs.
            Lc = Lc .* (1 - c_mask);
            % Compute SVD of class code matrix, and penalize sum of
            % soft-absolute values of singular values
            [U S V] = svd(codes);
            sv_idx = find(S);
            sv = S(sv_idx);
            e1 = 1e-3;
            e2 = 1e-2;
            sv_abs = sqrt(sv.^2 + e1);
            sv_sqs = sum(sv.^2 + e2);
            % Loss is sum of classification, diffusion, and nuclear-norm losses
            L_class = sum(sum(Lc)) / obs_count;
            L_dif = sum(Dp) / obs_count;
            L_nuc = sum(sv_abs) / sqrt(sv_sqs);
            L = L_class + (self.lam_dif * L_dif) + (self.lam_nuc * L_nuc);
            % Compute gradients if they are requested
            if (nargout > 1)  
                % dLc is gradient of loss with respect to the differences
                % between distances. dLc(i,j) is the derivative of the loss
                % with respect to ||f(x)-c_j||^2 - ||f(x_i)-c_{x_i}||^2, where
                % c_{x_i} is the codeword for the class of x_i and c_j is the
                % codeword for the j'th class
                dLdD = zeros(size(Dd));
                for c_num=1:c_count,
                    [lc dldd] = loss_func(Dd(:,c_num), uno, idx, 1);
                    dLdD(:,c_num) = dldd(:);
                end
                dLc = dLdD .* -(c_mask - 1);
                dLc_c = -sum(dLc,2);
                dLc(sub2ind(size(dLc),1:obs_count,c_idx')) = dLc_c(:);
                % dLdC is the gradient of the loss with respect to the code
                % elements for each class
                sqGrads = zeros(size(codes));
                dGrads = zeros(size(codes));
                for c_num=1:c_count,
                    sqGrads(c_num,:) = ...
                        sum(bsxfun(@times,codes(c_num,:),dLc(:,c_num)));
                    dGrads(c_num,:) = ...
                        sum(bsxfun(@plus,-F(c_idx==c_num,:),codes(c_num,:)));
                end
                % Compute gradients with respect to classification/diffusion
                dLdC = 2 * (sqGrads + (dGrads .* self.lam_dif) - (dLc' * F));
                % Compute gradients with respect to nuclear norm
                dS = zeros(size(S));
                for sv_num=1:numel(sv),
                    dS(sv_num,sv_num) = ...
                        (sv(sv_num) / (sv_abs(sv_num) * sqrt(sv_sqs))) - ...
                        ((sv(sv_num) * sum(sv_abs)) / sv_sqs^(3/2));
                end
                dLdS = (U * dS * V') .* self.lam_nuc;
                % Compute combined classification/diffusion/nuclear gradient
                dLdC = (dLdC(:) ./ obs_count) + dLdS(:);
            end
            return
        end
        
        function [ codes ]  = set_codewords(self, X, Y, iters)
            % Reset the current class codewords based on the observations in X
            % and Y. For now, set them to the means of each class, rescaled to
            % unit norm.
            %
            % Setup options for minFunc
            if ~exist('iters','var')
                iters = 10;
            end
            options = struct();
            options.Display = 'iter';
            options.Method = 'cg';
            options.Corr = 5;
            options.LS = 3;
            options.LS_init = 3;
            options.MaxIter = iters;
            options.MaxFunEvals = 100;
            options.TolX = 1e-10;
            % Set the loss function for code optimization
            F = self.evaluate(X);
            funObj = @( c ) self.code_loss_grad(c, F, Y, self.loss_func);
            codes = minFunc(funObj, self.c_codes(:), options);
            codes = reshape(codes,length(self.c_labels),self.l_count);
            self.c_codes = codes;
            % Display the singular values, scaled to have unit norm, so as to
            % emphasize conecntration of energy on main values
            fprintf('Normalized SVs: ');
            D = svd(codes);
            for i=1:numel(D),
                fprintf('%.4f, ',(D(i)/norm(D)));
            end
            fprintf('\n');
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
                codes = bsxfun(@rdivide, codes, sqrt(sum(codes.^2,2)).*2);
            end
            return
        end
        
    end % END METHODS (STATIC)
    
end % END CLASSDEF

