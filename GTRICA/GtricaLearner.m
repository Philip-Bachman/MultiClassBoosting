classdef GtricaLearner < Learner
    % A simple class for learning features adapted to a dataset.
    %

    properties
        % alpha is a smoothing factor for group normalization
        alpha
        % use_sosm determines whether to use sum-of-softmax group normalization
        % or polynomial group normalization
        use_sosm
        % obs_dim is the dimension of observations for which bases are learned
        obs_dim
        % group_size is the number of features in each normalization group
        group_size
        % group_count is the number of normalization groups to learn
        group_count
        % basis_count is the total number of bases (group_size * group_count)
        basis_count
        % l_logreg is the logistic regression l2-regularization weight
        l_logreg
        % l_class is the classification regularization weight
        l_class
        % l_smooth is the basis Tikhonov regularization weight
        l_smooth
        % l_spars is the basis response sparsity regularization weight
        l_spars
        % ab_iters is the number of minFunc iterations to allow when doing a
        % joint update of self.A and self.b
        ab_iters
        % A is a matrix whose rows are the learned bases
        A
        % b is the vector of regression coefficients
        b
        % C is a Tikhonov regularization matrix, for regularizing rows of A
        C
    end % END PROPERTIES
    
    methods
        function [ self ] = GtricaLearner(X, Y, opts)
            % Simple constructor for Multiple Classifier boosting via
            % sum-of-softmax.
            %
            if ~exist('opts','var')
                opts = struct();
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
            if ~isfield(opts,'group_size')
                self.group_size = 5;
            else
                self.group_size = opts.group_size;
            end
            if ~isfield(opts,'group_count')
                self.group_count = 5;
            else
                self.group_count = opts.group_count;
            end
            if ~isfield(opts,'l_logreg')
                self.l_logreg = 1e-4;
            else
                self.l_logreg = opts.l_logreg;
            end
            if ~isfield(opts,'l_class')
                self.l_class = 1e1;
            else
                self.l_class = opts.l_class;
            end
            if ~isfield(opts,'l_smooth')
                self.l_smooth = 1e-5;
            else
                self.l_smooth = opts.l_smooth;
            end
            if ~isfield(opts,'l_spars')
                self.l_spars = 0.0;
            else
                self.l_spars = opts.l_spars;
            end
            if ~isfield(opts,'ab_iters')
                self.ab_iters = 10;
            else
                self.ab_iters = opts.ab_iters;
            end
            self.nu = 1.0; % vestigial boosting property
            self.basis_count = self.group_size * self.group_count;
            self.l_logreg = self.l_logreg / self.basis_count;
            self.obs_dim = size(X,2);
            self.use_sosm = 1;
            % Compute Tikhonov regularization matrix, using lightly regularized
            % forms of the empirical covariance and precision matrices
            a = 1 - 1e-2;
            Cc = cov(X);
            Cc = (a * Cc) + ((1 - a) * (eye(self.obs_dim).*mean(diag(Cc))));
            self.C = pinv(Cc);
            % Initialize bases using the regularized empirical distribution
            self.A = mvnrnd(zeros(self.basis_count,self.obs_dim), Cc,...
                self.basis_count);
            self.A = GtricaLearner.l2row(self.A);
            % Initialize regression coefficients to zeros
            self.b = zeros(self.basis_count, 1);
            return
        end
        
        function [ L ] = extend(self, X, Y, keep_it)
            % Update the bases in self.A and coefficients in self.b using the
            % observations in X/Y. self.loss_func should be set prior to calling
            % this method such that loss and gradient of loss can be computed
            % like: [L dL] = self.loss_func(self.compute_features(self.A,X))
            %
            % Parameters:
            %   X: observations of size (obs_count x self.obs_dim)
            %   Y: target outputs, passable to self.loss_func
            % Outputs:
            %   L: loss over the observations in X/Y, post-update
            %
            % setup options for minFunc update of self.b
            options = struct();
            options.Display = 'final';
            options.Method = 'lbfgs';
            options.Corr = 5;
            options.LS = 1;
            options.LS_init = 3;
            options.MaxIter = 40;
            options.MaxFunEvals = 80;
            options.TolX = 1e-6;            
            % compute scalar/vector-valued feature activations
            [ Fs Fv ] = self.evaluate(X); 
            % set loss/gradient function for updating self.b
            funObj = @( w ) self.loss_wrapper_b(w, Fv, Y);
            self.b = minFunc(funObj, self.b, options);
            % setup options for minFunc joint update of self.A and self.b
            options = struct();
            options.Display = 'iter';
            options.Method = 'lbfgs';
            options.Corr = 5;
            options.LS = 1;
            options.LS_init = 3;
            options.TolX = 1e-6;
            options.MaxIter = self.ab_iters;
            options.MaxFunEvals = self.ab_iters*2;
            % set loss/gradient function for joint update of self.A and self.b
            funObj = @( w ) self.loss_wrapper_Ab(w, X, Y);
            Ab = [reshape(self.A, self.basis_count*self.obs_dim, 1);...
                reshape(self.b, self.basis_count, 1)];
            Ab = minFunc(funObj, Ab, options);
            self.A = reshape(Ab(1:self.basis_count*self.obs_dim),...
                self.basis_count, self.obs_dim);
            self.A = GtricaLearner.l2row(self.A);
            self.b = reshape(Ab(self.basis_count*self.obs_dim+1:end),...
                self.basis_count, 1);
            % compute post-update loss given X, Y, self.A, and self.b
            [ Fs Fv ] = self.evaluate(X);
            L = self.loss_func(Fs, Y, 1:size(Y,1));
            return
        end
        
        function [ L dLdAb ] = loss_wrapper_Ab(self, Ab, X, Y)
            % Wrap self.loss_func in a manner allowing minFunc optimization of
            % both self.A and self.b, given the observations in X and Y
            %
            % Parameters:
            %   Ab: [A(:); b(:)] (i.e. linearized bases and coefficients)
            %   X: observations with which to compute feature responses, etc
            %   Y: target classes for each obsrvation in X
            % Outputs:
            %   L: loss given the features/coeffs in Ab, using the observations
            %      in X and targets in Y, according to self.loss_func
            %   dLdAb: gradient of L with respect to elements of Ab
            %
            obs_count = size(X,1);
            A_t = Ab(1:self.basis_count*self.obs_dim);
            A_t = reshape(A_t, self.basis_count, self.obs_dim);
            b_t = Ab(self.basis_count*self.obs_dim+1:end);
            b_t = reshape(b_t, self.basis_count, 1);
            % Compute feature responses and auxiliary info
            [ F parts ] = self.compute_features(A_t, X);
            % Compute loss and gradient of loss with respect to scalar-valued
            % output for each observation
            [L dL] = self.loss_func(F*b_t, Y, 1:obs_count);
            % Compute sparsity, classification, regression regularization, and
            % smoothing objectives, then sum them to form combined objective
            obj_sp = sum(sum(F)) / (obs_count * self.group_count);
            obj_cl = L * self.l_class;
            obj_re = (self.l_logreg / 2) * sum(b_t.^2);
            obj_sm = trace(parts.An * self.C * parts.An') * ...
                (self.l_smooth / self.basis_count);
            L = obj_sp + obj_cl + obj_re + obj_sm;
            fprintf('sp=%.4f, cl=%.4f, re=%.4f, sm=%.4f\n',...
                    obj_sp,obj_cl,obj_re,obj_sm);
            if (nargout > 1)
                % Backpropagate class loss gradient through activation process
                dLdF = bsxfun(@times, b_t', dL);
                dLdA = self.backprop_grads(dLdF, X, parts);
                dLdA = dLdA .* (self.l_class / obs_count);
                % Backpropagate sparsity gradient through activation process
                dSdA = self.backprop_grads(ones(size(F)), X, parts);
                dSdA = dSdA ./ (obs_count * self.group_count);
                % compute gradients for Tikhonov regularization of filters
                dTdA = GtricaLearner.l2grad(parts.A, parts.An,...
                    parts.An_scales, (parts.An*self.C' + parts.An*self.C));
                dTdA = dTdA .* (self.l_smooth / self.basis_count);
                % Mash all the gradients together
                dA = dLdA + dSdA + dTdA;
                % compute gradients for self.b
                dB = sum(bsxfun(@times, F, dL));
                dB = (dB' .* (self.l_class/obs_count)) + (self.l_logreg * b_t);
                % package the "linearized" joint gradient for A and b
                dLdAb = [reshape(dA, self.basis_count*self.obs_dim, 1);...
                    reshape(dB, self.basis_count, 1)];
            end
            return
        end
        
        function [ L dLdB ] = loss_wrapper_b(self, w, F, Y)
            % Wrap self.loss_func in a manner allowing minFunc optimization of
            % self.b, given the feature responses in F, for the targets in Y
            %
            obs_count = size(F,1);
            if (nargout > 1)
                [ L dL ] = self.loss_func(F*w, Y, 1:obs_count);
                L = (L * self.l_class) + ((self.l_logreg / 2) * sum(w.^2));
                dLdB = sum(bsxfun(@times, F, dL));
                dLdB = (dLdB' ./ size(F,1)) .* self.l_class;
                dLdB =  dLdB + (self.l_logreg * w);
            else
                L = self.loss_func(F*w, Y, 1:obs_count);
                L = (L * self.l_class) + ((self.l_logreg / 2) * sum(w.^2));
            end
            return
        end
        
        function [ F Fv ] = evaluate(self, X)
            % Evaluate the vector-valued output of the current feature set for
            % the observations in X, and then use b to convert to scalar-valued
            % outputs.
            %
            % Parameters:
            %   X: observations for which to compute the feature outputs
            % Outputs:
            %   F: scalar-valued output for each input observation
            %   Fv: vector-valued output (i.e. output for each feature)
            %
            Fv = self.compute_features(self.A, X);
            F = Fv * self.b;
            return
        end
        
        function [ F parts ] = compute_features(self, A, X)
            % Evaluate the vector-valued output of the current feature set for
            % the observations in X.
            %
            % Parameters:
            %   A: matrix of bases to use in feature computation
            %   X: observations for which to compute the feature outputs
            % Outputs:
            %   F: vector-valued output (i.e. output for each feature)
            %   parts: information needed for gradients, etc.
            %
            if (self.use_sosm == 1)
                [F parts] = self.features_sosm(A, X);
            else
                [F parts] = self.features_poly(A, X);
            end
            return
        end
        
        function [ dLdA ] = backprop_grads(self, dLdF, X, parts)
            % Backpropagate gradients at each feature output to the bases.
            %
            % Parameters:
            %   dLdF: gradients of some loss function with respect to the
            %         response of each feature in self.A to each obs in X
            %   X: observations producing the gradients in dLdF
            %   parts: info about feature computation
            % Outputs:
            %   dLdA: gradients of the loss function with respect to each
            %         element of each basis in self.A
            %
            if (self.use_sosm == 1)
                dLdA = self.grads_sosm(dLdF, X, parts);
            else
                dLdA = self.grads_poly(dLdF, X, parts);
            end
            return
        end

        function [ F parts ] = features_sosm(self, A, X)
            % Compute sosm features for X, given the bases in A
            % Group the filters for group soft-max normalizationv
            groups = false(self.group_count,self.basis_count);
            for g=1:self.group_count,
                group_start = ((g - 1) * self.group_size) + 1;
                group_end = ((g - 1) * self.group_size) + self.group_size;
                groups(g,group_start:group_end) = true;
            end
            % e controls smoothing during soft-absolute activation
            e = 1e-8; 
            % Feed Forward
            [An, An_scales] = GtricaLearner.l2row(A); % Row normalize A
            Fl = (An*X')'; % Linear Activation
            Fsa = sqrt(Fl.^2 + e); % Soft absolute activation
            Fsaea = exp(Fsa .* self.alpha); % Exponential of soft absolute
            Fsasm = zeros(size(Fsaea)); % Soft-max activations
            for g=1:self.group_count,
                group = groups(g,:);
                Fsasm(:,group) = bsxfun(@rdivide, Fsaea(:,group),...
                    sum(Fsaea(:,group),2));
            end
            Fsasmsa = Fsa .* Fsasm; % Scaled soft-max activations
            % Package partial activations for use in backpropagation etc.
            F = Fsasmsa;
            parts = struct();
            parts.A = A;
            parts.An = An;
            parts.An_scales = An_scales;
            parts.F = F;
            parts.Fl = Fl;
            parts.Fsa = Fsa;
            parts.Fsasm = Fsasm;
            parts.groups = groups;
            return
        end

        function [ dLdA ] = grads_sosm(self, dLdF, X, parts)
            % Backpropagate the gradient in dLdF through the feature activation
            % computation in self.features_sosm().
            Fsasm = parts.Fsasm;
            Fsa = parts.Fsa;
            Fl = parts.Fl;
            F = parts.F;
            % Gradient with respect to the soft abosolute activation
            %    dL    dL    dF
            %   ---- = -- * ----
            %   dFsa   dF   dFsa
            dLdFsa = dLdF .* (Fsasm + (self.alpha * F));
            dLdFsa_1 = dLdF .* F;
            for g=1:self.group_count,
                group = parts.groups(g,:);
                dLdFsa_2 = bsxfun(@times, Fsasm(:,group),...
                    sum(dLdFsa_1(:,group),2));
                dLdFsa(:,group) = dLdFsa(:,group) - (self.alpha * dLdFsa_2);
            end
            % Gradient with respect to linear activations
            %  dL     dL    dFsa
            %  --- = ---- * ----
            %  dFl   dFsa   dFl
            dLdFl = dLdFsa .* (Fl ./ Fsa);
            % Backprop dLdFl through observations
            dLdA = dLdFl' * X;
            % Backprop through per-basis normalization
            dLdA = GtricaLearner.l2grad(...
                parts.A, parts.An, parts.An_scales, dLdA);
            return
        end

        function [ F parts ] = features_poly(self, A, X)
            % Compute features for the observations in X, using the bases in
            % self.A, and polynomial-based group normalization
            % Group the filters for group normalization
            groups = false(self.group_count,self.basis_count);
            for g=1:self.group_count,
                group_start = ((g - 1) * self.group_size) + 1;
                group_end = ((g - 1) * self.group_size) + self.group_size;
                groups(g,group_start:group_end) = true;
            end
            % e controls smoothing during soft-absolute activation
            e = 1e-8;
            % Feed Forward
            [An, An_scales] = GtricaLearner.l2row(A); % Row normalize A
            Fl = (An*X')'; % Linear Activation
            Fsa = sqrt(Fl.^2 + e); % Soft absolute activation
            F = zeros(size(Fsa)); % Group normalized soft absolute activations
            Gn = zeros(size(Fsa,1),self.group_count); % Normalization factors
            for g=1:self.group_count,
                % For each normalization group, compute and store the group
                %  normalization factor and normalized filter responses
                group = groups(g,:);
                Gn(:,g) = sum(Fsa(:,group).^self.alpha,2) + 1e-8;
                F(:,group) = bsxfun(@rdivide,...
                    Fsa(:,group).^(self.alpha+1), Gn(:,g));
            end
            % Package partial activations for use in backpropagation etc.
            parts = struct();
            parts.A = A;
            parts.An = An;
            parts.An_scales = An_scales;
            parts.Fl = Fl;
            parts.Fsa = Fsa;
            parts.F = F;
            parts.Gn = Gn;
            parts.groups = groups;
            return
        end

        function [ dLdA ] = grads_poly(self, dLdF, X, parts)
            % Backpropagate the gradient in dL through the feature activation
            % described by self.features_poly()
            F = parts.F;
            Fsa = parts.Fsa;
            Fl = parts.Fl;
            Gn = parts.Gn;
            % Gradient with respect to the soft abosolute activation
            %    dL     dL     dFsan
            %   ---- = ----- * -----
            %   dFsa   dFsan   dFsa
            dLdFsa = zeros(size(F));
            for g=1:self.group_count,
                group = parts.groups(g,:);
                dLdG_1 = dLdF(:,group) .* ...
                    (((self.alpha + 1) * F(:,group)) ./ Fsa(:,group));
                dLdG_2 = bsxfun(@times,...
                    (self.alpha * Fsa(:,group).^(self.alpha-1)),...
                    sum(bsxfun(@rdivide, (dLdF(:,group) .* F(:,group)),...
                    Gn(:,g)), 2));
                dLdFsa(:,group) = dLdG_1 - dLdG_2;
            end
            % Gradient with respect to linear activations
            %  dL     dL    dFsa
            %  --- = ---- * ----
            %  dFl   dFsa   dFl
            dLdFl = dLdFsa .* (Fl ./ Fsa);
            % Backprop dLdFl through observations
            dLdA = dLdFl' * X;
            % Backprop through per-basis normalization
            dLdA = GtricaLearner.l2grad(...
                parts.A, parts.An, parts.An_scales, dLdA);
            return
        end
    end % END INSTANCE METHODS
    
    methods (Static = true)
        function [Y,N] = l2row(X)
            % Set rows in matrix X to (almost) unit norm
            N = sqrt(sum(X.^2,2) + 1e-8);
            Y = bsxfun(@rdivide,X,N);
            return
        end
        function [G] = l2grad(X, Y, N, D)
            % Backpropagate through normalization for unit norm
            G = bsxfun(@rdivide, D, N) -...
                bsxfun(@times, Y, sum(D.*X, 2) ./ (N.^2));
            return
        end
    end % END STATIC METHODS
    
end % END CLASSDEF

