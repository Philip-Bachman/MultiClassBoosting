classdef BumpLearner < Learner
    % A simple class for boosting bumps. This is an experimental technique,
    % details of the algorithm are still being worked out. 
    %
    % Accepted options:
    %   opts.nu: shrinkage/regularization term for boosting
    %   opts.loss_func: Loss function handle to a function that can be wrapped
    %                   around hypothesis outputs F as @(F)loss_func(F,Y).
    %   opts.lambda: Regularization weight for controlling bump width
    %   opts.bump_count: The number of potential bump centers to consider
    %
    
    properties
        % bumps is a cell array of "bump" structs, each of which contains a bump
        % center, scale, covariance, and weight.
        bumps
        % possi_bumps is a cell array of structs, each of which contains info
        % pertaining to a particular "potential bump (center)".
        possi_bumps
        % lambda is a regularization weight controlling the tradeoff between
        % bump breadth and precision of gradient fitting during boosting.
        lambda
    end
    
    methods
        function [ self ] = BumpLearner(X, Y, opts)
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
            if ~isfield(opts,'lambda')
                self.lambda = 1e-4;
            else
                self.lambda = opts.lambda;
            end
            if ~isfield(opts,'bump_count')
                opts.bump_count = size(X,1);
            end
            % Compute a constant step to apply to all observations
            self.bumps = {};
            self.set_constant_bump(X, Y);
            self.set_possi_bumps(X, Y, abs(Y), opts.bump_count);
            return
        end
        
        function [ L ] = extend(self, X, Y, keep_it)
            % Extend the current set of bumps, based on the observations in X,
            % using the loss function loss_func (if supplied), and retaining the
            % changes if keep_it==1.
            if ~exist('keep_it','var')
                keep_it = 1;
            end
            F = self.evaluate(X);
            obs_count = size(X,1);
            [L dLdF] = self.loss_func(F, Y, 1:obs_count);
            % The parameters zeta and alpha are a "flattener" and a "weakener" 
            % respectively. The flattener controls the shape of the
            % regularization penalty and the weakener controls its magnitude.
            % The weakener is not really necessary, but it makes (manually)
            % finding a good lambda for the loss function easier.
            zeta = 0.5;
            alpha = 0.1;
            scales = self.possi_bumps{1}.scales;
            reg_pens = ((scales.^(-zeta)).*alpha) .* self.lambda;
            % Get a copy of dLdF, repeated for each possible sigma scale
            dLdF_ss = repmat(dLdF,1,numel(scales));
            % Find the best bump, using the observations in X as candidates
            bump = struct();
            best_err = (sum(dLdF.^2) / obs_count) + max(reg_pens);
            pb = self.possi_bumps;
            for i=1:length(self.possi_bumps),
                % Get the predictions for this bump across all scales.
                ss_preds = normpdf(...
                   bsxfun(@rdivide, pb{i}.dists, pb{i}.scales),...
                   0.0, 1.0);
                % Get least squares fit of these predictions to gradients
                %err_pens = -abs((dLdF' * ss_preds) ./ sum(ss_preds));
                weights = (dLdF' * ss_preds) ./ sum(ss_preds.^2);
                weights(isnan(weights)) = 0;
                weighted_preds = ss_preds * diag(weights);
                err_pens = sum((weighted_preds - dLdF_ss).^2) ./ obs_count;
                all_pens = err_pens + reg_pens;
                [ min_ss_err min_ss_num ] = min(all_pens);
                if (min_ss_err < best_err)
                    % If a new best-gradient-matching bump has been found, store
                    % its information
                    bump.mu = pb{i}.mu;
                    bump.dfunc = pb{i}.dfunc;
                    bump.scale = scales(min_ss_num);
                    bump.preds = ss_preds(:,min_ss_num);
                    bump.reg_pen = reg_pens(min_ss_num);
                    bump.err_pens = err_pens;
                    best_err = min_ss_err;
                end
            end
            % For the bump that best fits the gradient given the bumps from
            % which this learner is currently composed, find loss-minimizing
            % amount  of this bump to add to the current learner.
            idx = 1:size(F,1);
            step_func = @( f ) self.loss_func(f, Y, idx);
            [ w ] = self.find_step(F, bump.preds, step_func);
            bump.weight = w * self.nu;
            bump.preds = bump.preds .* bump.weight;
            bump.grad = dLdF;
            bump.grad_err = dLdF - bump.preds;
            % Evaluate the loss after adding this bump, and store bump info
            L = self.loss_func(F + bump.preds, Y, 1:obs_count);
            fprintf('    Bump scale: %.4f, bump weight: %.4f\n',...
                bump.scale, bump.weight);
            if (keep_it == 1)
                self.bumps{end + 1} = bump;
            end
            return 
        end
        
        function [ F ] = evaluate(self, X, idx)
            % Evaluate the bumps underlying this learner.
            %
            % Parameters:
            %   X: observations to evaluate (a bias column will be added)
            %   idx: indices of bumps to evaluate (optional, with the default
            %        behavior being evaluation of all splits).
            % Output:
            %   F: joint hypothesis output for each observation in X
            %
            if ~exist('idx','var')
                idx = 1:length(self.bumps);
            end
            if (idx == -1)
                idx = length(self.bumps);
            end
            obs_count = size(X,1);
            F = zeros(obs_count,1);
            for i=1:numel(idx),
                b_num = idx(i);
                b = self.bumps{b_num};
                if (b_num == 1)
                    % The first bump in self.bumps is always a constant bump,
                    % and is the only constant bump.
                    F = F + b.weight;
                else 
                    % The remaining bumps require some more computation
                    dists = b.dfunc(X);
                    F = F + (normpdf(dists./b.scale,0.0,1.0) .* b.weight);
                end
            end
            return
        end
        
        function [ L ] = set_constant_bump(self, X, Y)
            % Get a constant bump that minimizes loss for the data in X, given
            % the bumps in self.bumps{2:end} (bumps{1} is the constant bump).
            %
            % Parameters:
            %   X: training inputs
            %   Y: target classes for inputs in X
            %
            % Output:
            %   L: loss after updating the constant bump
            %
            if (length(self.bumps) > 1)
                F = self.evaluate(X);
            else
                F = zeros(size(X,1),1);
            end
            Fs = ones(size(F));
            % Get the loss-minimizing weight for a constant bump, given the
            % current predictions of the bumps in self.bumps
            idx = 1:size(Y,1);
            step_func = @( f ) self.loss_func(f, Y, idx);
            [ w ] = self.find_step(F, Fs, step_func);
            L = self.loss_func(F + Fs.*w, Y);
            % Inject the constant bump structure into self.bumps
            bump = struct();
            bump.mu = mean(X);
            bump.dfunc = @( x ) zeros(size(x,1));
            bump.scale = 0;
            bump.weight = w;
            bump.reg_pen = 0;
            bump.err_pen = L;
            bump.reg_effect = 0;
            self.bumps{1} = bump;
            return
        end
        
        function [ val ] = set_possi_bumps(self, X, Y, W, bump_count)
            % Compute info describing a number of potential bump centers, based
            % on the data in X and the importance weights for the data in X
            % given by W.
            %
            % Parameters:
            %   X: observations from which to derive possible bump centers
            %   Y: classes of observations in X
            %   W: non-negative importance weights for rows of X
            %   bump_count: number of bump centers for which to compute info
            % Outputs:
            %   NA
            %
            bump_idx = zeros(bump_count,1);
            all_idx = 1:length(W);
            all_w = W(:);
            for i=1:bump_count,
                b_idx = randsample(1:length(all_idx),1,true,all_w);
                bump_idx(i) = all_idx(b_idx);
                all_idx(b_idx) = [];
                all_w(b_idx) = [];
            end
            % Set up the scales at which to check each bump
            small_scales = []; %linspace(0.1, 1.0, 5);
            large_scales = linspace(0.5, 4.0, 25);
            scales = [small_scales large_scales];
            self.possi_bumps = cell(bump_count,1);
            for i=1:bump_count,
                b_idx = bump_idx(i);
                mu = X(b_idx,:);
                dfunc = BumpLearner.get_dfunc(X, W, mu);
                dists = dfunc(X);
                mu_info = struct();
                mu_info.mu = mu;
                mu_info.dfunc = dfunc;
                mu_info.dists = dists;
                mu_info.scales = scales;
                self.possi_bumps{i} = mu_info;
            end
            val = 0;
            return
        end
        
    end
    
    methods (Static = true)
        
        function [ dfunc ] = get_dfunc(X, W, mu)
            % Get a suitable distance function/kernelish thing for a bump
            % centered at the point mu, using info in X and W.
            %
            nn_count = 5; % Number of neighbors to include within one SD
            dists = sqrt(sum(bsxfun(@minus, X, mu).^2,2));
            [d_vals d_idx] = sort(dists,'ascend');
            scale = d_vals(nn_count);
            dfunc = @( Xd )( sqrt(sum(bsxfun(@minus, Xd, mu).^2,2)) ./ scale );
            %dim = randi(size(X,2));
            %scale = max(sqrt((X(d_idx(1:nn_count),dim) - mu(dim)).^2));
            %dfunc = @( Xd ) (sqrt((Xd(:,dim) - mu(dim)).^2) ./ scale);
            return
        end
        
        function [ sigma ] = get_sigma(X, W, mu)
            % Compute a location/scale specific covariance matrix for the row
            % in X indicated by mu_idx. Weight information in Y may be used.
            %
            % Parameters:
            %   X: input observations (obs_count x obs_dim)
            %   W: input observation weights (obs_count x 1)
            %   mu: the ean of the point for which to compute sigma
            %
            % Output:
            %   sigma: the location/scale specific covariance at the given point
            %
            obs_dim = size(X,2);
            alpha = 0.75;
            % The number of neighbors to include within one standard deviation
            % of the weighting kernel for computing locally weighted covariance.
            nn_count = max(5,round(obs_dim/4));
            % Get the distances from mu to points in X
            d_sqs = (bsxfun(@minus, X, mu)).^2;
            dists = sqrt(sum(d_sqs,2));
            [dists_val dists_idx] = sort(dists,'ascend');
            sigma = zeros(obs_dim,obs_dim);
            for dim=1:obs_dim,
                sigma(dim,dim) = max(d_sqs(dists_idx(1:nn_count),dim));
            end
            mean_var = mean(diag(sigma));
            sigma = (alpha*sigma) + ((1 - alpha)*(eye(obs_dim) .* mean_var));
            return
        end
        
        function [ sigma ] = get_sigma_class(X, Y, mu_idx)
            % Compute a location/scale specific covariance matrix for the row
            % in X indicated by mu_idx. Class information in Y may also be used.
            %
            % Parameters:
            %   X: input observations (obs_count x obs_dim)
            %   Y: input observation classes (obs_count x 1)
            %   mu_idx: index into X of the point at which to compute sigma
            %
            % Output:
            %   sigma: the location/scale specific covariance at the given point
            %
            obs_dim = size(X,2);
            % The number of neighbors to include within one standard deviation
            % of the weighting kernel for computing locally weighted covariance.
            nn_count = max(5,round(obs_dim/4));
            mu = X(mu_idx,:);
            mu_class = Y(mu_idx);
            X_same = X(Y == mu_class,:);
            % Get the distances from mu to points in its class
            d_same = sqrt(sum((X_same - repmat(mu,size(X_same,1),1)).^2,2));
            % Get the standard deviations to use for locally-weighted estimates
            mu_std = quantile(d_same, nn_count / numel(d_same));
            % Compute the weights to use for locally-weighted estimates
            mu_weights = normpdf(d_same, 0.0, mu_std);
            mu_weights = mu_weights ./ sum(mu_weights);
            % Compute the locally-weighted means for the mu and mup groups
            mu_mu = mean(X_same .* repmat(mu_weights, 1, size(X_same, 2)));
            % Compute the shared intra-group covariance
            W_same = zeros(obs_dim,obs_dim);
            for i=1:size(X_same,1),
                xs = X_same(i,:) - mu_mu;
                W_same = W_same + (xs' * xs).*mu_weights(i);
            end
            % Use a regularized form of the in-class covariance
            alpha = 0.5;
            sigma = pinv(W_same.*alpha + (diag(diag(W_same)).*(1 - alpha)));
            return
        end
        
    end
    
end

