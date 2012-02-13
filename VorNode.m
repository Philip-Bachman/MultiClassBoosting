classdef VorNode < handle
% VorNode is a simple class for use in a boosted voronoi tree learner.
properties
    % vor_points is a matrix, each row of which is one of the points on which
    % this node will do "voronoi" splitting
    vor_points
    % weight is the weight to give observations assigned to this node
    weight
    % has_children indicates if this node should assign weights to inputs, or if
    % it should query children nodes for these weights
    has_children
    % children is a cell array of VorNodes directly descending from this one
    children
    % sample_idx and sample_count is for use in training
    sample_idx
end

methods
    function self = VorNode(weight)
        % construct a basic VorNode object
        if exist('weight','var')
            self.weight = weight;
        else
            self.weight = 0;
        end
        self.has_children = false;
        self.children = {};
        self.sample_idx = [];
    end

    function leaf_list = get_leaf(self, X)
        % find the leaf in the subtree rooted at node to which x belongs
        leaf_list = cell(size(X,1),1);
        if self.has_children,
            c_count = length(self.children);
            [dv_min min_idx] = self.compute_dists(X);
            for c_num=1:c_count,
                c_leaves = self.children{c_num}.get_leaf(X(min_idx==c_num,:));
                leaf_list(min_idx==c_num) = c_leaves;
            end
        else
            leaf_list{1:end} = self;
        end
        return
    end
    
    function weight_list = get_weight(self, X)
        % find the weights for X in the subtree rooted at self
        weight_list = zeros(size(X,1),1);
        % compute assignment of each point in X to points in self.vor_points
        [dv_min min_idx] = self.compute_dists(X);
        if self.has_children,
            % if the present node has children, ask appropriate child for weight
            % of each observation in X
            for c_num=1:length(self.children),
                idx = find(min_idx==c_num);
                weight_list(idx) = self.children{c_num}.get_weight(X(idx,:));
            end
        else
            % if no children, then assign the weight of this node
            weight_list(:) = self.weight;
        end
        return
    end
    
    function [dv_min min_idx] = compute_dists(self, X)
        % For now, compute all distances as basic squared euclidean
        v_count = size(self.vor_points,1);
        dists = zeros(size(X,1),v_count);
        for v_num=1:v_count,
            dists(:,v_num) = sum(...
                    bsxfun(@minus, X, self.vor_points(v_num,:)).^2, 2);
        end
        [dv_min min_idx] = min(dists,[],2);
        return
    end
    
end % methods

end % classdef

    
    
