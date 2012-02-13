classdef TreeNode < handle
% TreeNode is a simple class/structure to be used in boosted binary decision
% trees. The TreeNode class is currently suited only to use with numerical
% features and single-feature splits based on < and > comparisons.
properties
   split_feat
   split_val
   has_children
   left_child
   right_child
   sample_idx
   sample_count
   weight
end

methods
    function self = TreeNode(s_feat, s_val)
        % construct a basic TreeNode object
        if exist('s_feat','var')
            self.split_feat = s_feat;
        else
            self.split_feat = 0;
        end
        if exist('s_val','var')
            self.split_val = s_val;
        else
            self.split_val = 0;
        end
        self.has_children = false;
        self.left_child = -1;
        self.right_child = -1;
        self.sample_idx = [];
        self.sample_count = 0;
        self.weight = 0;
    end

    function leaf_list = get_leaf(self, X)
        % find the leaf in the subtree rooted at self to which x belongs
        leaf_list = cell(size(X,1),1);
        if self.has_children,
            right_idx = X(:,self.split_feat) >= self.split_val;
            right_leaves = self.right_child.get_leaf(X(right_idx,:));
            left_leaves = self.left_child.get_leaf(X(~right_idx,:));
            leaf_list(right_idx) = right_leaves;
            leaf_list(~right_idx) = left_leaves;    
        else
            leaf_list{1:end} = self;
        end
    end
    
    function weight_list = get_weight(self, X)
        % find the weight of x in the subtree rooted at self
        weight_list = zeros(size(X,1),1);
        if self.has_children,
            right_idx = X(:,self.split_feat) >= self.split_val;
            right_weights = self.right_child.get_weight(X(right_idx,:));
            left_weights = self.left_child.get_weight(X(~right_idx,:));
            weight_list(right_idx) = right_weights;
            weight_list(~right_idx) = left_weights;    
        else
            weight_list(:) = self.weight;
        end
    end
    
    function add_sample(self, s_idx)
        % add the given sample index to this self's sample index list
        self.sample_idx(self.sample_count+1) = s_idx;
        self.sample_count = self.sample_count + 1;
    end
    
end % methods

end % classdef

    
    
