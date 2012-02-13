classdef Learner < handle
    % Abstract class/interface for implementing incrementally extensible
    % hypotheses that perform functional gradient descent (a.k.a. boosting).
    %
    % Note: constructors for each type of Learner take as input a set of
    % observations, a set of target classes, and an options structure. Learners 
    % vary in how they use these, and what options they expect to be present.
    % For class-specific help, see the class files (available via help).
    %
    
    properties
        % nu is a shrinkage factor for this learner, controlling the rate at
        % which it seeks to minimize the loss (regularization, effectively)
        nu
        % loss_func is the loss function (a function of the outputs of the
        % learner at a set of observations and the target values for those
        % observations) that will be used to guide learning.
        loss_func
    end
    
    methods (Abstract)
        % The extend function for a learner takes as input a set of observations
        % and a set of target classes (or, generally, scalar values), towards
        % which learning will be guided. The parameter "keep_it" determines if
        % the computed hypothesis extension will be retained or discarded.
        result = extend(observations, classes, keep_it)
        % The evaluate function returns the current output of the learner for
        % each of the passed observations. The multiple class classfier returns
        % a vector for each input, while all other classifiers return a single
        % real-valued scalar for each input.
        result = evaluate(observations)
    end
    
end

