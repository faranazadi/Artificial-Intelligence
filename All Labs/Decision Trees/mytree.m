classdef mytree
    methods(Static)
        
        % The training process - builds the model, m.
        % fit() takes a copy of all the training examples, training
        % labels and sets up the tree structure.
        function m = fit(train_examples, train_labels)
            % 'ID' of the node 
			emptyNode.number = [];
            
            % The examples and class labels stored in the node
            emptyNode.examples = [];
            emptyNode.labels = [];
            
            % Prediction based on the classes in the node
            emptyNode.prediction = [];
            
            % The value of how impure the class labels are in the node
            emptyNode.impurityMeasure = [];
            
            % If split, store 2 child nodes and split training data
            emptyNode.children = {};
            
            % The feature and it's value - used for splitting
            emptyNode.splitFeature = [];
            emptyNode.splitFeatureName = [];
            emptyNode.splitValue = [];

            m.emptyNode = emptyNode;
            
            % Sets up the root node 
            r = emptyNode;
            r.number = 1;
            r.labels = train_labels;
            r.examples = train_examples;
            r.prediction = mode(r.labels);

            m.min_parent_size = 10;
            m.unique_classes = unique(r.labels);
            m.feature_names = train_examples.Properties.VariableNames;
			m.nodes = 1;
            m.N = size(train_examples,1);

            % Attempt to split the root node
            m.tree = mytree.trySplit(m, r);

        end
        
        % A function to attempt to split nodes starting at the
        % root node
        % Aims to increase purity, if it can't, then it won't split
        function node = trySplit(m, node)

            % Not enough examples - don't split
            if size(node.examples, 1) < m.min_parent_size
				return
            end

            % See how pure the node is
            % Ideally we want a node to contain one class label
            node.impurityMeasure = mytree.weightedImpurity(m, node.labels);

            % Every feature
            for i=1:size(node.examples,2)

				fprintf('evaluating possible splits on feature %d/%d\n', i, size(node.examples,2));
                
                % Reorder the examples and labels based on this feature
                % Makes it easier to create different splits
				[ps,n] = sortrows(node.examples,i);
                ls = node.labels(n);
                biggest_reduction(i) = -Inf;
                biggest_reduction_index(i) = -1;
                biggest_reduction_value(i) = NaN;
                
                % Every unique value
                for j=1:(size(ps,1)-1)
                    % Checks to see if next value of re-ordered data is
                    % identical to the current one
                    % Stops splitting on the same value more than once
                    if ps{j,i} == ps{j+1,i}
                        continue;
                    end
                    
                    this_reduction = node.impurityMeasure - (mytree.weightedImpurity(m, ls(1:j)) + mytree.weightedImpurity(m, ls((j+1):end)));
                    
                    % If the current impurity reduction is bigger than the
                    % last recorded impurity reduction - store the new one
                    if this_reduction > biggest_reduction(i)
                        biggest_reduction(i) = this_reduction;
                        biggest_reduction_index(i) = j;
                    end
                end
				
            end

            [winning_reduction,winning_feature] = max(biggest_reduction);
            winning_index = biggest_reduction_index(winning_feature);

            % If it's possible to reduce impurity based on the calculations
            % above then split, if not then keep as is.
            if winning_reduction <= 0
                return
            else

                [ps,n] = sortrows(node.examples,winning_feature);
                ls = node.labels(n);

                node.splitFeature = winning_feature;
                node.splitFeatureName = m.feature_names{winning_feature};
                node.splitValue = (ps{winning_index,winning_feature} + ps{winning_index+1,winning_feature}) / 2;

                node.examples = [];
                node.labels = []; 
                node.prediction = [];

                node.children{1} = m.emptyNode;
                m.nodes = m.nodes + 1; 
                node.children{1}.number = m.nodes;
                node.children{1}.examples = ps(1:winning_index,:); 
                node.children{1}.labels = ls(1:winning_index);
                node.children{1}.prediction = mode(node.children{1}.labels);
                
                node.children{2} = m.emptyNode;
                m.nodes = m.nodes + 1; 
                node.children{2}.number = m.nodes;
                node.children{2}.examples = ps((winning_index+1):end,:); 
                node.children{2}.labels = ls((winning_index+1):end);
                node.children{2}.prediction = mode(node.children{2}.labels);
                
                node.children{1} = mytree.trySplit(m, node.children{1});
                node.children{2} = mytree.trySplit(m, node.children{2});
            end

        end
        
        % Calculates the impurity of a node - used to decide whether to
        % split or not
        % Ideally a perfectly pure/zero impurity node has class labels from
        % one class
        % Measured using Gini's Diversity Index (GDI)
        % A pure node has a GDI of 0
        % The greater the impurity the more positive it becomes
        function e = weightedImpurity(m, labels)

            weight = length(labels) / m.N;

            summ = 0;
            obsInThisNode = length(labels);
            
            % Loop through unique class labels
            for i=1:length(m.unique_classes)
                % Calculate the fraction of class labels which belong to
                % this particular class
                % Square the result
                % Add it to summ
                % Subtract from 1 later on
				pi = length(labels(labels==m.unique_classes(i))) / obsInThisNode;
                summ = summ + (pi*pi);
            
			end
            g = 1 - summ;
            
            e = weight * g;

        end

        function predictions = predict(m, test_examples)

            predictions = categorical;
            
            % Loop through the test examples
            for i=1:size(test_examples,1)
                
				fprintf('classifying example %i/%i\n', i, size(test_examples,1));
                
                % Store the current test example, call predict_one() and
                % store the result
                this_test_example = test_examples{i,:};
                this_prediction = mytree.predict_one(m, this_test_example);
                predictions(end+1) = this_prediction;
            
			end
        end
        
        % Calls descend_tree() function which will continue until it
        % reaches a leaf node and therefore a prediction
        function prediction = predict_one(m, this_test_example)    
			node = mytree.descend_tree(m.tree, this_test_example);
            prediction = node.prediction;
        
        end
        
        % Descend the decision tree recursively 
        % Apply the splitting rules (checks splitFeature and
        % splitValue) and compares to the current test example
        % Continues until a leaf node has been reached
        % Return the class prediction
        function node = descend_tree(node, this_test_example)
            
			if isempty(node.children)
                return;
            else
                if this_test_example(node.splitFeature) < node.splitValue
                    node = mytree.descend_tree(node.children{1}, this_test_example);
                else
                    node = mytree.descend_tree(node.children{2}, this_test_example);
                end
            end
        
		end
        
        % describe a tree:
        function describeNode(node)
            
			if isempty(node.children)
                fprintf('Node %d; %s\n', node.number, node.prediction);
            else
                fprintf('Node %d; if %s <= %f then node %d else node %d\n', node.number, node.splitFeatureName, node.splitValue, node.children{1}.number, node.children{2}.number);
                mytree.describeNode(node.children{1});
                mytree.describeNode(node.children{2});        
            end
        
		end
		
    end
end