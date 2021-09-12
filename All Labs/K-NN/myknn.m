% Lazy learner: not a typical training phase by definition - therefore very fast in this
% phase
% Classification slower the more examples - needs to check nearest neighbours
% in entire dataset
% No parameters and one hyperparameter - k
classdef myknn
    methods(Static)
        
        % The training process - builds the model, m.
        % fit() takes a copy of all the training examples, training
        % labels and k to create the structure m.
        function m = fit(train_examples, train_labels, k)
            
            % Start of z-score standardisation process
            % Stops a certain feature from drowning out other features
            % Makes sure all are scaled the same - can be compared
            % e.g. all features measured from 1-10
            % mean of the training data/standard deviation of the training
            % data
			m.mean = mean(train_examples{:,:});
			m.std = std(train_examples{:,:});
            for i=1:size(train_examples,1)
				train_examples{i,:} = train_examples{i,:} - m.mean;
                train_examples{i,:} = train_examples{i,:} ./ m.std;
            end
            % End of standardisation process
            
            % Store the training examples into m structure
            m.train_examples = train_examples;
            
            % Store the training labels into m structure 
            m.train_labels = train_labels;
            
            % Store the number of nearest neighbours to be used into m
            % structure
            m.k = k;
        
        end

        % This function makes use of the predict_one() function to make an 
        % array of predictions the classifier has made
        function predictions = predict(m, test_examples)

            predictions = categorical;
            
            % Loop through ALL of the test examples
            for i=1:size(test_examples,1)
                
                fprintf('classifying example example %i/%i\n', i, size(test_examples,1));
                
                % Store the current test example
                this_test_example = test_examples{i,:};
                
                % Start of standardisation process
                this_test_example = this_test_example - m.mean;
                this_test_example = this_test_example ./ m.std;
                % End of standardisation process
                
                % Call predict_one() on the current example
                this_prediction = myknn.predict_one(m, this_test_example);
                
                % Add the result to the predictions array
                predictions(end+1) = this_prediction;
            
            end
        
        end
        
        function prediction = predict_one(m, this_test_example)
            % Calculate the distance between the current example and all of
            % the other testing examples
            distances = myknn.calculate_distances(m, this_test_example);
            neighbour_indices = myknn.find_nn_indices(m, distances);
            prediction = myknn.make_prediction(m, neighbour_indices);
        
        end
 
        function distances = calculate_distances(m, this_test_example)
            
			distances = [];
            
            % Loop through all of the TRAINING examples, NOT testing
            % examples
			for i=1:size(m.train_examples,1)
                % Store current training example
				this_training_example = m.train_examples{i,:};
                
                % Calculate the distance between the current training
                % example and the current test example
                this_distance = myknn.calculate_distance(this_training_example, this_test_example);
                
                % Add the distance calculated to distances array
                distances(end+1) = this_distance;
            end
        
        end

        % Euclidean distance the straight line distance between 2 points
        % Used as a measure of similarity 
        % Uses Pythagoras' Thereom 
        function distance = calculate_distance(p, q)
			differences = q - p;
            squares = differences .^ 2;
            total = sum(squares);
            
            % Have to sqrt to get d and not d^2
            distance = sqrt(total);
        
		end

        function neighbour_indices = find_nn_indices(m, distances)
            % distances array needs to be sorted so it's clear which the
            % nearest neighbours are i.e. the ones with the shortest
            % distance
            % if k = 5, then the 5 shortest distances will be used
			[sorted, indices] = sort(distances);
            
            % An array of the original indices before they were sorted
            neighbour_indices = indices(1:m.k);
        
        end
        
        % Predicts which class the example belongs to
        % Checks which classes the nearest neighbours belong to 
        % Each label corresponds to an actual training example
        % It looks at the labels, comes up with the most common and uses
        % this to predict the class of the new/current testing example
        function prediction = make_prediction(m, neighbour_indices)
			neighbour_labels = m.train_labels(neighbour_indices);
            prediction = mode(neighbour_labels);
        
		end

    end
end