% Naive Bayes works on conditional probability - how likely something is to
% occur based if something else occurrs. These are dependent events.
% It treats features in the dataset equally and classes them as
% independent - class conditional independence. As a result, may not pick
% up on dependent relationships in the dataset.
% Simple and fast algorithm but doesn't work well with datasets containing
% lots of numeric features. e.g. works well identifying spam e-mails.

classdef mynb
    methods(Static)
        
        % The training process - builds the model, m.
        % Step 1: calculate probability densities
        % Step 2: calculate a prior (the likelihood of a class being
        % present based on how often it appears in the training data)
        function m = fit(train_examples, train_labels)
            % Get all of the possible classes/labels from the training data
            m.unique_classes = unique(train_labels);
            
            % Calculate how many classes the dataset has
            m.n_classes = length(m.unique_classes);

            m.means = {};
            m.stds = {};
            
            for i = 1:m.n_classes
            
                % Store the current class 
				this_class = m.unique_classes(i);
                
                % Grab all of the examples from this current, particular
                % class
                examples_from_this_class = train_examples{train_labels==this_class,:};
                
                % Now we have the examples from every class
                % Calculate the mean and standard deviation
                m.means{end+1} = mean(examples_from_this_class);
                m.stds{end+1} = std(examples_from_this_class);
            
			end
            
            m.priors = [];
            
            for i = 1:m.n_classes
                
                % Store the current class
				this_class = m.unique_classes(i);
                
                % Grab all of the examples from this current, particular
                % class
                examples_from_this_class = train_examples{train_labels==this_class,:};
                
                % Divide the number of examples from THIS CLASS by the
                % total number of ALL the examples
                % This number is used to estimate whether a chosen example
                % belongs to this particular class or not
                m.priors(end+1) = size(examples_from_this_class,1) / size(train_labels,1);
            
			end

        end

        % Predicts which class each examlple from the testing data belongs
        % to
        % Step 1: for the example being classified, based on each class,
        % calculate a likelihood
        % Step 2: Value proportional to posterior probability = each likelihood * the prior
        % Step 3: Predict which class the example belongs to using the
        % value calculated above 
        function predictions = predict(m, test_examples)

            predictions = categorical;

            % Loop through all of the test examples
            for i=1:size(test_examples,1)

				fprintf('classifying example %i/%i\n', i, size(test_examples,1));
                
                % Store the current test example
                this_test_example = test_examples{i,:};
                
                % Call predict_one() on the current example and store the
                % result in the predictions array
                this_prediction = mynb.predict_one(m, this_test_example);
                predictions(end+1) = this_prediction;
            
			end
        end

        function prediction = predict_one(m, this_test_example)

            % Loop over all of the classes
            for i=1:m.n_classes

                % Calculate the likelihood for the current example in the
                % current class
				this_likelihood = mynb.calculate_likelihood(m, this_test_example, i);
                this_prior = mynb.get_prior(m, i);
                posterior_(i) = this_likelihood * this_prior;
            
            end

            % Use max() to grab the most probable class (highest value in
            % the array)
            [winning_value_, winning_index] = max(posterior_);
            prediction = m.unique_classes(winning_index);

        end
        
        % Every feature (its value) is taken into account here - class conditional independence. 
        % Each feature = an independent event
        % Calcluate the likelihood of the example from the particular class
        % Done by looking at the distribution of the feature in examples in
        % the class
        function likelihood = calculate_likelihood(m, this_test_example, class)
            
			likelihood = 1;
            
            % Multiply the probability densitites for each feature in the
            % particular example
			for i=1:length(this_test_example)
                likelihood = likelihood * mynb.calculate_pd(this_test_example(i), m.means{class}(i), m.stds{class}(i));
            end
        end
        
        % Get the probability from the model
        % Return the prior(s) from particular class
        function prior = get_prior(m, class)
            
			prior = m.priors(class);
        
        end
        
        % Probability density function written without in-built functions
        % Calculates the probability density at a given value
        % in a normal distribution
        function pd = calculate_pd(x, mu, sigma)
        
			first_bit = 1 / sqrt(2*pi*sigma^2);
            second_bit = - ( ((x-mu)^2) / (2*sigma^2) );
            pd = first_bit * exp(second_bit);
        
		end
            
    end
end