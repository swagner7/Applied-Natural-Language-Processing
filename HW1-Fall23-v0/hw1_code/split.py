def split_data(data, split_ratio = 0.8):
    
    '''
	ToDo: Split the dataset into train and test data using the split_ratio.
	For Autograder purposes, there is no need to shuffle the dataset.

	Input:
		data: dataframe containing the dataset. 
		split_ratio: desired ratio of the train and test splits.
		
	Output:
		train: train split of the data
		test: test split of the data
	'''

    split_point = int(split_ratio * len(data))

    train = data.iloc[:split_point]
    test = data.iloc[split_point:]

    return(train, test)