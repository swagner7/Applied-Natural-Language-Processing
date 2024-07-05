def test_init_dataset(dataset_type):
    if ('IterableDataset' in dataset_type):
        print("Dataset type test passed \n")
    else:
        print("Dataset test faied. Type of dataset should be Iterable Dataset. Your type is: ", dataset_type, "\n")

def test_init_tokenizer(tokenizer_type):
    if ('DistilBertTokenizerFast' in tokenizer_type):
        print("Init tokenizer test passed \n")
    else: 
        print("Type of tokenizer should be Dilbert Tokenizer Fast. Your type is: ", tokenizer_type, "\n")

tokenizer_example_sentences = {"text": ["This sentence is the truth!!", "This sentence isn't the truth.", "One of these sentences isn't a truth?"]}

def test_tokenize(tokenizer_output):
    example_tokens_dict = {'tokens': [['[CLS]', 'This', 'sentence', 'is', 'the', 'truth', '!', '!', '[SEP]', '[PAD]'], ['[CLS]', 'This', 'sentence', 'isn', "'", 't', 'the', 'truth', '.', '[SEP]'], ['[CLS]', 'One', 'of', 'these', 'sentences', 'isn', "'", 't', 'a', '[SEP]']]}
   
    print(f'Expected Output: {example_tokens_dict}\n')
    print(f'Your Output: {tokenizer_output}')


    if ('tokens' not in tokenizer_output.keys()):
        print("Test failed, please check the structure of what you are returning")
        return
    if (len(tokenizer_output['tokens']) != 3):
        print("Test failed, please check to make sure you are handling batching")
        return
    if (len(tokenizer_output['tokens'][0]) != 10):
        print("Test failed, please check to make sure you are handling padding/truncation properly")
        return
    print("\nTokenizer functionality tests passed! \n")

def test_preprocess(first_row):
    first_token = first_row['tokens'][0]
    second_token = first_row['tokens'][1]
    if second_token == 'Ana' and first_token == '[CLS]':
        print("First row test passed")
    else:
        print("Preprocess test failed")

