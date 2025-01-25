import pickle
import torch
from torchvision import datasets


def prepare_data(data_path):
    with open(data_path) as f:
        data = f.read()

    ################################################################################
    # TODO: insert '#' at the end of each paragraph, delete all special chars;     #
    #  you can leverage data.replace() and data.split() to accomplish that.        #
    #  two or three lines of code should be sufficient.                            #
    ################################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    data = data.replace('\n\n', '#')
    data = ' '.join(data.split())
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ################################################################################
    #                              END OF YOUR CODE                                #
    ################################################################################
    
    voc2ind = {}
    # Compute voc2ind and transform the data into an integer representation of the tokens.
    unique_chars = set(data)
    for idx, char in enumerate(unique_chars):
        voc2ind[char] = idx
    ind2voc = {val: key for key, val in voc2ind.items()}

    # split into train & test datasets
    train_text = data[:int(0.8*len(data))]
    train_text = [voc2ind[char] for char in train_text]
    test_text = data[int(0.8*len(data)):]
    test_text = [voc2ind[char] for char in test_text]

    save_train_dataset_path = DATA_PATH + 'harry_potter_chars_train.pkl'
    save_test_dataset_path = DATA_PATH + 'harry_potter_chars_test.pkl'
    pickle.dump({'tokens': train_text, 'ind2voc': ind2voc, 'voc2ind':voc2ind}, open(save_train_dataset_path, 'wb'))
    pickle.dump({'tokens': test_text, 'ind2voc': ind2voc, 'voc2ind':voc2ind}, open(save_test_dataset_path, 'wb'))
    print("Vocabulary size: ",len(ind2voc))

    return save_train_dataset_path, save_test_dataset_path


class Vocabulary(object):
    def __init__(self, data_file):
        with open(data_file, 'rb') as data_file:
            dataset = pickle.load(data_file)
        self.ind2voc = dataset['ind2voc']
        self.voc2ind = dataset['voc2ind']

    # Returns a string representation of the tokens.
    def array_to_words(self, arr):
        return ''.join([self.ind2voc[int(ind)] for ind in arr])

    # Returns a torch tensor representing each token in words.
    def words_to_array(self, words):
        return torch.LongTensor([self.voc2ind[word] for word in words])

    # Returns the size of the vocabulary.
    def __len__(self):
        return len(self.voc2ind)


class HarryPotterDataset(torch.utils.data.Dataset):
    def __init__(self, data_file, sequence_length, batch_size):
        super(HarryPotterDataset, self).__init__() 

        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.vocab = Vocabulary(data_file)

        with open(data_file, 'rb') as data_pkl:
            dataset = pickle.load(data_pkl)
        
        self.tokens = dataset['tokens']
        self.voc2ind = dataset['voc2ind']
        self.ind2voc = dataset['ind2voc']

        self.data = None
        self.sequences_in_batch = None
        ################################################################################
        # TODO: split self.tokens to len(self.tokens)//batch_size chunks, store the    #
        #  reshaped data (and convert it to torch.LongTensor) in self.data;            #
        #  Then compute how many sequences are there in each chunk, store that in      #
        #  self.sequences_in_batch                                                     #
        ################################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        self.data = torch.LongTensor(self.tokens[:len(self.tokens) // self.batch_size * self.batch_size])
        self.data = self.data.view(self.batch_size, -1)
        self.sequences_in_batch = self.data.size(1) // self.sequence_length

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ################################################################################
        #                              END OF YOUR CODE                                #
        ################################################################################

    def __len__(self):
        ################################################################################
        # TODO: return the total number of sequences in dataset                        #
        ################################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        return self.sequences_in_batch * self.batch_size

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ################################################################################
        #                              END OF YOUR CODE                                #
        ################################################################################
        
    def __getitem__(self, idx):
        data = None
        ################################################################################
        # TODO: Based on idx, determine the chunk idx and the sequence idx of the chunk#
        #  fetch that sequence data from self.data and store that in data variable;    #
        #  Note the data length should be sequence_length + 1                          #
        ################################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        chunk_idx = idx // self.sequences_in_batch
        seq_idx = idx % self.sequences_in_batch
        start = seq_idx * self.sequence_length
        end = start + self.sequence_length + 1
        data = self.data[chunk_idx, start:end]

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ################################################################################
        #                              END OF YOUR CODE                                #
        ################################################################################

        # returns input data and label data (next token of input) with their length sequence_length
        return data[:-1], data[1:]

    def vocab_size(self):
        return len(self.vocab)


if __name__=='__main__':
    DATA_PATH = 'data/'
    with open(DATA_PATH + 'harry_potter.txt', 'r', encoding = 'utf-8') as f:
        text = f.read()
    print('Some sample text from the book -->')
    print(text[:1000])

    total_words = len(text.split())
    total_characters = len(text)
    unique_words = len(set(text.split()))
    unique_characters = len(set(text))
    paragraphs = text.split('\n\n')

    print ("Total words in book :", total_words)
    print ("Total characters in book :", total_characters)
    print ("Unique words in book :", unique_words)
    print ("Unique characters in book :", unique_characters)
    print ("Total Paragraphs :", len(paragraphs))

    print("Preparing data...")
    train_dataset_path, test_dataset_path = prepare_data(DATA_PATH + 'harry_potter.txt')
    print(f"Prepared data has been saved in {train_dataset_path} and {test_dataset_path}")