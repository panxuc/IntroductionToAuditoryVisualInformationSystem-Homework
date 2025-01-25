import torch

from model import HarryPotterTransformer
from dataset import Vocabulary


def generate(model, device, seed_words, input_length, output_length, vocab, temperature=1.0, strategy='sampling'):
    """
    Arguments:
        model: HarryPotterTransformer
        device: string, presents tensor's device
        seed_words: string
        input_length: int, model's max input length
        output_length: int, max output length during generation
        vocab: Vocabulary, defined in dataset.py
        temperature: float, temperature during model inference
        strategy: string, should be either 'sampling' based on model output probabilities 
                  or 'greedy' by choosing the maximum probability. 
    """

    model.eval()

    with torch.no_grad():

        output_arr = torch.LongTensor([-1]).to(device)

        ################################################################################
        # TODO: generate a paragraph from seed_words                                   #
        # To complete this task, you need to review the input and output formats       #
        # of various functions and models in other code file, and integrate them to    #
        # achieve our goal.                                                            #
        ################################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        output_arr = vocab.words_to_array(seed_words).to(device)
        input_arr = vocab.words_to_array(seed_words).to(device)
        for _ in range(output_length):
            if len(input_arr) > input_length:
                input_arr = input_arr[-input_length:]
            probabilities = model.inference(input_arr, temperature)
            if strategy == 'sampling':
                next_token = torch.multinomial(probabilities, num_samples=1).item()
            else:
                next_token = torch.argmax(probabilities).item()
            output_arr = torch.cat((output_arr, torch.LongTensor([next_token]).to(device)))
            input_arr = torch.cat((input_arr, torch.LongTensor([next_token]).to(device)))
            if vocab.ind2voc[next_token] == '#':
                break

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ################################################################################
        #                              END OF YOUR CODE                                #
        ################################################################################

        return vocab.array_to_words(output_arr[:-1]), len(output_arr)-1

# hyper-parameters, shouble be same as in main.py
SEQUENCE_LENGTH = 100
FEATURE_SIZE = 512
NUM_HEADS = 8
USE_CUDA = True

MODEL_TYPE = 'transformer'
DATA_PATH = 'data/'
EXP_PATH = f'exp_task5/'

if __name__=="__main__":
    use_cuda = USE_CUDA and torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")
    print('Using device', device)

    # build vocab
    vocab = Vocabulary(DATA_PATH + 'harry_potter_chars_train.pkl')

    # build model
    if MODEL_TYPE == 'transformer':
        model = HarryPotterTransformer(len(vocab), FEATURE_SIZE, NUM_HEADS).to(device)
        model.load_last_model(EXP_PATH + 'checkpoints/')
    else:
        raise NotImplementedError

    # generate
    # Try different seed_words. Find interesting ones.
    seed_words = "Harry Potter came to Tsinghua University and attended a course named Introduction to Auditory Visual Information System. "
    # Experiment with different temperatures to observe their impact on diversity and stability.
    temperature = 1.0
    # Experiment with different strategies.
    strategy = 'sampling'

    generated_sentence, array_length = generate(model, device, seed_words, SEQUENCE_LENGTH, 2000, vocab, temperature, strategy)
    print("Generated:\n", generated_sentence)
    print("Generated Length:", array_length)