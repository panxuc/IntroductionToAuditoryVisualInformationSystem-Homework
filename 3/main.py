import torch
import torch.optim as optim
import numpy as np
import multiprocessing

import util
from dataset import HarryPotterDataset
from model import HarryPotterTransformer
from generate import generate


def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def train(model, device, train_loader, optimizer, lr, epoch, log_interval, model_type='transformer'):
    model.train()
    losses = []
    hidden = None
    for batch_idx, (data, label) in enumerate(train_loader):
        data, label = data.to(device), label.to(device)
        if model_type == 'transformer':
            output = model(data)
        else:
            raise NotImplementedError
        pred = output.max(-1)[1]
        loss = model.loss(output, label)
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    return np.mean(losses)


def test(model, device, test_loader, model_type='transformer'):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        hidden = None
        for batch_idx, (data, label) in enumerate(test_loader):
            data, label = data.to(device), label.to(device)
            if model_type == 'transformer':
                output = model(data)
            else:
                raise NotImplementedError
            pred = output.max(-1)[1]
            loss = model.loss(output, label)
            correct_mask = pred.eq(label.view_as(pred))
            num_correct = correct_mask.sum().item()
            correct += num_correct
            test_loss += loss.item()

    test_loss /= len(test_loader)
    test_accuracy = 100. * correct / (len(test_loader.dataset) * test_loader.dataset.sequence_length)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset) * test_loader.dataset.sequence_length,
        100. * correct / (len(test_loader.dataset) * test_loader.dataset.sequence_length)))
    return test_loss, test_accuracy


# hyper-parameters for training
SEQUENCE_LENGTH = 100
BATCH_SIZE = 196
FEATURE_SIZE = 512
NUM_HEADS = 8
TEST_BATCH_SIZE = 196
EPOCHS = 20
LEARNING_RATE = 0.0002
WEIGHT_DECAY = 0.0005
USE_CUDA = True
PRINT_INTERVAL = 10
TEMPERATURE = 0.5

MODEL_TYPE = 'transformer'
DATA_PATH = 'data/'
EXP_PATH = 'exp_task5/'



def main():
    use_cuda = USE_CUDA and torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")
    print('Using device', device)
    num_workers = multiprocessing.cpu_count()
    print('num workers:', num_workers)

    # build dataset and dataloader
    data_train = HarryPotterDataset(DATA_PATH + 'harry_potter_chars_train.pkl', SEQUENCE_LENGTH, BATCH_SIZE)
    data_test = HarryPotterDataset(DATA_PATH + 'harry_potter_chars_test.pkl', SEQUENCE_LENGTH, TEST_BATCH_SIZE)
    vocab = data_train.vocab

    kwargs = {'num_workers': num_workers,
            'pin_memory': True,
            'drop_last': True} if use_cuda else {}

    train_loader = torch.utils.data.DataLoader(data_train, batch_size=BATCH_SIZE,
                                            shuffle=False, **kwargs)
    test_loader = torch.utils.data.DataLoader(data_test, batch_size=TEST_BATCH_SIZE,
                                            shuffle=False, **kwargs)

    # build model
    if MODEL_TYPE == 'transformer':
        model = HarryPotterTransformer(data_train.vocab_size(), FEATURE_SIZE, NUM_HEADS).to(device)
    else:
        raise NotImplementedError
    start_epoch = model.load_last_model(EXP_PATH + 'checkpoints/')
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(model)
    print(f"Parameters: {params/1e6}M")

    # build optimizer
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    train_losses, test_losses, test_accuracies = util.read_log(EXP_PATH + 'logs/log.pkl', ([], [], []))
    test_loss, test_accuracy = test(model, device, test_loader, model_type=MODEL_TYPE)
    test_accuracies.append((start_epoch, test_accuracy))

    print("Begin training...")
    try:
        for epoch in range(start_epoch, EPOCHS):
            lr = LEARNING_RATE * np.power(0.25, (int(epoch / 6)))
            train_loss = train(model, device, train_loader, optimizer, lr, epoch, PRINT_INTERVAL, model_type=MODEL_TYPE)
            test_loss, test_accuracy = test(model, device, test_loader, model_type=MODEL_TYPE)

            # log and save
            train_losses.append((epoch+1, train_loss))
            test_losses.append((epoch+1, test_loss))
            test_accuracies.append((epoch+1, test_accuracy))
            util.write_log(EXP_PATH + 'logs/log.pkl', (train_losses, test_losses, test_accuracies))
            util.plot(
                [[epoch for (epoch, loss) in train_losses], [epoch for (epoch, loss) in test_losses]],
                [[loss for (epoch, loss) in train_losses], [loss for (epoch, loss) in test_losses]],
                'loss', 'epoch', 'loss', EXP_PATH + 'figs/loss.png', ['train_loss', 'test_loss']
            )
            util.plot(
                [epoch for (epoch, acc) in test_accuracies],
                [acc for (epoch, acc) in test_accuracies],
                'test_acc', 'epoch', 'acc', EXP_PATH + 'figs/test_acc.png'
            )
            print(f"[Epoch {epoch}] train_loss: {train_loss}, test_loss: {test_loss}, test_accuracy: {test_accuracy}")
            model.save_best_model(test_accuracy, EXP_PATH + 'checkpoints/%03d.pt' % epoch)

            # generate sample
            # seed_words = 'Harry Potter'
            # if MODEL_TYPE == 'transformer':
            #     generated_sentence = generate(model, device, seed_words, SEQUENCE_LENGTH, 2000, vocab)
            # else:
            #     raise NotImplementedError
            # print(f'generated sample from seed words "{seed_words}":\t', generated_sentence)

    except KeyboardInterrupt as ke:
        print('Interrupted')
    except:
        import traceback
        traceback.print_exc()
    finally:
        print('Saving final model')
        model.save_model(EXP_PATH + 'checkpoints/%03d.pt' % epoch, 0)


if __name__=='__main__':
    main()