import io
import torch


def forbidden_list(word):
    forbidden_words = ['<unk>', 'N', '$']
    return True
    if word in forbidden_words:
        return False
    else:
        return True


def import_ptb_dataset(dataset_type='train', path='./data/ptb', batch_size=20, vocabulary={}):
    dataset_path = path + '/' + 'ptb.' + dataset_type + '.txt'
    # Here we read each line, and append each word that has not appeared previously to the list
    with open(dataset_path) as f:
        text = f.read()
        # We take the text, split it (default by sapce) than define it as a set (to remove duplicates) and finally sort it to have consistensy.
        if len(vocabulary) < 1:
            text_splited = text[1:].split(' ')
            word_span = sorted(set(text_splited))
            vocabulary = {}
            for count, word in enumerate(word_span):
                vocabulary[word] = count
        else:
            text_splited = text[1:].split(' ')
    # Here we take a large vector that maps the relative positions of the words in vocabulary
    word_pos = []
    for word in text_splited:
        word_pos.append(vocabulary[word])
    length_var = len(word_pos)
    num_allowed = (length_var//batch_size)*batch_size
    word_pos = torch.Tensor(word_pos[:num_allowed])
    # Super important
    # For some reason when using the embedding layer as input, if the input is floating rather than integer
    # the back prop algorithm is severely downgraded (factor of 3 w.r.t perplexity)
    return word_pos.view(batch_size, -1).int(), vocabulary
