import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds




class TextProcessor(object):
    LOWER_CASE_CHARS = list("abcdefghijklmnopqrstuvwxyz")
    UPPER_CASE_CHARS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    NUMBERS = list("0123456789")
    PUNCTUATION = list(".,!?:;\'\"")
    WHITESPACE = list(" \n\t")
    
    def __init__(self, 
        vocab,
        convert_to_lower=False,
        ignore_unknown=True):

        if convert_to_lower:
            # make sure vocab only contains lowercase
            vocab = sorted(set(map(lambda x: x.lower(), vocab)))

        self.token_to_idx = dict()
        self.idx_to_token = dict()
        for token in vocab:
            idx = len(self.token_to_idx)
            self.token_to_idx[token] = idx
            self.idx_to_token[idx] = token

        # adding special characters for padding and N/A
        self.token_to_idx["<PAD>"] = len(self.token_to_idx)
        self.idx_to_token[len(self.idx_to_token)] = "<PAD>"
        self.token_to_idx["<NAN>"] = len(self.token_to_idx)
        self.idx_to_token[len(self.idx_to_token)] = "<NAN>"
        self.vocab_size = len(self.token_to_idx)

        self.convert_to_lower = convert_to_lower
        self.ignore_unknown = ignore_unknown

    def convert_to_idx(self, token_list):
        idx_list = []
        for token in token_list:
            if self.convert_to_lower:
                cleaned_token = token.lower()
            else:
                cleaned_token = token

            if cleaned_token in self.token_to_idx:
                idx_list.append(self.token_to_idx[cleaned_token])
            else:
                if not self.ignore_unknown:
                    idx_list.append(self.token_to_idx["<NAN>"])
        return idx_list

    def convert_to_tokens(self, idx_list):
        token_list = []
        for idx in idx_list:
            if idx < self.vocab_size:
                token_list.append(self.idx_to_token[idx])
            else:
                raise Exception("Invalid idx")
        return token_list


def create_char_pred_ds(idx_list, seq_len=100, batch_size=32):
    curr_char = tf.convert_to_tensor(idx_list[:-1])
    next_char = tf.convert_to_tensor(idx_list[1:])
    
    ds = tf.data.Dataset.from_tensor_slices((curr_char, next_char))
    ds = ds.batch(seq_len, drop_remainder=True)
    ds = ds.batch(batch_size)
    return ds

def load_tiny_shakespeare():
    data = tfds.load(name='tiny_shakespeare')['train']
    text = [x for x in data][0]['text'].numpy().decode('utf-8')
    return text

if __name__ == "__main__":
    text = load_tiny_shakespeare()
    vocab = sorted(set(text))

    text_processor = TextProcessor(vocab)
    idx_list = text_processor.convert_to_idx(text)

    ds = create_char_pred_ds(idx_list)
    