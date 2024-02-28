# %% [markdown]
# ### Import Necessary Libraries

# %%
import numpy as np
import pandas as pd
import string  
import re
from tqdm import tqdm
from argparse import ArgumentParser

# %% [markdown]
# ### Load and Clean Dataset
import os
import sys

class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)

    def flush(self):
        for f in self.files:
            f.flush()

parser = ArgumentParser()
parser.add_argument('--idx', type=int, default=0)
parser.add_argument('--w2', type=float, default=0.2)
parser.add_argument('--w3', type=float, default=0.3)
parser.add_argument('--w4', type=float, default=0.4)
parser.add_argument('--w5', type=float, default=0.5)
parser.add_argument('--w6', type=float, default=0.6)
parser.add_argument('--w7', type=float, default=0.7)
parser.add_argument('--q', type=float, default=0.75)

args = parser.parse_args()
idx = args.idx
w2 = args.w2
w3 = args.w3
w4 = args.w4
w5 = args.w5
w6 = args.w6
q = args.q

args = vars(args)
# print(args)

# 重定向输出到文件和终端
f = open(f'{idx}_output.txt', 'w')
sys.stdout = Tee(sys.stdout, f)


# %%
# Load word set
word_path = './words_250000_train.txt'
word_file = open(word_path, 'r')
words = word_file.readlines()

# %%
# print(f'Total Number of Words in Dataset: {len(words)}')

# %%
# Clean word set by removing whitespace, converting to lowercase, removing duplicates, and removing words with numbers
words = [w.strip('\n').lower() for w in words]
words = list(set(words))
words = [w for w in words if w.isalpha()]

# %%
# print(f'Total Number of Words in Dataset After Cleaning: {len(words)}')

# %%
# Split word set into training and validation
train_idx = np.random.choice(len(words), len(words) // 2, replace=False)
val_idx = np.array(list(set(range(len(words))) - set(train_idx)))

train_words = list(np.array(words)[train_idx])
val_words = list(np.array(words)[val_idx])

# %%
# print(f'Total Number of Words in Training Dataset: {len(train_words)}')
# print(f'Total Number of Words in Training Dataset: {len(val_words)}')

# %% [markdown]
# ### Define Helper Functions

# %%
def letter2index(letter):
    return ord(letter) - ord('a')

# %%
def index2letter(ind):
    return chr(ind + ord('a'))

# %%
class HangmanGame():
    
    def __init__(self, corpus, N_LIVES=6):

        self.corpus = corpus
        self.N_LIVES = N_LIVES

    def word2info(self):

        info = []

        for letter in self.word:
            if letter in self.guessed:
                info.append(letter)
            else:
                info.append('_')

        return info

    def start(self, verbose=False):

        self.word = np.random.choice(self.corpus, 1)[0]
        self.unused = set(string.ascii_lowercase)
        self.guessed = set()
        self.info = self.word2info()
        self.LIVES_LEFT = self.N_LIVES
        self.ongoing = True
        self.verbose = verbose

        if self.verbose:
            print(self.info)
            print(f'LIVES LEFT: {self.LIVES_LEFT}')

        return self.info

    def guess(self, letter):

        if not self.ongoing:
            
            if self.verbose:
                print('The game is already over! Stop making guesses!')
            return -2
        
        elif letter in self.guessed:
            
            if self.verbose:
                print('You already guessed this letter! Try another letter!')
            return 0
        
        else:

            update_set = set([letter])
            self.guessed.update(update_set)
            self.unused = self.unused - update_set
            self.info = self.word2info()

            if letter not in self.word:
                self.LIVES_LEFT -= 1

            if self.LIVES_LEFT == 0:

                if self.verbose:
                    print(f'You lose! The word was {self.word}.')
                return -1
            elif ''.join(self.info) == self.word:
                if self.verbose:
                    print('You win!')
                return 2
            else:
                if self.verbose:
                    print(self.info)
                    print(f'LIVES LEFT: {self.LIVES_LEFT}')
                return self.info

# %%
def count_ngrams(corpus, n):

    ngrams = np.zeros((26,) * n)

    for word in corpus:
        for i in range(0, len(word)-n+1):

            indices = np.zeros(n, dtype='int')

            for j in range(n):
                indices[j] = letter2index(word[i+j])

            indices = tuple(indices)
            ngrams[indices] += 1
            
    return ngrams

# %%
def mask_vec(vec, guessed, mask_val=0):

    vec = vec.copy()

    if len(guessed) > 0:
        
        for letter in guessed:
            vec[letter2index(letter)] = 0

    return vec

# %%
def get_window(lst, index, max_width):
    
    max_side_width = max_width // 2
    start_width = 0
    for i in range(1, max_width):
        if (index - i >= 0) and (lst[index - i] != '_'):
            start_width += 1
        else:
            break

    end_width = 0
    for i in range(1, max_width):
        if (index + i < len(lst)) and (lst[index + i] != '_'):
            end_width += 1
        else:
            break
    
    if (end_width >= max_side_width) and (start_width >= max_side_width):
        end_width = max_side_width
        start_width = max_side_width
    elif end_width < max_side_width:
        start_width = min(2*max_side_width - end_width, start_width)
    elif start_width < max_side_width:
        end_width = min(2*max_side_width - start_width, end_width)

    start = index - start_width
    end = index + end_width

    sublist = lst[start:end + 1]
    return sublist, start_width, end_width

# %%
def make_guess(grams, info, guessed, max_width=1):

    probs = np.zeros(26)

    n = len(info)

    for i in range(n):

        if info[i] == '_':
            single_probs = np.zeros(26)
            window, left_width, right_width = get_window(info, i, max_width=max_width)
            while left_width > 0 or right_width > 0:
                m = len(window)
                vec = grams[m-1]
                sub = 0
                for j in range(m):
                    if j != left_width:
                        vec = np.take(vec, letter2index(window[j]), axis=j-sub)
                        sub += 1
                vec = vec.copy()
                vec = mask_vec(vec, guessed)
                if vec.sum() != 0:
                    single_probs += args[f'w{left_width+right_width+1}'] * vec / vec.sum()
                if left_width >= right_width:
                    left_width -= 1
                    window = window[1:]
                else:
                    right_width -= 1
                    window = window[:-1]
            if single_probs.sum() == 0:
                single_probs = grams[0]
            probs += single_probs / single_probs.sum()
    
    probs = mask_vec(probs, guessed, mask_val=-1)
    return index2letter(np.argmax(probs))



# %%
def info2regex(info, unused):

    regex = r'^'
    unknown = rf"[{''.join(list(unused))}]"

    for space in info:
        if space == '_':
            regex = regex + unknown
        else:
            regex = regex + rf'{space}'

    regex = regex + r'$'

    return regex

# %%
vowels = 'a e i o u'.split()

# %%
def vowel_percentage(word):
    vowel_count = sum(1 for char in word if char in vowels)
    total_letters = sum(1 for char in word if char != '_') #len(word)
    if total_letters == 0:
        percentage = 0
    else:
        percentage = (vowel_count / total_letters) * 100
    return percentage

# %% [markdown]
# ### Run Trials for Algorithm

# %%
N_TRIALS = 1000
game = HangmanGame(val_words)

# %%
full_grams = []
max_ngrams = 7

for i in range(1, max_ngrams + 1):
    full_grams.append(count_ngrams(train_words, i))

# %%
wins = 0

print(f"idx {idx} start run")
# %%
for trial in range(N_TRIALS):

    info = game.start()
    win = False
    unused = game.unused.copy()
    guessed = game.guessed.copy()
    n_lives = game.LIVES_LEFT
    
    m = len(info)
    vps = [vowel_percentage(word) for word in train_words if len(word) <= m]
    threshold = np.quantile(np.array(vps), q=q)

    
    while (n_lives > 0) and (win == False):
        # print("local: ", guessed)
        # print("game: ", game.guessed)
        guess = make_guess(full_grams, info, guessed, max_width=max_ngrams)

        output = game.guess(guess)

        if type(output) == list:
            info = output
        elif output == 2:
            win = True

        unused = game.unused.copy()
        guessed = game.guessed.copy()

        vp = vowel_percentage(info)

        if (vp >= threshold):
            unused = unused - set(vowels)
            guessed.update(set(vowels))

        n_lives = game.LIVES_LEFT

    if win:
        wins += 1


# %%
print(f"idx: {idx}, max_width: {max_ngrams}")
print(idx, args)
print(f'{idx}, Win Percentage on Validation Set: {wins / N_TRIALS}')
