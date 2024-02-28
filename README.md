# The ngrams algorithm for hangman

This is the report for the ngrams algorithm for hangman.

## Introduction

The key idea is to use n-grams to model the correlation of letters in the given dictionary. Then, we can use the n-grams to predict the middle letter by its neighboring letters. However, single n-gram are not enough to accurately predict the middle letter, so we use a combination of n-grams to improve the prediction accuracy. The weights of different n-grams ware adjusted by hyper-parameter search. As the vowels happends frequently, thus the model will guess vowels firstly. But in a single word, the proportion of vowels is not so high. So we limite the proportion of vowels to be q quantile, and the model will not guess vowels if the proportion of vowels is greater than q quantile.

## Hangman

The game is a two-player game. The first player is the computer, and the second player is the user. The computer will randomly choose a word from the dictionary and return the length of word, and the user need to guess the word. If the user guesses a letter that is in the word, the word is redisplayed with all instances of that letter shown in the correct positions, along with any letters correctly guessed on previous turns. If the letter does not appear in the word, the user is charged with an incorrect guess. The user keeps guessing letters until either (1) the user has correctly guessed all the letters in the word or (2) the user has made six incorrect guesses.

## Angorithm

The algorithm has 3 steps:

1. **Init data:** For all words in the dictionary, calculate the number of occurences of all n-grams (for n = 1 to 7). This step is memory-cosuming if n is large. Also, we calculate the q quantile of the proportion of vowels in all words less than the length of the word need to be guessed.
2. **Check guessed** Before guessing a new letter, we first check the proportion of vowels in exist word. If the proportion is larger than q quantile, we add all the vowels to the guessed list temporarily. Then we will not guess the vowels in this turn. We may continue to guess the vowels in the next turn if the proportion is small than q quantile.
3. **Guess letter:** For each letter in the word need to be guessed, claculate the predicted probability(filtering the guessed letter) of each n-gram. Then combining the probabilities of all n-grams by pre-defined weights, we get the predicted probability of a position. Then we sum the predicted probability of all positions to get the predicted probability for the whole word and pick the letter with the highest predicted probability as the guessed letter in this turn.

After **init data**, we will continue to run step 2 and 3 until either (1) the word has been correctly guessed or (2) the number of turns exceeds the limit.

## Discussion

This algorithm is robust than previous max information algorithm. It can guess unknow word with high accuracy. But this algorithm is memory-cumsumptive. It needs to store all the n-grams of the word in the dictionary. Also, there are other method to solve this problem, such as Deep RL, it will save a lot of memory.

## Result

After search the haper-parameter, I found the best parameter is `{'w2': 0.2, 'w3': 0.3, 'w4': 0.4, 'w5': 0.5, 'w6': 0.6, 'w7': 0.7, 'q': 0.85}` and get the 0.539 accuracy in ramdom 1000 test.

## Next

Use the [Optuna](https://zh-cn.optuna.org/tutorial/configurations.html) tool to search the hyper-parameter.

## Reference

https://github.com/massachusett/hangman
https://github.com/chrisconley/hangman
