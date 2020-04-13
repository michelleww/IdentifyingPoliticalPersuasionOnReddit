This is the introduction for bonus part.

1. Add two features for negative and positive word count. (in a1_extractFeatures.py)
Note that the negative-words.txt and positive-words.txt must be placed under the same directory as a1_extractFeatures.py to run
Reference:
https://gist.github.com/mkulakowski2/4289437
https://gist.github.com/mkulakowski2/4289441
Compute the number of positive words in a comment and the number of negative words in a comment.
Then add these two features to the original feature list to form a new feature list with length 146.
These features will be stored in a new npz file named 'feats_with_sentimental.npz'

2. Add the Latent Dirichlet Allocation. (in a1_extractFeatures.py)
Use the following page as a tutorial
Reference: 
https://ourcodingclub.github.io/2018/12/10/topic-modelling-python.html
Print the topic information in the console and output the new features including the previous features to a new file named 'feats_LDA.npz'
