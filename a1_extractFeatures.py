import numpy as np
import argparse
import json
import re
import pandas as pd
import os
from datetime import timedelta
import time
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

# Provided wordlists.
FIRST_PERSON_PRONOUNS = {
    'i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours'}
SECOND_PERSON_PRONOUNS = {
    'you', 'your', 'yours', 'u', 'ur', 'urs'}
THIRD_PERSON_PRONOUNS = {
    'he', 'him', 'his', 'she', 'her', 'hers', 'it', 'its', 'they', 'them',
    'their', 'theirs'}
SLANG = {
    'smh', 'fwb', 'lmfao', 'lmao', 'lms', 'tbh', 'rofl', 'wtf', 'bff',
    'wyd', 'lylc', 'brb', 'atm', 'imao', 'sml', 'btw', 'bw', 'imho', 'fyi',
    'ppl', 'sob', 'ttyl', 'imo', 'ltr', 'thx', 'kk', 'omg', 'omfg', 'ttys',
    'afn', 'bbs', 'cya', 'ez', 'f2f', 'gtr', 'ic', 'jk', 'k', 'ly', 'ya',
    'nm', 'np', 'plz', 'ru', 'so', 'tc', 'tmi', 'ym', 'ur', 'u', 'sol', 'fml'}

# global variables 
feat_class = ['Left', 'Center', 'Right', 'Alt']
feat_data = {}
warr = pd.read_csv('/u/cs401/Wordlists/Ratings_Warriner_et_al.csv')
bgl = pd.read_csv('/u/cs401/Wordlists/BristolNorms+GilhoolyLogie.csv')

def extract1(comment):
    ''' This function extracts features from a single comment

    Parameters:
        comment : string, the body of a comment (after preprocessing)

    Returns:
        feats : numpy Array, a 173-length vector of floating point features (only the first 29 are expected to be filled, here)
    '''    
    feats = np.zeros(173)

    if comment:
        # patterns for first 14 features
        patterns = [
        # 1. Number of words in uppercase (≥ 3 letters long)
        (r'\b[A-Z]{3,}/[A-Z]{2,}\b', 0),
        # 2. Number of first-person pronouns
        (r'(\b('+ '|'.join(FIRST_PERSON_PRONOUNS) + ')/PRP)', re.I),
        # 3. Number of second-person pronouns
        (r'(\b('+ '|'.join(SECOND_PERSON_PRONOUNS) + ')/PRP)', re.I),
        # 4. Number of third-person pronouns
        (r'(\b('+ '|'.join(THIRD_PERSON_PRONOUNS) + ')/PRP)', re.I),
        # 5. Number of coordinating conjunctions
        (r'(/CC\b)', 0),
        # 6. Number of past-tense verbs
        (r'(/VBD\b)', 0),
        # 7. Number of future-tense verbs
        (r"(gonna/|will/|'ll/)|(be/VBP\sgo/VBG\sto/TO\s\w+/VB)", re.I),
        # 8. Number of commas
        (r'(\s?,/,\s?)', 0),
        # 9. Number of multi-character punctuation tokens 
        (r"[!?.:;\\\"',\(\)`\-\[\]]{2,}/", 0),
        # 10. Number of common nouns
        (r'(/NN|/NNS)\b', 0),
        # 11. Number of proper nouns
        (r'(/NNP|/NNPS)\b', 0),
        # 12. Number of adverbs
        (r'(/RB|/RBR|/RBS)\b', 0),
        # 13. Number of wh- words
        (r'(/WDT|/WP|/WP\$|/WRB)\b', 0),
        # 14. Number of slang acronyms
        (r'(\b(' + '|'.join(SLANG) + ')/.+)', re.I)
        ]
        
        for i, pattern in enumerate(patterns):
            feats[i] = len(re.findall(pattern[0], comment, pattern[1]))

        # 15. Average length of sentences, in tokens
        # ignore empty split results
        sentences = [s for s in comment.split('\n') if s]
        # make sure there is no extra newline character
        words = [t for t in comment.split() if t!='/n']
        if(len(sentences) > 0):
            feats[14] = len(words)/len(sentences)
        # 16. Average length of tokens, excluding punctuation-only tokens, in characters 
        removed_pun = [re.sub(r'/((_*-*[A-Z]{2,}-*\S*)|[^/]\W+)$', '', token) for token in words if not re.findall(r"[!?.:;\\\"',\(\)`\-\[\]]+/", token)]
        if len(removed_pun) > 0:
            removed_pun_len = [len(t) for t in removed_pun]
            feats[15] = sum(removed_pun_len)/len(removed_pun_len)
        # 17. Number of sentences.
        feats[16] = len(sentences)
    
    # 18. Average of AoA (100-700) from Bristol, Gilhooly, and Logie norms
    feats[17] = bgl.get('AoA (100-700)').mean()
    # 19. Average of IMG from Bristol, Gilhooly, and Logie norms
    feats[18] = bgl.get('IMG').mean()
    # 20. Average of FAM from Bristol, Gilhooly, and Logie norms
    feats[19] = bgl.get('FAM').mean()
    # 21. Standard deviation of AoA (100-700) from Bristol, Gilhooly, and Logie norms 
    feats[20] = bgl.get('AoA (100-700)').std()
    # 22. Standard deviation of IMG from Bristol, Gilhooly, and Logie norms
    feats[21] = bgl.get('IMG').std()
    # 23. Standard deviation of FAM from Bristol, Gilhooly, and Logie norms
    feats[22] = bgl.get('FAM').std()
    # 24. Average of V.Mean.Sum from Warringer norms
    feats[23] = warr.get('V.Mean.Sum').mean()
    # 25. Average of A.Mean.Sum from Warringer norms
    feats[24] = warr.get('A.Mean.Sum').mean()
    # 26. Average of D.Mean.Sum from Warringer norms
    feats[25] = warr.get('D.Mean.Sum').mean()
    # 27. Standard deviation of V.Mean.Sum from Warringer norms
    feats[26] = warr.get('V.Mean.Sum').std()
    # 28. Standard deviation of A.Mean.Sum from Warringer norms
    feats[27] = warr.get('A.Mean.Sum').std()
    # 29. Standard deviation of D.Mean.Sum from Warringer norms
    feats[28] = warr.get('D.Mean.Sum').std()

    return feats
    
    
def extract2(feats, comment_class, comment_id):
    ''' This function adds features 30-173 for a single comment.

    Parameters:
        feats: np.array of length 173
        comment_class: str in {"Alt", "Center", "Left", "Right"}
        comment_id: int indicating the id of a comment

    Returns:
        feats : numpy Array, a 173-length vector of floating point features (this 
        function adds feature 30-173). This should be a modified version of 
        the parameter feats.
    '''
    if comment_id in feat_data[comment_class][0]:
        # get the feature index based on the comment_id
        feat_idx = feat_data[comment_class][0].index(comment_id)
        # extract feature from the feats files based on the commen_class and the index
        feats_additional = feat_data[comment_class][1][feat_idx]
        feats[29:] = feats_additional
    else:
        print("ID: %s is not found in class: %s" % (comment_id, comment_class))
    return feats

def add_sentiment_feature(comment, positive_list, negative_list):
    ''' This function is for sentimental features.

    Parameters:
        comment: comment that need to be preocessed
        positive_list: the list of positive words
        negative_list: the list of negative words

    Returns:
        feat_senti : numpy Array, length 2 vector
    '''
    feat_senti = np.zeros(2)
    pos = 0
    neg = 0
    # remove punctuation only tokens and the token flag
    words = [t for t in comment.split() if t!='/n']
    removed_pun = [re.sub(r'/((_*-*[A-Z]{2,}-*\S*)|[^/]\W+)$', '', token) for token in words if not re.findall(r"[!?.:;\\\"',\(\)`\-\[\]]+/", token)]
    for i in removed_pun:
        if i in positive_list:
            pos += 1
        elif i in negative_list:
            neg += 1
    feat_senti[0] = pos
    feat_senti[1] = neg
    return feat_senti

def LDA(data, outf):
    ''' This function is for LDA topic modelling.

    Parameters:
        data: comments data including body and class
        outf: the output file name

    Returns:
        feats : numpy Array, can be used as feature vector
    '''
    vectorizer = CountVectorizer(stop_words='english')
    comments = [c['body'] for c in data]
    labels = np.array([[feat_class.index(c['cat'])] for c in data])
    tf = vectorizer.fit_transform(comments)
    topic_modeller = LatentDirichletAllocation(n_components=10, batch_size=100, random_state=2)
    topic_modeller.fit(tf)
    feature_names = vectorizer.get_feature_names()

    topic_dict = {}
    no_top_words = 10
    for topic_idx, topic in enumerate(topic_modeller.components_):
        topic_dict["Topic %d words" % (topic_idx)]= ['{}'.format(feature_names[i])
                        for i in topic.argsort()[:-no_top_words - 1:-1]]
        topic_dict["Topic %d weights" % (topic_idx)]= ['{:.1f}'.format(topic[i])
                        for i in topic.argsort()[:-no_top_words - 1:-1]]
    print(pd.DataFrame(topic_dict))

    features = np.concatenate((topic_modeller.fit_transform(tf), labels), axis=1)
    return features


def main(args):
    start = time.time()
    data = json.load(open(args.input))
    feats = np.zeros((len(data), 173+1))
    feats_with_senti = np.zeros((len(data), 173+1+2))

    # load feature data
    feats_dir = os.path.join(args.a1_dir, 'feats')  
    for f in feat_class:
        feat_data[f] = [[ID.strip() for ID in open(os.path.join(feats_dir, f + '_IDs.txt')).readlines()],
            np.load(os.path.join(feats_dir, f + '_feats.dat.npy'))]

    # see https://gist.github.com/mkulakowski2/4289437 as reference for positive word list
    # load positive word
    with open('positive-words.txt', 'r') as file:
        pos_list = [w.replace('\n', '').strip() for w in file.readlines()]
    # see https://gist.github.com/mkulakowski2/4289441 as reference for negative word list
    # load negative word
    with open('negative-words.txt', 'r') as file:
        neg_list = [w.replace('\n', '').strip() for w in file.readlines()]

    for i, comment in enumerate(data):
        # use extract1 to find the first 29 features
        feat_29 = extract1(comment['body'])
        # extract2 to copy LIWC features (features 30-173)
        feats[i,:-1] = extract2(feat_29, comment['cat'], comment['id'])
        # the last one is the integer representing class
        feats[i, 173] = feat_class.index(comment['cat'])
        # bonus for sentimental feature(ie. positive, negative)
        feats_with_senti[i,:173] = feats[i,:173]
        feats_with_senti[i,173:175] = add_sentiment_feature(comment['body'], pos_list, neg_list)
        feats_with_senti[i,175] = feats[i,173]
        
    np.savez_compressed(args.output, feats)
    np.savez_compressed(args.output.replace('.', '_with_sentimental.'), feats_with_senti)
    elapsed = (time.time() - start)
    print('Finish feature extractions(including sentimental features) at: ' + str(timedelta(seconds=elapsed)) + ' mins.\n')

    feats_LDA = np.zeros((len(data), 173+1+2+10))
    start = time.time()
    # generating features for bonus. Using LDA as features
    LDA_outf = args.output.replace('.', '_LDA.')
    LDA_feats = LDA(data, LDA_outf)
    feats_LDA[:,:175] = feats_with_senti[:,:175]
    # LDA_feats already included  the label
    feats_LDA[:,175:] = LDA_feats
    np.savez_compressed(LDA_outf, feats_LDA)
    elapsed = (time.time() - start)
    print('Finish LDA topic modelling at: ' + str(timedelta(seconds=elapsed)) + ' mins.\n')
       
if __name__ == "__main__": 

    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("-i", "--input", help="The input JSON file, preprocessed as in Task 1", required=True)
    parser.add_argument("-p", "--a1_dir", help="Path to csc401 A1 directory. By default it is set to the cdf directory for the assignment.", default="/u/cs401/A1/")
    args = parser.parse_args()        

    main(args)
