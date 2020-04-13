import sys
import argparse
import os
import json
import re
import spacy
import html
from datetime import timedelta
import time


nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
sentencizer = nlp.create_pipe("sentencizer")
nlp.add_pipe(sentencizer)


def preproc1(comment , steps=range(1, 5)):
    ''' This function pre-processes a single comment

    Parameters:                                                                      
        comment : string, the body of a comment
        steps   : list of ints, each entry in this list corresponds to a preprocessing step  

    Returns:
        modComm : string, the modified comment 
    '''
    modComm = comment
    # check empty string
    if not modComm:
        return modComm
    if 1 in steps:  # replace newlines with spaces
        modComm = re.sub(r"\n{1,}", " ", modComm)
    if 2 in steps:  # unescape html
        modComm = html.unescape(modComm)
    if 3 in steps:  # remove URLs
        modComm = re.sub(r"(http|www)\S+", "", modComm)
    if 4 in steps:  # remove duplicate spaces
        modComm = re.sub(r'(\s{2,})', ' ', modComm)

    utt = nlp(modComm)
    com_list = []
    for sent in utt.sents:
        sent_list = []
        for token in sent:
            temp = token
            if str(token.lemma_).startswith('-'):
                temp = token.text
            else:
                temp = token.lemma_
            # Write "/POS" after each token.
            sent_list.append(temp + '/' + token.tag_)
        # Insert "\n" between sentences.
        com_list.append(' '.join(sent_list)+'\n')
    modComm = ''.join(com_list)
    return modComm


def main(args):
    allOutput = []
    indir = args.a1_dir
    for subdir, dirs, files in os.walk(indir):
        for file in files:
            if file not in ['Center', 'Right', 'Left', 'Alt']:
                continue
            fullFile = os.path.join(subdir, file)
            print( "Processing " + fullFile)

            data = json.load(open(fullFile))
            starting_idx = args.ID[0] % len(data)
            # select appropriate args.max lines
            for i in range(starting_idx, starting_idx+args.max):
                if i == len(data):
                    print('Finish Loading %d data, reached the end of the %s data.\n' %(i-starting_idx, file))
                    break
                elif i- starting_idx+1 == args.max:
                    print('Finish Loading %d data. Reached the maximum amount of data that is allowed for loading.\n' %args.max)
                # read line
                j_line = json.loads(data[i])
                # need id, cat, body fileds
                new = {}
                new['cat'] = file
                new['id'] = j_line['id']
                # replace the 'body' field with the pre-processed text
                new['body'] = preproc1(j_line['body'])
                allOutput.append(new)
            

    fout = open(args.output, 'w')
    fout.write(json.dumps(allOutput))
    fout.close()


if __name__ == "__main__":
    start = time.time()
    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument('ID', metavar='N', type=int, nargs=1,
                        help='your student ID')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("--max", type=int, help="The maximum number of comments to read from each file", default=10000)
    parser.add_argument("--a1_dir", help="The directory for A1. Should contain subdir data. Defaults to the directory for A1 on cdf.", default='/u/cs401/A1')
    
    args = parser.parse_args()

    if (args.max > 200272):
        print( "Error: If you want to read more than 200,272 comments per file, you have to read them all." )
        sys.exit(1)
    
    main(args)
    elapsed = (time.time() - start)
    print(str(timedelta(seconds=elapsed)))
