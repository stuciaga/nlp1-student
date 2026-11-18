import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

import part2  # noqa: E402
from datasets import load_dataset  # noqa: E402
from part2.pruned_embeddings import PrunedEmbeddings  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--out", required=True, help="CSV file to write")
parser.add_argument("--sample", type=int, default=0, help="first N test reviews only (quick test)")
args = parser.parse_args()


def clean(text):
    """Same cleaning as in training: drop HTML tags, normalise whitespace."""
    return " ".join(re.sub(r"</?(br|i|em|hr)\s*/?>", " ", text).split())


emb_model = PrunedEmbeddings(Path(__file__).resolve().parent / "results" / "imdb" / "embeddings")
tokenizer = part2.clslib.BasicTokenizer(
    lowercase=True, remove_punct=False, remove_stopwords=False, language="english", vocab=emb_model.vocab,
)

test = pd.DataFrame(load_dataset("mteb/imdb")["test"])   # no de-duplication: one row per review_id
test["text"] = test["text"].map(clean)
if args.sample:
    test = test.head(args.sample)
texts = test["text"].tolist()
labels = test["label"].to_numpy()

# we will be using textstat as a way to otain some descriptive features of text; 
# we encourage you to look for things that are meaningful to your own task/dataset
# this is just to demo the kind of thing you can do 
# import textstat  

# import pandas as pd

table = []
systems = dict()
items = dict()

# # each model uses a certain tokenizer, and we don't want to accidentally use the wrong one
# name_to_tokenizer = {
#     'embed-and-pool': tokenizer_for_bow, 
#     'lstm': tokenizer_for_rnn
# }

for model_family, model_data in {"embed-and-pool": "part2/results/imdb/embed-and-pool-trainable/baseline", "lstm": "part2/results/imdb/lstm/baseline"}.items(): # for all our trained classifiers
    for run, seed in enumerate([0, 3, 5]):
        # we get the devtest set that we kep aside for each run
        # both texts and labels
        # texts = [text for text, label in my_test_sets[run]['devtest']]
        # labels = np.array([label for text, label in my_test_sets[run]['devtest']], dtype=int)
        
        for fold in range(5):            
            # load our model
            trained_nn = part2.experiment.load(f"{model_data}/seed-{seed}/fold-{fold}/model", emb_model, device='cuda')

            # test it on the devtest
            logprobs = part2.experiment.predict_logprobs(
                trained_nn, 
                tokenizer, 
                texts, 
                64, 
                'cuda'
            )
            preds = np.argmax(logprobs, -1)
            conf = np.exp(logprobs).max(-1)
    
            # then featurize the texts and compute the correctness flags
            for i in range(len(preds)):                
                                
                # find an int id for each system
                sid = systems.get(model_family, None)                
                if sid is None:
                    sid = len(systems)
                    systems[model_family] = sid
                # find an int id for each item
                item = i                 
                item_id = items.get(item, None)                
                if item_id is None:
                    item_id = len(items)
                    items[item] = item_id

                # text, pred, and label
                x = texts[i]
                y_ = preds[i]
                y = labels[i]

                # put it all in a table, along with some features
                table.append([
                    model_family, 
                    item,
                    sid, 
                    item_id, 
                    run, 
                    fold, 
                    y, 
                    y_, 
                    int(y == y_), # correctness flag
                    conf[i],
                    # some features
                    # textstat.lexicon_count(x),
                    # textstat.flesch_reading_ease(x),
                    # textstat.reading_time(x),
                    # textstat.flesch_kincaid_grade(x)
                ])

df = pd.DataFrame(table, columns=['system', 'item', 'system_id', 'item_id', 'run', 'fold', 'label', 'pred', 'correct', 'confidence'])

df.to_csv(args.out, index = False)