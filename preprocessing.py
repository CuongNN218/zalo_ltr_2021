import os
import json
from tqdm import tqdm
import pickle
from utils import bm25_tokenizer
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data", default="zac2021-ltr-data", type=str, help="path to raw data")
    parser.add_argument("--save_path", default="generated_data", type=str, help="path to save doc refer.")
    args = parser.parse_args()

    data = json.load(open(os.path.join(args.raw_data, "legal_corpus.json")))
    print("=======================")
    print("Start create doc refer.")
    documents = []
    doc_refers = []
    for law_article in tqdm(data):
        law_id = law_article["law_id"]
        law_articles = law_article["articles"]
        
        for sub_article in law_articles:
            article_id = sub_article["article_id"]
            article_title = sub_article["title"]
            article_text = sub_article["text"]
            article_full = article_title + " " + article_text
            doc_refers.append([law_id, article_id, article_full])
    os.makedirs(args.save_path, exist_ok=True)
    with open(os.path.join(args.save_path,"doc_refers_saved.pkl"), "wb") as doc_refer_file:
        pickle.dump(doc_refers, doc_refer_file)
    print("Created Doc Data.")