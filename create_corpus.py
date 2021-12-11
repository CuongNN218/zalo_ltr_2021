import json
import os
from tqdm import tqdm
import argparse

def load_json(corpus_path):
    data = json.load(open(corpus_path))
    return data["items"]

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="", type=str, help="path to training data")
    args = parser.parse_args()

    cp = open("corpus.txt", "w")
    raw_data = args.data
    corpus_path = os.path.join(raw_data, "legal_corpus.json")

    data = json.load(open(corpus_path))

    save_dict = {}
    for law_article in tqdm(data):
        law_id = law_article["law_id"]
        law_articles = law_article["articles"]
        
        for sub_article in law_articles:
            article_id = sub_article["article_id"]
            article_title = sub_article["title"]
            article_text = sub_article["text"]
            article_full = article_title + ". " + article_text
            article_full = article_full.replace("\n", " ")
            cp.write(article_full + "\n")
            
            concat_id = law_id + "_" + article_id
            if concat_id not in save_dict:
                save_dict[concat_id] = {"title": article_title, "text": article_text}
                
    with open('legal_dict.json', 'w') as outfile:
        json.dump(save_dict, outfile)

    corpus_path_train = os.path.join(raw_data, "train_question_answer.json")
    items = load_json(corpus_path_train)

    for item in tqdm(items):
        question = item["question"]
        cp.write(question + "\n")

    corpus_path_test = os.path.join(raw_data, "public_test_question.json")
    items = load_json(corpus_path_test)

    for item in tqdm(items):
        question = item["question"]
        cp.write(question + "\n")

    cp.close()

