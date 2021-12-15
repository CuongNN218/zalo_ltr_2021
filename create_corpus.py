import json
import os
import re
from tqdm import tqdm
import argparse

def load_json(corpus_path):
    data = json.load(open(corpus_path))
    return data["items"]

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./data", type=str, help="path to training data")
    parser.add_argument("--save_dir", default="./generated_data", type=str, help="path to training data")
    args = parser.parse_args()
    os.makedirs(args.save_dir,exist_ok=True)
    cp = open(os.path.join(args.save_dir, "corpus.txt"), "w")
    corpus_path = os.path.join(args.data_dir, "legal_corpus.json")

    data = json.load(open(corpus_path))

    save_dict = {}
    co_f = open(os.path.join(args.save_dir, "cocondenser_data.json"), "w")
    count = 0
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
            
            # Save data for cocondenser 
            spans = [article_title]
            passages = re.split(r"\n[0-9]+\. |1\. ", article_text)
            for idx, p in enumerate(passages):
                if p != "":
                    article_full = article_title + ". " + p
                    article_full = article_full.replace("\n", " ")
                    spans.append(p)
            co_f.write("#".join(spans) + "\n")
            
            concat_id = law_id + "_" + article_id
            if concat_id not in save_dict:
                count += 1
                save_dict[concat_id] = {"title": article_title, "text": article_text}
    
    co_f.close()
    print(count)
    # exit()
    print("Create legal dict from raw data")
    with open(os.path.join(args.save_dir, "legal_dict.json"), "w") as outfile:
        json.dump(save_dict, outfile)
    print("Finish")
    corpus_path_train = os.path.join(args.data_dir, "train_question_answer.json")
    items = load_json(corpus_path_train)

    for item in tqdm(items):
        question = item["question"]
        cp.write(question + "\n")

    corpus_path_test = os.path.join(args.data_dir, "public_test_question.json")
    items = load_json(corpus_path_test)

    for item in tqdm(items):
        question = item["question"]
        cp.write(question + "\n")

    cp.close()
