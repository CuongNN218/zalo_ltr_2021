import os
import json
import pickle
import numpy as np
from tqdm import tqdm
from rank_bm25 import *
import argparse
from utils import bm25_tokenizer, calculate_f2
# from config import Config

class Config:
    data_path = "zac2021-ltr-data"
    save_bm25 = "saved_model"
    top_k_bm25 = 2
    bm25_k1 = 0.4
    bm25_b = 0.6

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # load document to save running time, 
    # must run 1 time if we change pre-process step
    parser.add_argument("--load_docs", action="store_false")
    parser.add_argument("--num_eval", default=500, type=str)
    args = parser.parse_args()
    cfg = Config()
    
    save_path = cfg.save_bm25
    os.makedirs(save_path, exist_ok=True)

    raw_data = cfg.data_path
    corpus_path = os.path.join(raw_data, "legal_corpus.json")

    data = json.load(open(corpus_path))

    if args.load_docs:
        print("Process documents")
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
                    
                tokens = bm25_tokenizer(article_full)
                documents.append(tokens)
                doc_refers.append([law_id, article_id, article_full])
        
        with open(os.path.join(save_path, "documents_manual"), "wb") as documents_file:
            pickle.dump(documents, documents_file)
        with open(os.path.join(save_path,"doc_refers_saved"), "wb") as doc_refer_file:
            pickle.dump(doc_refers, doc_refer_file)
    else:
        with open(os.path.join(save_path, "documents_manual"), "rb") as documents_file:
            documents = pickle.load(documents_file)
        with open(os.path.join(save_path,"doc_refers_saved"), "rb") as doc_refer_file:
            doc_refers = pickle.load(doc_refer_file)
            

    # Grid_search, evaluate on training question
    # raw_data = "zac2021-ltr-data"
    train_path = os.path.join(raw_data, "train_question_answer.json")
    data = json.load(open(train_path))
    items = data["items"]
    print(len(items))

    bm25 = BM25Plus(documents, k1=cfg.bm25_k1, b=cfg.bm25_b)
    with open(os.path.join(save_path, "bm25_Plus_04_06_model_full_manual_stopword"), "wb") as bm_file:
        pickle.dump(bm25, bm_file)
        
    total_f2 = 0
    total_precision = 0
    total_recall = 0
    
    k = args.num_eval
    for idx, item in tqdm(enumerate(items)):
        if idx >= k:
            continue

        question_id = item["question_id"]
        question = item["question"]
        relevant_articles = item["relevant_articles"]
        actual_positive = len(relevant_articles)
        
        tokenized_query = bm25_tokenizer(question)
        doc_scores = bm25.get_scores(tokenized_query)
        
        # Get top N
        # N large -> reduce precision, increase recall
        # N small -> increase precision, reduce recall
        predictions = np.argpartition(doc_scores, len(doc_scores) - cfg.top_k_bm25)[-cfg.top_k_bm25:]
        
        # Trick to balance precision and recall
        if doc_scores[predictions[1]] - doc_scores[predictions[0]] >= 2.7:
            predictions = [predictions[1]]

        true_positive = 0
        false_positive = 0
        for idx, idx_pred in enumerate(predictions):
            pred = doc_refers[idx_pred]
            # print(pred, doc_scores[idx_pred])
            
            # Remove prediction with too low score: 20
            if doc_scores[idx_pred] >= 20:
                for article in relevant_articles:
                    if pred[0] == article["law_id"] and pred[1] == article["article_id"]:
                        true_positive += 1
                    else:
                        false_positive += 1
                    
        precision = true_positive/(true_positive + false_positive + 1e-20)
        recall = true_positive/actual_positive
        f2 = calculate_f2(precision, recall)
        total_precision += precision
        total_recall += recall
        total_f2 += f2
        
    print(f"Average F2: \t\t\t\t{total_f2/k}")
    print(f"Average Precision: {total_precision/k}")
    print(f"Average Recall: {total_recall/k}\n")