import os
import json
import pickle
from re import S
from unicodedata import name
import numpy as np
from tqdm import tqdm
from rank_bm25 import *
from utils import bm25_tokenizer, calculate_f2, load_json
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--stopword", default="manual", type=str)
    parser.add_argument("--model_path", default="saved_model/bm25_Plus_04_06_model_full_manual_stopword", type=str)
    parser.add_argument("--data_path", default="zac2021-ltr-data", type=str, help="path to input data")
    parser.add_argument("--save_pair_path", default="pair_data/", type=str, help="path to save pair sentence directory")
    args = parser.parse_args()

    train_path = os.path.join(args.data_path, "train_question_answer.json")
    training_data = json.load(open(train_path))

    training_items = training_data["items"]

    with open(args.model_path, "rb") as bm_file:
        bm25 = pickle.load(bm_file)
    with open("saved_model/doc_refers_saved", "rb") as doc_refer_file:
        doc_refers = pickle.load(doc_refer_file)

    doc_data = json.load(open(os.path.join("legal_dict.json")))

    save_pairs = []

    total_f2 = 0
    total_precision = 0
    total_recall = 0
    k = len(training_items)
    top_n = 50
    for idx, item in tqdm(enumerate(training_items)):
        if idx >= k:
            continue

        question_id = item["question_id"]
        question = item["question"]
        relevant_articles = item["relevant_articles"]
        actual_positive = len(relevant_articles)
        
        tokenized_query = bm25_tokenizer(question)
        doc_scores = bm25.get_scores(tokenized_query)
        # top_pred = np.sort(doc_scores)[-3:]
        # #top_pred = np.unique(top_pred, axis=0)
        # predictions = []
        # for top in top_pred:
        #     predictions.append(np.where(doc_scores == top)[0][0])
        predictions = np.argpartition(doc_scores, len(doc_scores) - top_n)[-top_n:]
        # if doc_scores[predictions[1]] - doc_scores[predictions[0]] >= 2.6:
        #      predictions = [predictions[1]]
        
        for article in relevant_articles:
            save_dict = {}
            save_dict["question"] = question
            concat_id = article["law_id"] + "_" + article["article_id"]
            save_dict["document"] = doc_data[concat_id]["title"] + " " + doc_data[concat_id]["text"]
            save_dict["relevant"] = 1
            save_pairs.append(save_dict)
        # print(question)
        # print(relevant_articles)

        true_positive = 0
        false_positive = 0
        for idx, idx_pred in enumerate(predictions):
            pred = doc_refers[idx_pred]
                
            #print(pred, doc_scores[idx_pred])
            #if doc_scores[idx_pred] >= 20:
            check = 0
            for article in relevant_articles:
                if pred[0] == article["law_id"] and pred[1] == article["article_id"]:
                    true_positive += 1
                    check += 1
                    #print(doc_data[pred[0] + "_" + pred[1]])
                else:
                    false_positive += 1
            
            if check == 0:
                save_dict = {}
                save_dict["question"] = question
                concat_id = pred[0] + "_" + pred[1]
                save_dict["document"] = doc_data[concat_id]["title"] + " " + doc_data[concat_id]["text"]
                save_dict["relevant"] = 0
                save_pairs.append(save_dict)
                    
        if true_positive + false_positive == 0:
            precision = 0
        else:
            precision = true_positive/(true_positive + false_positive)
        recall = true_positive/actual_positive
        f2 = calculate_f2(precision, recall)
        total_precision += precision
        total_recall += recall
        total_f2 += f2
        
    print(f"Average F2: {total_f2/k}")
    print(f"Average Precision: {total_precision/k}")
    print(f"Average Recall: {total_recall/k}")



    save_path = args.save_pair_path
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, f"save_pairs_top{top_n}"), "wb") as pair_file:
        pickle.dump(save_pairs, pair_file)
    print(len(save_pairs))