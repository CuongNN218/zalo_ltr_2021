import numpy as np
import json
import torch
from tqdm import tqdm
import string
from underthesea import sent_tokenize, word_tokenize
from rank_bm25 import *
import argparse
import os
import pickle
import glob
from utils import bm25_tokenizer

from sentence_transformers import SentenceTransformer, util

def encode_question(question_data, models):
    print("Start encoding questions.")
    question_embs = []
    for model in models:
        emb_quest_dict = {}
        for _, item in tqdm(enumerate(question_data)):
            question_id = item["question_id"]
            question = item["question"]
            emb_quest_dict[question_id] = model.encode(question)
        question_embs.append(emb_quest_dict)
    return question_embs

def load_encoded_legal_corpus(legal_data_path):
    print("Start loading legal corpus.")
    with open(legal_data_path, "rb") as f1:
        emb_legal_data = pickle.load(f1)
    return emb_legal_data

def load_bm25(bm25_path):
    with open(bm25_path, "rb") as bm_file:
        bm25 = pickle.load(bm_file)
    return bm25

def load_models(root, model_names):
    models = []
    for model_path in tqdm(model_names):
        model_path = os.path.join(args.saved_model, model_path)
        models.append(SentenceTransformer(model_path))
    return models

def load_question_json(question_path):
    question_path = glob.glob(f"{question_path}/*.json")[0]
    question_data = json.load(open(question_path))
    return question_data

if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data", type=str)
    parser.add_argument("--saved_model", default="saved_model", type=str)
    parser.add_argument("--bm25_path", default="saved_model/bm25_Plus_04_06_model_full_manual_stopword", type=str)
    args = parser.parse_args()

    # define path to model
    model_paths = ["phobert_pretrained_fulldata_continue_large_contrastive_3_eval",
                    "condenser_large_30eps_5eps_seed42_cls_round2",
                    "vibert_pretrained_fulldata_50_e5_b32_round_2"]

    print("Start loading model.")
    models = load_models(args.saved_model, model_paths)
    print("Number of pretrained models: ", len(models))

    # load question from json file
    question_data = load_question_json(args.data)
    items = question_data["items"]
    print("Number of questions: ", len(items))
    
    # load bm25 model 
    bm25 = load_bm25(args.bm25_path)
    
    # load corpus to search
    with open("saved_model/doc_refers_saved", "rb") as doc_refer_file:
        doc_refers = pickle.load(doc_refer_file)

    doc_data = json.load(open("legal_data/legal_dict.json"))

    # load pre encoded for legal corpus
    emb_legal_data = load_encoded_legal_corpus('enbedding_doc_data_ensemble_v3.pkl')

    # encode question for query
    question_embs = encode_question(items)

    # define top n for compare and range of score
    top_n = 61425
    range_score = 2.3

    pred_list = []

    print("Start calculating results.")
    for idx, item in tqdm(enumerate(items)):
        question_id = item["question_id"]
        question = item["question"]
        
        tokenized_query = bm25_tokenizer(question)
        doc_scores = bm25.get_scores(tokenized_query)

        weighted = [0.2, 0.45, 0.35]
        cos_sim = []

        for idx_2, model in enumerate(models):
            emb1 = question_embs[idx_2][question_id]
            emb2 = emb_legal_data[idx_2]
            scores = util.cos_sim(emb1, emb2)
            cos_sim.append(weighted[idx_2] * scores)
        cos_sim = torch.cat(cos_sim, dim=0)
        
        cos_sim = torch.sum(cos_sim, dim=0).squeeze(0).numpy()
        new_scores = doc_scores * cos_sim
        max_score = np.max(new_scores)

        predictions = np.argpartition(new_scores, len(new_scores) - top_n)[-top_n:]
        new_scores = new_scores[predictions]
        
        new_predictions = np.where(new_scores >= (max_score - range_score))[0]
        map_ids = predictions[new_predictions]
        new_scores = new_scores[new_scores >= (max_score - range_score)]

        pred_dict = {}
        pred_dict["question_id"] = question_id
        pred_dict["relevant_articles"] = []
        
        # post processing character error
        dup_ans = []
        for idx, idx_pred in enumerate(map_ids):
            pred = doc_refers[idx_pred]
            law_id = pred[0]
            article_id = pred[1]
            
            if law_id.endswith("nd-cp"):
                law_id = law_id.replace("nd-cp", "nđ-cp")
            if law_id.endswith("nđ-"):
                law_id = law_id.replace("nđ-", "nđ-cp")
            if law_id.endswith("nð-cp"):
                law_id = law_id.replace("nð-cp", "nđ-cp")
            if law_id == "09/2014/ttlt-btp-tandtc-vksndtc":
                law_id = "09/2014/ttlt-btp-tandtc-vksndtc-btc"
            if law_id + "_" + article_id not in dup_ans:
                dup_ans.append(law_id + "_" + article_id)
                pred_dict["relevant_articles"].append({"law_id": law_id, "article_id": article_id})
        pred_list.append(pred_dict)
    
    # extract result
    with open('/result/submission.json', 'w') as outfile:
        json.dump(pred_list, outfile)