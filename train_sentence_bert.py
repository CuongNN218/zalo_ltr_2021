from sentence_transformers import SentenceTransformer, models, InputExample, losses, LoggingHandler
from torch.utils.data import DataLoader
import pickle
from sentence_transformers import evaluation
import logging
import argparse
import os
#### Just some code to print debug information to stdout

def load_pair_data(pair_data_path):
    with open(pair_data_path, "rb") as pair_file:
        pairs = pickle.load(pair_file)
    return pairs

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model", default="", type=str, help="path to your language model")
    parser.add_argument("--max_seq_length", default=256, type=int, help="maximum sequence length")
    parser.add_argument("--pair_data_path", type=str, default="", help="path to saved pair data")
    parser.add_argument("--round", default=1, type=str, help="training round ")
    parser.add_argument("--num_val", default=2500, type=int, help="number of eval data")
    parser.add_argument("--epochs", default=5, type=int, help="Number of training epochs")
    parser.add_argument("--saved_model", default="", type=str, help="path to savd model directory.")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--lr", type=float, default=1e-5, help="learning rate for training")
    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])
    if round == 1:
        print(f"Training round 1")
        word_embedding_model = models.Transformer(args.pretrained_model, max_seq_length=args.max_seq_length)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
        print(model)
    else:
        print("Training round 2")
        model = SentenceTransformer(args.pretrained_model)
        print(model)

    save_pairs = load_pair_data(args.pair_data_path)
    print(f"There are {len(save_pairs)} pair sentences.")
    train_examples = []
    sent1 = []
    sent2 = []
    scores = []
    num_train = len(save_pairs) - args.num_val

    for idx, pair in enumerate(save_pairs):
        relevant = float(pair["relevant"])
        question = pair["question"]
        document = pair["document"]
        if idx <= num_train:
            example = InputExample(texts=[question, document], label=relevant)
            train_examples.append(example)
        else:
            sent1.append(question)
            sent2.append(document)
            scores.append(relevant)

    print("Number of sample for training: ", len(train_examples))

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=args.batch_size)
    train_loss = losses.ContrastiveLoss(model)

    output_path = args.saved_model
    os.makedirs(output_path, exist_ok=True)

    evaluator = evaluation.BinaryClassificationEvaluator(sent1, sent2, scores)
    
    model.fit(train_objectives=[(train_dataloader, train_loss)], 
            epochs=args.epochs, 
            warmup_steps=1000,
            optimizer_params={'lr': 1e-5},
            save_best_model=True,
            evaluator=evaluator,
            evaluation_steps=args.num_eval,
            output_path=output_path,
            use_amp=True,
            show_progress_bar=True)

