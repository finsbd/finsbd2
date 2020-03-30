import json
import math
from pathlib import Path
from typing import Dict, List


def mean(list_: List[float]):
    """ Compute mean of a list of float and ignore nan value. """

    # ignore nan value
    list_ = [e for e in list_ if not math.isnan(e)]
    if len(list_) == 0:
        # all values were nan or list was empty
        return float("nan")
    else:
        # mean
        return sum(list_) / len(list_)


def evaluate(true: Dict[str, List[dict]],
             pred: Dict[str, List[dict]]):

    """ Compute recall, precision, f1-score for each class and also for
    substask1 and subtask2. """

    scores = {}

    for key in ("sentence", "item", "list", "item1", "item2", "item3", "item4"):

        true_pairs = [(x["start"], x["end"]) for x in true[key]]
        pred_pairs = [(x["start"], x["end"]) for x in pred[key]]

        if true_pairs or pred_pairs:

            false_negative = true_pairs.copy()
            false_positive = pred_pairs.copy()

            # get true positive, both start and end index matches
            tp = 0
            for pair in pred_pairs:
                if pair in true_pairs:
                    tp += 1
                    false_negative.remove(pair)
                    false_positive.remove(pair)

            # false negative and false positive
            fn = len(false_negative)
            fp = len(false_positive)

            # sanity check
            assert len(true_pairs) == len(false_negative) + tp
            assert len(pred_pairs) == len(false_positive) + tp

            # precision
            p = tp / (tp + fp) if (tp + fp) else 0

            # recall
            r = tp / (tp + fn) if (tp + fn) else 0

            # f1 score
            f1_score = 2 * (p * r) / (p + r) if (p + r) else 0

        else:
            # return nan which means label does not exist in the document
            p = float("nan")
            r = float("nan")
            f1_score = float("nan")

        scores[key] = {
            "precision": p,
            "recall": r,
            "f1": f1_score,
        }

    # subtask 1 mean average f1
    subtask1 = ("sentence", "item", "list")
    scores["subtask1"] = {"f1": mean([scores[k]["f1"] for k in subtask1])}

    # subtask 2 mean average f1
    subtask2 = ("item1", "item2", "item3", "item4")
    scores["subtask2"] = {"f1": mean([scores[k]["f1"] for k in subtask2])}

    return scores


def main(lang, split="train", dev_ratio=0.3):
    assert lang in ("en", "fr")
    print(f"\nEvaluation (lang={lang})")

    # load data
    data_dir = Path(f"data/finsbd2_{split}/{lang}")
    list_ann = list(data_dir.glob("*.finsbd2.json"))
    total = len(list_ann)

    # only evaluate on dev (if you plan to train a model)
    nb_dev = int(round(dev_ratio * total, 0))
    train = list(range(0, total - nb_dev))
    dev = list(range(total - nb_dev, total))

    # compute f1-score between ground-truth and predictions
    all_scores = list()
    for i in dev:
        ann_path = list_ann[i]
        pred_path = ann_path.with_suffix(".pred.json")
        assert pred_path.is_file()
        print(f"\n\t[{i+1}/{total}] processing '{pred_path.name}'")

        true = json.load(ann_path.open("r"))
        pred = json.load(pred_path.open("r"))

        # compute scores for document
        scores = evaluate(true, pred)
        all_scores.append(scores)

    # mean average of f1-score of all PDFs in dev set
    print(f"\n\tFinal macro f1-score (lang={lang}):")
    macro_score = {}
    for key in (
        "sentence",
        "list",
        "item",
        "item1",
        "item2",
        "item3",
        "item4",
        "subtask1",
        "subtask2",
    ):
        macro_score[key] = mean([scores[key]["f1"] for scores in all_scores])
        print(f"\t\t- {key}: {round(macro_score[key], 3)}")


if __name__ == "__main__":
    main("en")
    main("fr")
