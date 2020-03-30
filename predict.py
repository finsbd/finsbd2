import json
import spacy
from pathlib import Path


def predict_boundaries(pdf_text, nlp):
    """ Segment text into sentences using spacy rule-based method (fast).
    Spacy conveniently provides off-the-shelf sentence segmentation method
    that preserves original character indexation.
    """

    # sentence segmentation done by spacy
    doc = nlp(pdf_text)
    # Try to replace `nlp` by other methods (ML, DL, rule-based, etc.)

    # one JSON output per PDF
    predictions = {
        "sentence": list(),
        "list": list(),
        "item": list(),
        "item1": list(),
        "item2": list(),
        "item3": list(),
        "item4": list(),
    }

    for sentence in doc.sents:

        # spacy conveniently provide start and end character index
        beg_i = sentence.start_char
        end_i = sentence.end_char

        # increment starting index to avoid space character
        while pdf_text[beg_i].isspace() and beg_i < end_i - 1:
            beg_i += 1

        # only sentence segmentation was done
        if beg_i < end_i:
            predictions["sentence"].append({"start": beg_i, "end": end_i})

        # missing start and item for all other labels !

    return predictions


def main(lang, split="train", dev_ratio=0.3):
    assert lang in ("en", "fr")
    print(f"\nPrediction (lang={lang})")

    # load data
    data_dir = Path(f"data/finsbd2_{split}/{lang}")
    list_ann = list(data_dir.glob("*.finsbd2.json"))
    total = len(list_ann)

    # only predict on dev (if you plan to train a model)
    nb_dev = int(round(dev_ratio * total, 0))
    train = list(range(0, total - nb_dev))
    dev = list(range(total - nb_dev, total))

    # load rule-based spacy model (fast) or train your own here :)
    nlp = spacy.load(lang, disable=["tagger", "parser", "ner"])
    sentencizer = nlp.create_pipe("sentencizer")
    nlp.add_pipe(sentencizer)

    # predict on each file in dev
    for i in dev:
        ann_path = list_ann[i]
        print(f"\n\t[{i+1}/{total}] processing '{ann_path.name}'")

        data = json.load(ann_path.open("r"))
        text = data["text"]

        # rule-based segmentation
        boundaries = predict_boundaries(text, nlp)

        # dump file into a JSON
        pred_path = ann_path.with_suffix(".pred.json")
        json.dump(boundaries, pred_path.open("w"), indent=2)
        print(f"\t\t- predictions saved in '{pred_path.name}'")


if __name__ == "__main__":
    main("en")
    main("fr")
