# FinSBD-2

Starter kit for shared task 
[FinSBD-2](https://sites.google.com/nlg.csie.ntu.edu.tw/finnlp2020/shared-task-finsbd-2).

Get training data by registering with 
[this form here](https://forms.gle/NixDGuVjrdFMjYhR9). 
After we review your registration, we will provide you with the training data.
There are 29 annotated PDFs (6 English + 23 French) in the training data. 
All annotations provided in the context of the shared task are licensed under 
the creative common license 
[CC BY-NC-ND](https://creativecommons.org/licenses/by-nc-nd/2.0/).

The test data will be released on the 1st of May 2020 and is composed of 
2 English and 10 French PDFs.


### Installation

The starter kit requires python 3.6. In addition, [spaCy](https://spacy.io/usage) 
is required to launch the segmentation script example:
- `pip install -U ispacy`
- `python -m spacy download en`
- `python -m spacy download fr`

However, in practice, you do need spaCy to create a segmentation model, feel 
free to be creative and come up with your own solution with different tools!

Clone starter kit:
- `git clone https://github.com/finsbd/finsbd2.git`
- `cd finsbd2`


### Train data

After downloading and unzipping training data in the folder data `data/`, 
you will end up with both english and french original PDFs and JSON annotations:
- English: 
    - `data/finsbd2_train/en/*.pdf`
    - `data/finsbd2_train/en/*.finsbd2.json`
- French:
    - `data/finsbd2_train/fr/*.pdf`
    - `data/finsbd2_train/fr/*.finsbd2.json`
    
Each FinSBD2 json possesses the boundaries of each type of segment, which is 
a pair of indexes in the text, `start` and `end`. The segment text 
is simply obtainable by doing `text[start: end]`. Starting character is 
obtainable by doing `text[start]` and ending character by doing `text[end-1]`.

BONUS: we also included coordinates of each boundaries so that you can visualize 
the data if needed. But the task is still prediction of boundaries indexes.
    
Example of training data:
```
{
    "text": "Ce document fournit des informations essentielles aux investisseurs ...",
    
    "sentence": [ {"start": 17, "end": 53, "coordinates": ... }, ... ],
    
    "list": [ {"start": 1080, "end": 1267, "coordinates": ... }, ... ],
    
    "item": [ ... ],
    
    "item1": [ ... ],
    
    "item2": [ ... ],
    
    "item3": [ ... ],
    
    "item4": [ ... ]
}
```


### Evaluation script

You can launch starter prediction script and evaluation script: 
- `python predict.py`   (poor baseline)
- `python evaluate.py`

Evaluation script expects predictions to be written in a JSON ending with 
suffix `.finsbd2.pred.json`:
- English: `data/finsbd2_train/en/*.finsbd2.pred.json`
- French: `data/finsbd2_train/fr/*.finsbd2.pred.json`


### Evaluation metric

The evaluation metric is a standard mean average of all f1-scores computed on 
each PDF. For each class, a prediction consist of a pair of integers corresponding 
respectively to the starting and ending indexes of the segment in the text.

Example of one prediction:
- `{"start": <start index: int>, "end": <end index: int>}`

A segment can be of the following 7 classes:
- sentence
- list
- item
- item1
- item2
- item3
- item4

Note that boundaries of item overlaps with those of item1, item2, item3 and item4:
- item1 = item1 U item2 U item3 U item4


Example of predictions saved in a JSON:
```
{
    "sentence": [ {"start": 17, "end": 53}, {"start": 56, "end": 107}, ... ],

    "list": [ {"start": 1080, "end": 1267}, ... ],

    "item": [ ... ],

    "item1": [ ... ],

    "item2": [ ... ],

    "item3": [ ... ],

    "item4": [ ... ]
}
```

A prediction is considered right if both starting and ending character indexes are right. 

In subtask 1, the f1-score of each PDF is the mean of f1-scores of classes 
sentence, list and item.

In subtask 2, the f1-score of each PDF is the mean of f1-scores of classes 
item1, item2, item3 and item4.
