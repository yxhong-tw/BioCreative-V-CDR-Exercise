# BioCreative-V-CDR-Exercise
This is the National Cheng Kung University (NCKU) Intelligent Knowledge Management (IKM) Lab's exercise for the new incoming students in the Fall 2023 semester.

## Before Start
### Requirements
You can install the requirements by running the following command:
```
pip install -r requirements.txt
```

### Usage
You can edit `configs` in `main.py`, then run this program with the following command:
```
python main.py
```

## Model
### Arch. 1: Only LM
I choose `bert-base-uncased` as the pretrained model to do this task.  

### ~~Arch. 2: DeBERTa + BiGRU + CRF~~
I have no time to finish, so sad.

#### ~~Potential Problems~~
~~1. Tokenizer ValueError~~
```
ValueError: Couldn't instantiate the backend tokenizer from one of: 
(1) a `tokenizers` library serialization file, 
(2) a slow tokenizer instance to convert or 
(3) an equivalent slow tokenizer class to instantiate and convert. 
You need to have sentencepiece installed to convert a slow tokenizer to a fast one.
```

~~Solution: Install sentencepiece~~
```
pip install sentencepiece
```

> ~~Reference: [PegasusTokenizer requires the SentencePiece library but it was not found in your environment · Issue #8963 · huggingface/transformers](https://github.com/huggingface/transformers/issues/8963)~~

~~2. Tokenizer ImportError~~
```
DebertaV2Converter requires the protobuf library but it was not found in your environment. Checkout the instructions on the
installation page of its repo: https://github.com/protocolbuffers/protobuf/tree/master/python#installation and follow the ones
that match your environment. Please note that you may need to restart your runtime after installation.
```

~~Solution: Install protobuf~~
```
pip install protobuf
```

> ~~Reference: [Protobuf · Issue #10020 · huggingface/transformers](https://github.com/huggingface/transformers/issues/10020)~~

## Evaluation
I calculate the `Micro` and `Macro` `Precision`, `Recall`, `F1-score` through the `seqeval` module to evalute the model performance.

## Experiment
### Environment
```
Python Modules Info:
numpy                == 1.26.0
seqeval              == 1.2.2
tabulate             == 0.9.0
torch                == 2.0.1
transformers         == 4.33.3

CUDA, GPU Driver and OS Info:
- CUDA Version       == 12.2
- GPU Driver Version == 535.104.12
- Ubuntu             == 22.04.3 (LTS)

Hardware Info:
- GPU                == NVIDIA GeForce RTX 3090
```

### Results
![Precision](https://drive.google.com/uc?export=view&id=1reDf1SnaX1Fa1CQWr9ejOkoNMD_2H-Yi)
![Recall](https://drive.google.com/uc?export=view&id=1rjEKtdgdUkLeekfgp3IdVzFhA1Yg_1GD)
![F1-Score](https://drive.google.com/uc?export=view&id=1rh_yax4cVvpArf11c3VoB90XtmifbTcY)

If you want to see more results, you can find them in `logs` folder.
