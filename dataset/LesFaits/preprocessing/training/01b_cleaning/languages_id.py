import pandas as pd


langs_id = [

    {
        "lang": "English",
        "dataset_id": "en",
        "stopwords_id": "en",
        "flagged_words_id": "en",
        "fasttext_id": "en",
        "sentencepiece_id": "en",
        "kenlm_id": "en",
    },
    {
        "lang": "French",
        "dataset_id": "fr",
        "stopwords_id": "fr",
        "flagged_words_id": "fr",
        "fasttext_id": "fr",
        "sentencepiece_id": "fr",
        "kenlm_id": "fr",
    }
   
]
langs_id = pd.DataFrame(langs_id)
