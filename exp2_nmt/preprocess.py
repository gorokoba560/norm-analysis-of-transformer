from mosestokenizer import *
import sentencepiece as spm


Languages = ["de", "en"]
Datas = ["train", "valid", "test"]
progress_dir_path = './work/data_in_progress/'
processed_dir_path = './work/processed_data/'
bpe_dir_path = './work/bpe_model_and_vocab/'


# tokenize train data by moses tokenizer
print("tokenizing by moses tokenizer...")
for ln in Languages:
    with open('./data/europarl-v7.de-en.' + ln, "r", encoding="utf-8") as fin:
        with open(progress_dir_path + 'tokenized_train_valid.' + ln, "w", encoding="utf-8") as fout:
            with MosesTokenizer(ln) as tokenize:
                for sent in fin:
                    sent = sent.strip()
                    tokenized_sent = ' '.join(tokenize(sent)) + "\n"
                    fout.write(tokenized_sent)


# remove sentences appeared in test data from train data & lowercase train data
print("removing sentences appeared in test data from train data & lowercasing...")
test_sentences = {}
removed_count = 0
not_removed_count = 0

for ln in Languages:
    with open(progress_dir_path + 'test.uc.' + ln, encoding="utf-8") as f:
        test_sentences[ln] = {l for l in f}

with open(progress_dir_path + 'tokenized_train_valid.en', encoding="utf-8") as en, open(progress_dir_path + 'tokenized_train_valid.de', encoding="utf-8") as de:
    with open(processed_dir_path + 'train_valid.en', "w", encoding="utf-8") as fo_en, open(processed_dir_path + 'train_valid.de', "w", encoding="utf-8") as fo_de:
        for e, d in zip(en, de):
            if e in test_sentences["en"] and d in test_sentences["de"]:
                # print(e + "|||" + d)
                removed_count += 1
            else:
                fo_en.write(e.lower())
                fo_de.write(d.lower())
                not_removed_count += 1
print("removed", removed_count, "sentences from train data")


# lowercase test data
for ln in Languages:
    with open("./work/data_in_progress/test.uc." + ln, encoding="utf-8") as fi, open("./work/processed_data/test." + ln, "w", encoding="utf-8") as fo:
        for line in fi:
            fo.write(line.lower())


# split train_valid data into train and valid data (final 1000 sentences are used as valid data)
for ln in Languages:
    with open(processed_dir_path + 'train_valid.' + ln, encoding="utf-8") as f:
        with open(processed_dir_path + 'train.' + ln, "w", encoding="utf-8") as ft, open(processed_dir_path + 'valid.' + ln, "w", encoding="utf-8") as fv:
            for i, line in enumerate(f):
                if i < not_removed_count - 1000:
                    ft.write(line)
                else:
                    fv.write(line)


# train BPE models
print('training BPE models...')
for lang in Languages:
    spm.SentencePieceTrainer.train("--input_sentence_size=100000000 --model_prefix=" + bpe_dir_path + lang + " --model_type=bpe --num_threads=4 --split_by_unicode_script=1 --split_by_whitespace=1  --remove_extra_whitespaces=1 --add_dummy_prefix=0 --normalization_rule_name=identity --vocab_size=10000 --character_coverage=1.0 --input=" + processed_dir_path + 'train.' + lang)


# tokenize each data by trained BPE models
print('tokenizing by trained BPE models...')
for lang in Languages:
    sp = spm.SentencePieceProcessor()
    sp.load(bpe_dir_path + lang + '.model')
    
    for phase in Datas:
        with open(processed_dir_path + phase + '.bpe.' + lang, 'w', encoding="utf-8") as fo:
            with open(processed_dir_path + phase + '.' + lang, encoding="utf-8") as fi:
                for line in fi:
                    fo.write(" ".join(sp.EncodeAsPieces(line.strip())) + "\n")