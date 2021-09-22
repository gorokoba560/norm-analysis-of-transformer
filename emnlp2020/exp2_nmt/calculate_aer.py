import torch
from fairseq.models.transformer import TransformerModel
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import itertools
from collections import Counter
from scipy.stats import spearmanr, pearsonr
import warnings


device = "cuda" if torch.cuda.is_available() else "cpu"

test_de_bpe = "./work/processed_data/test.bpe.de"
test_de_word = "./work/processed_data/test.de"
test_en_bpe = "./work/processed_data/test.bpe.en"
test_en_word = "./work/processed_data/test.en"
gold_alignment = "./work/gold_alignment/alignment.talp"
seeds = [2253, 5498, 9819, 9240, 2453]

warnings.simplefilter('ignore')


def load_model(seed):
    """
    Given a seed (as a integer), load the corresponding model.
    """
    model = TransformerModel.from_pretrained(
      'work/checkpoints_seed' + str(seed),
      checkpoint_file='checkpoint_best.pt',
      data_name_or_path='work/processed_data/fairseq_preprocessed_data',
    ).to(device)
    model.eval()
    return model


def convert_bpe_word(bpe_sent, word_sent):
    """
    Given a sentence made of BPE subwords and words (as a string), 
    returns a list of lists where each sub-list contains BPE subwordss
    for the correponding word.
    """

    splited_bpe_sent = bpe_sent.split()
    word_sent = word_sent.split()
    word_to_bpe = [[] for _ in range(len(word_sent))]
    
    word_i = 0
    for bpe_i, token in enumerate(splited_bpe_sent):
        if token.startswith("▁"):
            word_i += 1
        word_to_bpe[word_i].append(bpe_i)

    for word in word_to_bpe:
        assert len(word) != 0
    
    word_to_bpe.append([len(splited_bpe_sent)])
    return word_to_bpe


def get_word_word_attention(token_token_attention, src_word_to_bpe, trg_word_to_bpe, remove_EOS=True):
    """
    Given a token-to-token matrix, convert it into word-to-word attention.
    """

    word_word_attention = np.array(token_token_attention)
    not_word_starts = []
    for word in src_word_to_bpe:
        not_word_starts += word[1:]

    # sum up the attention weights or vector norms for all tokens in a source word that has been split
    for word in src_word_to_bpe:
        word_word_attention[:, word[0]] = word_word_attention[:, word].sum(axis=-1)
    word_word_attention = np.delete(word_word_attention, not_word_starts, -1)

    not_word_starts = []
    for word in trg_word_to_bpe:
        not_word_starts += word[1:]

    # mean the attention weights or vector norms for all tokoens in a target word that has been split
    for word in trg_word_to_bpe:
        word_word_attention[word[0]] = np.mean(word_word_attention[word], axis=0)

    word_word_attention = np.delete(word_word_attention, not_word_starts, 0)
    if remove_EOS:
        word_word_attention = np.delete(word_word_attention, -1, 0)

    return word_word_attention


def parse_single_alignment(string, reverse=False, one_add=False, one_indexed=False):
    """
    Given an alignment (as a string such as "3-2" or "5p4"), return the index pair.
    """
    assert '-' in string or 'p' in string

    a, b = string.replace('p', '-').split('-')
    a, b = int(a), int(b)

    if one_indexed:
        a = a - 1
        b = b - 1
    
    if one_add:
        a = a + 1
        b = b + 1

    if reverse:
        a, b = b, a

    return a, b


def extract_matrix(seed):
    """
    Run a model to the test set and extract attention weights and vector norms.
    """
    with open(test_de_bpe, encoding="utf-8") as fbpe:
        src_bpe_sents = fbpe.readlines()

    with open(test_en_bpe, encoding="utf-8") as fbpe:
        tgt_bpe_sents = fbpe.readlines()

    model = load_model(seed)
    
    extract_matrix = []
    eos_id = model.tgt_dict.index("</s>")

    for i in range(len(src_bpe_sents)):
        de = src_bpe_sents[i].strip()
        en = tgt_bpe_sents[i].strip()
        de_idx = model.encode(de).view(1,-1).to(device)
        en_idx = torch.tensor([eos_id] + [model.tgt_dict.index(t) for t in en.split()]).view(1,-1).to(device)

        outputs = model.models[0](de_idx, de_idx.size()[-1], en_idx, output_all_attentions=True, output_all_norms=True)
        a = outputs[1]['attn']
        afx = outputs[1]['norms'][1]
        summed_afx = outputs[1]['norms'][2]

        extract_matrix.append({"attn":a, "afx":afx, "summed_afx":summed_afx})

    with open('./work/results_seed' + str(seed) + '/alignments/extracted_matrix.pkl', 'wb') as f:
        pickle.dump(extract_matrix, f)

    return None


def alignment_extract(seed, setting = "AWO"):
    """
    Construct alignments from the extracted matrix (attention weights or vector norms).
    """
    with open('./work/results_seed' + str(seed) + '/alignments/extracted_matrix.pkl', 'rb') as f:
        extract_matrix = pickle.load(f)

    with open(test_de_bpe, encoding="utf-8") as fbpe:
        src_bpe_sents = fbpe.readlines()
    with open(test_de_word, encoding="utf-8") as fword:
        src_word_sents = fword.readlines()

    with open(test_en_bpe, encoding="utf-8") as fbpe:
        tgt_bpe_sents = fbpe.readlines()
    with open(test_en_word, encoding="utf-8") as fword:
        tgt_word_sents = fword.readlines()

    for mode in ["attn", "summed_afx"]:
        for l in range(6):
            with open("./work/results_seed" + str(seed) + "/alignments/hypothesis-{}-{}-{}".format(mode,l,setting), "w") as f:
                for i in range(len(src_bpe_sents)):
                    src_bpe_sent = src_bpe_sents[i]
                    src_word_sent = src_word_sents[i]
                    src_word_to_bpe = convert_bpe_word(src_bpe_sent, src_word_sent)
                    src_len = len(src_word_sent.split())

                    tgt_bpe_sent = tgt_bpe_sents[i]
                    tgt_word_sent = tgt_word_sents[i]
                    tgt_word_to_bpe = convert_bpe_word(tgt_bpe_sent, tgt_word_sent)
                    
                    if mode == "summed_afx":
                        attention_matrix = torch.squeeze(extract_matrix[i][mode][l]).detach().numpy()
                    else:
                        attention_matrix = torch.squeeze(extract_matrix[i][mode][l]).mean(dim=0).detach().numpy()

                    if setting == "AWI":
                        attention_matrix = attention_matrix[list(range(1,len(attention_matrix)))+[0]]
                    attention_matrix = get_word_word_attention(attention_matrix, src_word_to_bpe, tgt_word_to_bpe)
                    attention_matrix = np.argmax(attention_matrix, -1)

                    for t, s_a in enumerate(attention_matrix):
                        if s_a != src_len:
                            f.write("{}-{} ".format(t+1, s_a+1))
                    f.write("\n")    

    for mode in ["attn", "afx"]:
        for l in range(6):
            for h in range(4):
                with open("./work/results_seed" + str(seed) + "/alignments/hypothesis-{}-{}-{}-{}".format(mode,l,h,setting), "w") as f:
                    for i in range(len(src_bpe_sents)):
                        src_bpe_sent = src_bpe_sents[i]
                        src_word_sent = src_word_sents[i]
                        src_word_to_bpe = convert_bpe_word(src_bpe_sent, src_word_sent)
                        src_len = len(src_word_sent.split())

                        tgt_bpe_sent = tgt_bpe_sents[i]
                        tgt_word_sent = tgt_word_sents[i]
                        tgt_word_to_bpe = convert_bpe_word(tgt_bpe_sent, tgt_word_sent)

                        attention_matrix = torch.squeeze(extract_matrix[i][mode][l])[h].detach().numpy()
                        if setting == "AWI":
                            attention_matrix = attention_matrix[list(range(1,len(attention_matrix)))+[0]]
                        attention_matrix = get_word_word_attention(attention_matrix, src_word_to_bpe, tgt_word_to_bpe)
                        attention_matrix = np.argmax(attention_matrix, -1)

                        for t, s_a in enumerate(attention_matrix):
                            if s_a != src_len:
                                f.write("{}-{} ".format(t+1, s_a+1))
                        f.write("\n")
    return None


def layer_aer_calculate(seed, setting="AWO"):
    """
    Calculate AER (alignment error rate) for constructed alignments in the layer-level.
    """
    ids = {"attn":0, "summed_afx":1}
    results = np.zeros((2,6))

    sure, possible = [], []

    with open(gold_alignment, 'r') as f:
        for line in f:
            sure.append(set())
            possible.append(set())

            for alignment_string in line.split():

                sure_alignment = True if '-' in alignment_string else False
                alignment_tuple = parse_single_alignment(alignment_string, reverse=True)

                if sure_alignment:
                    sure[-1].add(alignment_tuple)
                possible[-1].add(alignment_tuple)

    target_sentences = []
    with open(test_en_word, encoding="utf-8") as fe:
        for en in fe:
            target_sentences.append(en.split())

    source_sentences = []
    with open(test_de_word,encoding="utf-8") as fd:
        for de in fd:
            source_sentences.append(de.split())

    assert len(sure) == len(possible)
    assert len(target_sentences) == len(source_sentences)
    assert len(sure) == len(source_sentences)

    for mode in ["attn", "summed_afx"]:
        for l in range(6):
            hypothesis = []

            with open("./work/results_seed" + str(seed) + "/alignments/hypothesis-{}-{}-{}".format(mode,l,setting)) as f:
                for line in f:
                    hypothesis.append(set())

                    for alignment_string in line.split():
                        alignment_tuple = parse_single_alignment(alignment_string)
                        hypothesis[-1].add(alignment_tuple)

            sum_a_intersect_p, sum_a_intersect_s, sum_s, sum_a = 4 * [0.0]

            for S, P, A in itertools.zip_longest(sure, possible, hypothesis):
                sum_a += len(A)
                sum_s += len(S)
                sum_a_intersect_p += len(A.intersection(P))
                sum_a_intersect_s += len(A.intersection(S))

            precision = sum_a_intersect_p / sum_a
            recall = sum_a_intersect_s / sum_s
            aer = 1.0 - ((sum_a_intersect_p + sum_a_intersect_s) / (sum_a + sum_s))
            results[ids[mode]][l] = aer
    return results


def head_aer_calculate(seed, setting="AWO"):
    """
    Calculate AER (alignment error rate) for constructed alignments in the heaed-level.
    """
    ids = {"attn":0, "afx":1}
    results = np.zeros((2,6,4))

    sure, possible = [], []

    with open(gold_alignment, 'r') as f:
        for line in f:
            sure.append(set())
            possible.append(set())

            for alignment_string in line.split():

                sure_alignment = True if '-' in alignment_string else False
                alignment_tuple = parse_single_alignment(alignment_string, reverse=True)

                if sure_alignment:
                    sure[-1].add(alignment_tuple)
                possible[-1].add(alignment_tuple)


    target_sentences = []
    with open(test_en_word,encoding="utf-8") as fe:
        for en in fe:
            target_sentences.append(en.split())

    source_sentences = []
    with open(test_de_word,encoding="utf-8") as fd:
        for de in fd:
            source_sentences.append(de.split())

    assert len(sure) == len(possible)
    assert len(target_sentences) == len(source_sentences)
    assert len(sure) == len(source_sentences)


    for mode in ["attn", "afx"]:
        for l in range(6):
            for h in range(4):
                hypothesis = []

                with open("./work/results_seed" + str(seed) + "/alignments/hypothesis-{}-{}-{}-{}".format(mode,l,h,setting)) as f:
                    for line in f:
                        hypothesis.append(set())

                        for alignment_string in line.split():
                            alignment_tuple = parse_single_alignment(alignment_string)
                            hypothesis[-1].add(alignment_tuple)

                sum_a_intersect_p, sum_a_intersect_s, sum_s, sum_a = 4 * [0.0]

                for S, P, A in itertools.zip_longest(sure, possible, hypothesis):
                    sum_a += len(A)
                    sum_s += len(S)
                    sum_a_intersect_p += len(A.intersection(P))
                    sum_a_intersect_s += len(A.intersection(S))

                precision = sum_a_intersect_p / sum_a
                recall = sum_a_intersect_s / sum_s
                aer = 1.0 - ((sum_a_intersect_p + sum_a_intersect_s) / (sum_a + sum_s))
                results[ids[mode]][l][h] = aer
    return results


def layer_AER_visualization(seed, layer_results, setting):
    """
    Visualize the layer-level AER results in a heatmap.
    """
    plt.figure(figsize=(9,5), dpi=500)
    df = pd.DataFrame(layer_results, columns=list(range(1,7)),index=["Attention weights α ", "Vector-norms \n(ours)"])
    sns.set(font_scale=1.2)
    sns.heatmap(df,cmap="Reds",cbar=False,square=True, annot=True, fmt='.1f',annot_kws={"size": 14})
    plt.gcf().subplots_adjust(bottom=0.2)
    plt.gcf().subplots_adjust(left=0.35)
    plt.xlabel("layer", fontsize=15)
    plt.savefig("results/layer_aer_seed{}_{}.png".format(seed, setting))
    plt.close()


def head_AER_visualization(seed, head_results, setting):
    """
    Visualize the head-level AER results in a heatmap.
    """
    plt.figure(figsize=(4,6), dpi=100)
    df = pd.DataFrame(head_results[1], columns=list(range(1,5)),index=list(range(1,7)))
    sns.set(font_scale=1.2)
    sns.heatmap(df,cmap="Purples",cbar=False,square=True, annot=True, fmt='.1f',annot_kws={"size": 14})
    plt.gcf().subplots_adjust(left=0.2)
    plt.xlabel("head", fontsize=15)
    plt.ylabel("layer", fontsize=15)
    plt.savefig("results/norm_head_aer_seed{}_{}.png".format(seed, setting))
    plt.close()

    
def correlation_with_averaged_norm(seed, head_results, setting):
    """
    Visualize the averaged norm (head-level) in a heatmap and calculates the correlation between AER and averaged norm.
    """
    with open('./work/results_seed' + str(seed) + '/alignments/extracted_matrix.pkl', 'rb') as f:
        extract_matrix = pickle.load(f)

    averaged_norm = np.zeros((6,4))

    mode_id = {"attn":0, "afx":1}

    mode = "afx"
    for l in range(6):
        for h in range(4):
            total = 0
            count = 0
            for idx in range(len(extract_matrix)):
                afx = torch.squeeze(extract_matrix[idx][mode][l])[h].detach().numpy()
                total += afx.sum()
                x, y = afx.shape
                count += x*y
            averaged_norm[l][h] = total/count

    plt.figure(figsize=(4,6), dpi=100)
    df = pd.DataFrame(averaged_norm, columns=list(range(1,5)),index=list(range(1,7)))
    sns.set(font_scale=1.2)
    sns.heatmap(df,cmap="Greens",cbar=False,square=True, annot=True, fmt='.2f',annot_kws={"size": 14})
    plt.gcf().subplots_adjust(left=0.2)
    plt.xlabel("head", fontsize=15)
    plt.ylabel("layer", fontsize=15)
    plt.savefig("results/averaged_norm_seed{}_{}.png".format(seed, setting))
    plt.close()
    
    aer = head_results[1].reshape(24)
    norm = averaged_norm.reshape(24)

    spearman, pvalue = spearmanr(aer, norm)
    pearson, pvalue = pearsonr(aer, norm)
    
    highest_norm_layer = np.argmax(np.sum(averaged_norm, 1))
    highest_norm_head = (np.argmax(averaged_norm.reshape(24))//4, np.argmax(averaged_norm.reshape(24))%4)
    
    return spearman, pearson, highest_norm_layer, highest_norm_head


def list_mean(lis):
    """
    Given a list of numerical values, return the mean value.
    """
    return sum(lis)/len(lis)


if __name__ == '__main__':
    
    seed_average_layer_mean_weight = {"AWO":[], "AWI":[]}
    seed_average_layer_mean_norm = {"AWO":[], "AWI":[]}
    seed_average_best_layer_weight = {"AWO":[], "AWI":[]}
    seed_average_best_layer_norm = {"AWO":[], "AWI":[]}
    seed_average_norm_highest_layer = {"AWO":[], "AWI":[]}
    seed_average_norm_highest_head = {"AWO":[], "AWI":[]}
    seed_average_spearman = {"AWO":[], "AWI":[]}
    seed_average_pearson = {"AWO":[], "AWI":[]}
    
    for seed in seeds:
        print("seed: ", seed)
        extract_matrix(seed)
        
        for setting in ["AWO", "AWI"]:
            if setting == "AWO":
                print("--------------- Alignmen with output (AWO) setting ---------------")
            else:
                print("--------------- Alignmen with input (AWI) setting ---------------")
            
            alignment_extract(seed, setting=setting)
            layer_results = layer_aer_calculate(seed, setting) * 100
            head_results = head_aer_calculate(seed, setting) * 100
            
            if seed == 2253:
                seed_average_layer_results = layer_results.copy()
            else:
                seed_average_layer_results += layer_results.copy()
            
            layer_mean = np.mean(layer_results, 1)
            print("layer mean:")
            print("\tAttention weight:\t\t\t", layer_mean[0])
            print("\tOur norm:\t\t\t\t", layer_mean[1])
            seed_average_layer_mean_weight[setting].append(layer_mean[0])
            seed_average_layer_mean_norm[setting].append(layer_mean[1])
            
            print("best layer:")
            print("\tAttention weight:\t", "layer", np.argmin(layer_results[0])+1, "\t", np.min(layer_results[0]))
            print("\tOur norm:\t\t", "layer", np.argmin(layer_results[1])+1, "\t", np.min(layer_results[1]))
            seed_average_best_layer_weight[setting].append(np.min(layer_results[0]))
            seed_average_best_layer_norm[setting].append(np.min(layer_results[1]))
            
            layer_AER_visualization(seed, layer_results, setting)
            head_AER_visualization(seed, head_results, setting)
            spearman, pearson, highest_norm_layer, highest_norm_head = correlation_with_averaged_norm(seed, head_results, setting)
            
            print("layer with the highest average norm:")
            print("\tOur norm:\t", "\tlayer", highest_norm_layer+1, "\t", layer_results[1][highest_norm_layer])
            seed_average_norm_highest_layer[setting].append(layer_results[1][highest_norm_layer])
            
            print("head with the highest average norm:")
            print("\tOur norm:\t", "layer", highest_norm_head[0]+1, "head", highest_norm_head[1]+1, "\t", head_results[1][highest_norm_head[0]][highest_norm_head[1]])
            seed_average_norm_highest_head[setting].append(head_results[1][highest_norm_head[0]][highest_norm_head[1]])
            
            print("correlation with averaged norm:")
            print("\tSpearman's ρ:\t\t\t\t", spearman)
            print("\tPearson's r:\t\t\t\t", pearson)
            seed_average_spearman[setting].append(spearman)
            seed_average_pearson[setting].append(pearson)
        print("\n\n")
        
    print("5 seeds average")
    for setting in ["AWO", "AWI"]:
        if setting == "AWO":
            print("---------- Alignmen with output (AWO) setting ----------")
        else:
            print("---------- Alignmen with input (AWI) setting ----------")
        
        print("layer mean:")
        print("\tAttention weight:\t\t\t", list_mean(seed_average_layer_mean_weight[setting]))
        print("\tOur norm:\t\t\t\t", list_mean(seed_average_layer_mean_norm[setting]))
        
        print("best layer:")
        print("\tAttention weight:\t\t\t", list_mean(seed_average_best_layer_weight[setting]))
        print("\tOur norm:\t\t\t\t", list_mean(seed_average_best_layer_norm[setting]))
        
        print("layer with the highest average norm:")
        print("\tOur norm:\t\t\t\t", list_mean(seed_average_norm_highest_layer[setting]))
        
        print("head with the highest average norm:")
        print("\tOur norm:\t\t\t\t", list_mean(seed_average_norm_highest_head[setting]))
        
        print("correlation with averaged norm:")
        print("\tSpearman's ρ:\t\t\t\t", list_mean(seed_average_spearman[setting]))
        print("\tPearson's r:\t\t\t\t", list_mean(seed_average_pearson[setting]))
        
        layer_AER_visualization("average", seed_average_layer_results/5, setting)