import csv
import statistics
from collections import defaultdict

def clean_model_name(name):
    # Existing name mapping...
    name_mapping = {
        "bm25s": "BM25",
        "Salesforce_SFR-Embedding-2_R": "SFR-Embedding-2-R",
        "unicamp-dl_mt5-base-mmarco-v2": "mT5-base-mmarco",
        "facebook_mcontriever-msmarco": "mContriever-msmarco",
        "castorini_mdpr-tied-pft-msmarco-ft-all": "mDPR-tied-PFT",
        "Alibaba-NLP_gte-multilingual-base": "GTE-base",
        "jhu-clsp_FollowIR-7B": "FollowIR-7B",
        "jhu-clsp_mFollowIR-7B-all-2ep": "mFollowIR-7B-all-2ep",
        "jhu-clsp_mFollowIR-7B-fas-2ep": "mFollowIR-7B-fas-2ep",
        "jhu-clsp_mFollowIR-7B-zho-2ep": "mFollowIR-7B-zho-2ep",
        "jhu-clsp_mFollowIR-7B-rus-2ep": "mFollowIR-7B-rus-2ep",
        "jhu-clsp_mFollowIR-7B-all": "mFollowIR-7B-all",
        "jhu-clsp_mFollowIR-7B-fas": "mFollowIR-7B-fas",
        "jhu-clsp_mFollowIR-7B-zho": "mFollowIR-7B-zho",
        "jhu-clsp_mFollowIR-7B-rus": "mFollowIR-7B-rus",
        "BAAI_bge-reranker-v2-m3": "BGE-reranker-v2-m3",
        "Alibaba-NLP_gte-Qwen2-7B-instruct": "GTE-Qwen2-7B",
        "GritLM_GritLM-7B": "GritLM-7B",
        "intfloat_e5-mistral-7b-instruct": "E5-Mistral-7B",
        "jinaai_jina-reranker-v2-base-multilingual": "Jina-reranker-v2-base",
        "nomic-ai_nomic-embed-text-v1.5": "Nomic-embed-text",
        "samaya-ai_promptriever-llama2-7b-v1": "Promptriever-LLaMA2-7B",
        "samaya-ai_promptriever-llama3.1-8b-v1": "Promptriever-LLaMA3.1-8B",
        "samaya-ai_promptriever-mistral-v0.1-7b-v1": "Promptriever-Mistral-7B",
        "samaya-ai_RepLLaMA-reproduced": "RepLLaMA",
        "intfloat_multilingual-e5-base": "mE5-base",
        "intfloat_multilingual-e5-large": "mE5-large",
        "intfloat_multilingual-e5-small": "mE5-small",
        "mistralai_Mistral-7B-Instruct-v0.2": "Mistral-7B-Instruct"
    }
    return name_mapping.get(name, name)

def is_cross_encoder(model_name):
    # Existing cross encoder list...
    cross_encoders = [
        "unicamp-dl_mt5-base-mmarco-v2",
        "unicamp-dl_mt5-13b-mmarco-100k",
        "jhu-clsp_mFollowIR-7B-all",
        "jhu-clsp_mFollowIR-7B-fas",
        "jhu-clsp_mFollowIR-7B-zho",
        "jhu-clsp_mFollowIR-7B-rus",
        "mFollowIR-7B-all",
        "mFollowIR-7B-fas",
        "mFollowIR-7B-zho",
        "mFollowIR-7B-rus",
        "mFollowIR-7B-all-2ep",
        "mFollowIR-7B-fas-2ep",
        "mFollowIR-7B-zho-2ep",
        "mFollowIR-7B-rus-2ep",
        "jhu-clsp_mFollowIR-7B-*",
        "mistralai_Mistral-7B-Instruct-v0.2",
        "jhu-clsp_FollowIR-7B",
        "BAAI_bge-reranker-v2-m3",
        "jinaai_jina-reranker-v2-base-multilingual"
    ]
    return any(ce in model_name for ce in cross_encoders)

def format_number(num, is_pmrr=False):
    return f"{float(num * 100):.1f}" if is_pmrr else f"{float(num):.3f}"

def bold_if_best(value, best_value, is_pmrr=False):
    if value == best_value:
        return r"\textbf{" + format_number(value, is_pmrr) + "}"
    return format_number(value, is_pmrr)

def generate_latex_tables(csv_file):
    models = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    followir_models = defaultdict(lambda: defaultdict(dict))
    
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            model = row['Model']
            type_ = row['Type']
            if type_ == 'normal':
                lang = row['Language Pair'][:3]
            else:  # cross_lingual
                lang = row['Language Pair'].split('-')[2][:3]
            
            encoder_type = "cross_encoder" if is_cross_encoder(model) else "bi_encoder"
            
            if "mFollowIR" in model or "FollowIR" in model:
                followir_models[clean_model_name(model)][lang] = {
                    'ndcg@20': float(row['ndcg_at_20']),
                    'p-MRR': float(row['p-MRR'])
                }
            if "mFollowIR" not in model:
                models[type_][encoder_type][clean_model_name(model)][lang] = {
                    'ndcg@20': float(row['ndcg_at_20']),
                    'p-MRR': float(row['p-MRR'])
                }

    latex_tables = {}
    for type_ in ['normal', 'cross_lingual']:
        latex = r"""\begin{table*}[t]
\centering
\caption{Results for mFollowIR dataset """ + (f"(Cross Lingual Retrieval)" if type_ == "cross_lingual" else "") + r""" across three language subsets (Persian, Chinese, Russian). nDCG@20 and p-MRR scores are reported. Best value in each column is bolded per encoder type. Models are sorted by average p-MRR, with the highest at the bottom.}
\label{tab:mfollowir_""" + type_ + r"""}
\vspace{0.5em}
\resizebox{\textwidth}{!}{%
\begin{tabular}{ll|cc@{\hspace{1em}}c|cc@{\hspace{1em}}c|cc@{\hspace{1em}}c|cc}
\toprule
& & \multicolumn{3}{c|}{Persian} & \multicolumn{3}{c|}{Chinese} & \multicolumn{3}{c|}{Russian} & \multicolumn{2}{c}{Average} \\
\cmidrule(l){3-5} \cmidrule(l){6-8} \cmidrule(l){9-11} \cmidrule(l){12-13}
& Model & nDCG@20 & & p-MRR & nDCG@20 & & p-MRR & nDCG@20 & & p-MRR & nDCG@20 & p-MRR \\
\midrule
"""

        for encoder_type in ['cross_encoder', 'bi_encoder']:
            # Calculate best scores for this encoder type
            best_scores = {lang: {'ndcg@20': 0, 'p-MRR': -float('inf')} for lang in ['fas', 'zho', 'rus', 'Avg']}
            for model_data in models[type_][encoder_type].values():
                avg_ndcg = statistics.mean(model_data[lang]['ndcg@20'] for lang in ['fas', 'zho', 'rus'])
                avg_pmrr = statistics.mean(model_data[lang]['p-MRR'] for lang in ['fas', 'zho', 'rus'])
                for lang in ['fas', 'zho', 'rus']:
                    if model_data[lang]['ndcg@20'] > best_scores[lang]['ndcg@20']:
                        best_scores[lang]['ndcg@20'] = model_data[lang]['ndcg@20']
                    if model_data[lang]['p-MRR'] > best_scores[lang]['p-MRR']:
                        best_scores[lang]['p-MRR'] = model_data[lang]['p-MRR']
                if avg_ndcg > best_scores['Avg']['ndcg@20']:
                    best_scores['Avg']['ndcg@20'] = avg_ndcg
                if avg_pmrr > best_scores['Avg']['p-MRR']:
                    best_scores['Avg']['p-MRR'] = avg_pmrr

            models_of_type = models[type_][encoder_type]
            sorted_models = sorted(models_of_type.items(), key=lambda x: statistics.mean(data['p-MRR'] for lang, data in x[1].items()))
            
            if sorted_models:
                latex += r"\multirow{" + str(len(sorted_models)) + r"}{*}{\rotatebox[origin=c]{90}{\parbox[c]{1.5cm}{\centering \scriptsize " + ("Cross-encoder" if encoder_type == "cross_encoder" else "Bi-encoder") + r"}}} "

                for model, data in sorted_models:
                    avg_ndcg = statistics.mean(data[lang]['ndcg@20'] for lang in ['fas', 'zho', 'rus'])
                    avg_pmrr = statistics.mean(data[lang]['p-MRR'] for lang in ['fas', 'zho', 'rus'])
                    
                    latex += f"& {model} & "
                    
                    for lang in ['fas', 'zho', 'rus']:
                        ndcg = data[lang]['ndcg@20']
                        pmrr = data[lang]['p-MRR']
                        latex += f"{bold_if_best(ndcg, best_scores[lang]['ndcg@20'])} & & "
                        latex += f"{bold_if_best(pmrr, best_scores[lang]['p-MRR'], is_pmrr=True)} & "
                    
                    latex += f"{bold_if_best(avg_ndcg, best_scores['Avg']['ndcg@20'])} & "
                    latex += f"{bold_if_best(avg_pmrr, best_scores['Avg']['p-MRR'], is_pmrr=True)}"
                    
                    latex += r" \\" + "\n"
                
                if encoder_type == 'cross_encoder':
                    latex += r"\midrule" + "\n"

        latex += r"""\bottomrule
\end{tabular}
}
\vspace{-0.5em}
\end{table*}"""

        latex_tables[type_] = latex

    # Generate the third table for mFollowIR and FollowIR models
    followir_latex = r"""\begin{table*}[t]
\centering
\caption{Results for mFollowIR and FollowIR models across three language subsets (Persian, Chinese, Russian). nDCG@20 and p-MRR scores are reported.}
\label{tab:followir_models}
\vspace{0.5em}
\resizebox{\textwidth}{!}{%
\begin{tabular}{l|cc@{\hspace{1em}}c|cc@{\hspace{1em}}c|cc@{\hspace{1em}}c|cc}
\toprule
& \multicolumn{3}{c|}{Persian} & \multicolumn{3}{c|}{Chinese} & \multicolumn{3}{c|}{Russian} & \multicolumn{2}{c}{Average} \\
\cmidrule(l){2-4} \cmidrule(l){5-7} \cmidrule(l){8-10} \cmidrule(l){11-12}
Model & nDCG@20 & & p-MRR & nDCG@20 & & p-MRR & nDCG@20 & & p-MRR & nDCG@20 & p-MRR \\
\midrule
"""

    best_scores = {lang: {'ndcg@20': 0, 'p-MRR': -float('inf')} for lang in ['fas', 'zho', 'rus', 'Avg']}
    for model, data in followir_models.items():
        avg_ndcg = statistics.mean(data[lang]['ndcg@20'] for lang in ['fas', 'zho', 'rus'])
        avg_pmrr = statistics.mean(data[lang]['p-MRR'] for lang in ['fas', 'zho', 'rus'])
        for lang in ['fas', 'zho', 'rus']:
            if data[lang]['ndcg@20'] > best_scores[lang]['ndcg@20']:
                best_scores[lang]['ndcg@20'] = data[lang]['ndcg@20']
            if data[lang]['p-MRR'] > best_scores[lang]['p-MRR']:
                best_scores[lang]['p-MRR'] = data[lang]['p-MRR']
        if avg_ndcg > best_scores['Avg']['ndcg@20']:
            best_scores['Avg']['ndcg@20'] = avg_ndcg
        if avg_pmrr > best_scores['Avg']['p-MRR']:
            best_scores['Avg']['p-MRR'] = avg_pmrr

    sorted_models = sorted(followir_models.items(), key=lambda x: statistics.mean(data['p-MRR'] for lang, data in x[1].items()), reverse=True)

    for model, data in sorted_models:
        avg_ndcg = statistics.mean(data[lang]['ndcg@20'] for lang in ['fas', 'zho', 'rus'])
        avg_pmrr = statistics.mean(data[lang]['p-MRR'] for lang in ['fas', 'zho', 'rus'])
        
        followir_latex += f"{model} & "
        
        for lang in ['fas', 'zho', 'rus']:
            ndcg = data[lang]['ndcg@20']
            pmrr = data[lang]['p-MRR']
            followir_latex += f"{bold_if_best(ndcg, best_scores[lang]['ndcg@20'])} & & "
            followir_latex += f"{bold_if_best(pmrr, best_scores[lang]['p-MRR'], is_pmrr=True)} & "
        
        followir_latex += f"{bold_if_best(avg_ndcg, best_scores['Avg']['ndcg@20'])} & "
        followir_latex += f"{bold_if_best(avg_pmrr, best_scores['Avg']['p-MRR'], is_pmrr=True)}"
        
        followir_latex += r" \\" + "\n"

    followir_latex += r"""\bottomrule
\end{tabular}
}
\vspace{-0.5em}
\end{table*}"""

    latex_tables['followir'] = followir_latex

    return latex_tables

# Usage
csv_file = 'results_summary.csv'
latex_tables = generate_latex_tables(csv_file)

print("Normal Table:")
print(latex_tables['normal'])
print("\nCross-lingual Table:")
print(latex_tables['cross_lingual'])
print("\nFollowIR and mFollowIR Table:")
print(latex_tables['followir'])