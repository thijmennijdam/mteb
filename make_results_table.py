import csv
import statistics

def format_model_name(name):
    # Remove common prefixes and suffixes
    name = name.split('__')[-1]  # Remove everything before the last double underscore
    name = name.replace('-msmarco', '')
    name = name.replace('-ft-all', '')
    
    # Capitalize words and handle special cases
    words = name.split('-')
    formatted_words = []
    for word in words:
        if word.lower() in ['e5', "sfr"]:
            formatted_words.append(word.upper())
        elif word.lower() == 'pft':
            formatted_words.append('PFT')
        elif word.lower() == "mdpr":
            formatted_words.append('mDPR')
        elif word.lower() == "mcontriever":
            formatted_words.append('mContriever')
        elif word.lower() == "2_r":
            formatted_words.append('2-R')
        else:
            formatted_words.append(word.capitalize())
    
    return ' '.join(formatted_words)

def format_number(num):
    return f"{float(num * 100):.1f}"

def bold_if_best(value, best_value):
    if value == best_value:
        return r"\textbf{" + format_number(value) + "}"
    return format_number(value)

def generate_latex_tables(csv_file):
    models = {'normal': {}, 'cross_lingual': {}}
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            model = format_model_name(row['Model'])
            type_ = row['Type']
            if model not in models[type_]:
                models[type_][model] = {'fas': {}, 'zho': {}, 'rus': {}}
            
            if type_ == 'normal':
                lang = row['Language Pair'][:3]
            else:  # cross_lingual
                lang = row['Language Pair'].split('-')[2][:3]
            
            models[type_][model][lang]['ndcg@20'] = float(row['ndcg@20'])
            models[type_][model][lang]['p-MRR'] = float(row['p-MRR'])

    latex_tables = {}
    for type_ in ['normal', 'cross_lingual']:
        # Calculate average p-MRR for each model
        for model, data in models[type_].items():
            avg_pmrr = statistics.mean(data[lang]['p-MRR'] for lang in ['fas', 'zho', 'rus'])
            models[type_][model]['avg_pmrr'] = avg_pmrr

        # Sort models by average p-MRR
        sorted_models = sorted(models[type_].items(), key=lambda x: x[1]['avg_pmrr'])

        latex = r"""\begin{table*}[t]
\centering
%\resizebox{\textwidth}{!}{%
\begin{tabular}{l|cc|cc|cc|cc}
\toprule
 \multirow{3}{*}{Model} & \multicolumn{8}{c}{mFollowIR """ + (f"({type_.replace('_', ' ').title()})" if type_ == "cross_lingual" else "") + r"""} \\
 & \multicolumn{2}{c}{Farsi} & \multicolumn{2}{c}{Chinese} & \multicolumn{2}{c|}{Russian} & \multicolumn{2}{c}{Average} \\
 & nDCG@20 & p-MRR & nDCG@20 & p-MRR & nDCG@20 & p-MRR & nDCG@20 & p-MRR \\
\midrule
"""

        best_scores = {lang: {'ndcg@20': 0, 'p-MRR': -float('inf')} for lang in ['fas', 'zho', 'rus', 'Avg']}

        for model, data in models[type_].items():
            avg_ndcg = statistics.mean(data[lang]['ndcg@20'] for lang in ['fas', 'zho', 'rus'])
            avg_pmrr = statistics.mean(data[lang]['p-MRR'] for lang in ['fas', 'zho', 'rus'])
            
            for lang in ['fas', 'zho', 'rus', 'Avg']:
                if lang != 'Avg':
                    ndcg = data[lang]['ndcg@20']
                    pmrr = data[lang]['p-MRR']
                else:
                    ndcg = avg_ndcg
                    pmrr = avg_pmrr
                
                if ndcg > best_scores[lang]['ndcg@20']:
                    best_scores[lang]['ndcg@20'] = ndcg
                if pmrr > best_scores[lang]['p-MRR']:
                    best_scores[lang]['p-MRR'] = pmrr

        for model, data in sorted_models:
            row = f"{model} & "
            for lang in ['fas', 'zho', 'rus']:
                ndcg = data[lang]['ndcg@20']
                pmrr = data[lang]['p-MRR']
                row += f"{bold_if_best(ndcg, best_scores[lang]['ndcg@20'])} & "
                row += f"{bold_if_best(pmrr, best_scores[lang]['p-MRR'])} & "
            
            avg_ndcg = statistics.mean(data[lang]['ndcg@20'] for lang in ['fas', 'zho', 'rus'])
            avg_pmrr = statistics.mean(data[lang]['p-MRR'] for lang in ['fas', 'zho', 'rus'])
            row += f"{bold_if_best(avg_ndcg, best_scores['Avg']['ndcg@20'])} & "
            row += f"{bold_if_best(avg_pmrr, best_scores['Avg']['p-MRR'])} \\\\"
            
            latex += row + "\n"

        latex += r"""\bottomrule
\end{tabular}
%}
\caption{Results for mFollowIR dataset """ + (f"({type_.replace('_', ' ').title()} Retrieval)" if type_ == "cross_lingual" else "") + r""" across three language subsets (Farsi, Chinese, Russian). nDCG@20 and p-MRR scores are reported. Best value in each column is bolded. Models are sorted by average p-MRR, with the highest at the bottom.}
\label{tab:mfollowir_""" + type_ + r"""}
\vspace{-0.5em}
\end{table*}"""

        latex_tables[type_] = latex

    return latex_tables

# Usage
csv_file = 'results_summary.csv'
latex_tables = generate_latex_tables(csv_file)

print("Normal Table:")
print(latex_tables['normal'])
print("\nCross-lingual Table:")
print(latex_tables['cross_lingual'])