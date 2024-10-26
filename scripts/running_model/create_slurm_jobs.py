"""Creates slurm jobs for running models on all tasks"""

from __future__ import annotations

import subprocess
from collections.abc import Iterable
from pathlib import Path

import mteb


def create_slurm_job_file(
    model_name: str,
    task_name: str,
    results_folder: Path,
    slurm_prefix: str,
    slurm_jobs_folder: Path,
) -> Path:
    """Create slurm job file for running a model on a task"""
    slurm_job = f"{slurm_prefix}\n"
    slurm_job += f"mteb run -m {model_name} -t {task_name} --output_folder {results_folder.resolve()} --co2_tracker true --batch_size 4"

    model_path_name = model_name.replace("/", "__")

    slurm_job_file = slurm_jobs_folder / f"{model_path_name}_{task_name}.sh"
    with open(slurm_job_file, "w") as f:
        f.write(slurm_job)
    return slurm_job_file


def create_slurm_job_files(
    model_names: list[str],
    tasks: Iterable[mteb.AbsTask],
    results_folder: Path,
    slurm_prefix: str,
    slurm_jobs_folder: Path,
) -> list[Path]:
    """Create slurm job files for running models on all tasks"""
    slurm_job_files = []
    for model_name in model_names:
        for task in tasks:
            slurm_job_file = create_slurm_job_file(
                model_name,
                task.metadata.name,
                results_folder,
                slurm_prefix,
                slurm_jobs_folder,
            )
            slurm_job_files.append(slurm_job_file)
    return slurm_job_files


def run_slurm_jobs(files: list[Path]) -> None:
    """Run slurm jobs based on the files provided"""
    for file in files:
        subprocess.run(["sbatch", file])


if __name__ == "__main__":
    # SHOULD BE UPDATED
    slurm_prefix = """#!/bin/bash
#SBATCH --job-name=mteb
#SBATCH --nodes=1
#SBATCH --partition=a3
#SBATCH --gres=gpu:8                 # number of gpus
#SBATCH --time 24:00:00             # maximum execution time (HH:MM:SS)
#SBATCH --output=/data/niklas/jobs/%x-%j.out           # output file name
"""
##SBATCH --exclusive
    project_root = Path(__file__).parent / ".." / ".." / ".."
    # results_folder = project_root / "results"
    results_folder = Path("/data/niklas/results/results")
    slurm_jobs_folder = Path(__file__).parent / "slurm_jobs"

    model_names = [
        #"sentence-transformers/all-MiniLM-L6-v2",
        #"sentence-transformers/all-MiniLM-L12-v2",
        #"sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        #"sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        #"sentence-transformers/all-mpnet-base-v2",
        #"sentence-transformers/LaBSE",
        #"intfloat/multilingual-e5-large-instruct",
        #"intfloat/e5-mistral-7b-instruct",
        #"GritLM/GritLM-7B",
        #"GritLM/GritLM-8x7B",
        #"intfloat/multilingual-e5-small",
        #"intfloat/multilingual-e5-base",
        #"intfloat/multilingual-e5-large",
        "BAAI/bge-m3",
        #"Alibaba-NLP/gte-multilingual-base",
        #"voyage-multilingual-2",
    ]

    # expanding to a full list of tasks
    tasks = mteb.get_tasks(
        #task_types=[
        #    "BitextMining",
        #    "Classification",
        #    "Clustering",
        #    "MultilabelClassification",
        #    "PairClassification",
        #    "Reranking",
        #    "Retrieval",
        #    "InstructionRetrieval",
        #    "STS",
        #    "Summarization",
        #],
        tasks=[
            #"LivedoorNewsClustering",
            #"FaithDial",
            #"STS22",
            #"StatcanDialogueDatasetRetrieval",
            #"WikipediaRetrievalMultilingual"
            #"RARbMath"
            #"Touche2020",
            #"WebLINXCandidatesReranking",
            #"MultiLongDocRetrieval",
            #"CodeEditSearchRetrieval",
            #"BUCC.v2",
            #"OpusparcusPC",

            "DBPediaHardNegatives",
            "MIRACLRetrievalHardNegatives",
            "NeuCLIR2023RetrievalHardNegatives",
            "NeuCLIR2022RetrievalHardNegatives",
            "RiaNewsRetrievalHardNegatives",
            "HotpotQA-PLHardNegatives",
            "DBPedia-PLHardNegatives",
            "NQ-PLHardNegatives",
            "Quora-PLHardNegatives",
            "MSMARCO-PLHardNegatives",
            "ClimateFEVERHardNegatives",
            "QuoraRetrievalHardNegatives",
            "FEVERHardNegatives",
            "HotpotQAHardNegatives",
            "NQHardNegatives",
            "TopiOCQAHardNegatives",
            "MSMARCOHardNegatives",

            "BornholmBitextMining",
            "BibleNLPBitextMining",
            "DiaBlaBitextMining",
            "FloresBitextMining",
            "IN22GenBitextMining",
            "IndicGenBenchFloresBitextMining",
            "NollySentiBitextMining",
            "NorwegianCourtsBitextMining",
            "NTREXBitextMining",
            "NusaTranslationBitextMining",
            "NusaXBitextMining",
            "Tatoeba",
            "BulgarianStoreReviewSentimentClassfication",
            "CzechProductReviewSentimentClassification",
            "GreekLegalCodeClassification",
            "DBpediaClassification",
            "FinancialPhrasebankClassification",
            "PoemSentimentClassification",
            "ToxicConversationsClassification",
            "TweetTopicSingleClassification",
            "EstonianValenceClassification",
            "FilipinoShopeeReviewsClassification",
            "GujaratiNewsClassification",
            "SentimentAnalysisHindi",
            "IndonesianIdClickbaitClassification",
            "ItaCaseholdClassification",
            "KorSarcasmClassification",
            "KurdishSentimentClassification",
            "MacedonianTweetSentimentClassification",
            "AfriSentiClassification",
            "AmazonCounterfactualClassification",
            "CataloniaTweetClassification",
            "CyrillicTurkicLangClassification",
            "IndicLangClassification",
            "MasakhaNEWSClassification",
            "MassiveIntentClassification",
            "MultiHateClassification",
            "NordicLangClassification",
            "NusaParagraphEmotionClassification",
            "NusaX-senti",
            "ScalaClassification",
            "SwissJudgementClassification",
            "NepaliNewsClassification",
            "OdiaNewsClassification",
            "PunjabiNewsClassification",
            "PolEmo2.0-OUT",
            "PAC",
            "SinhalaNewsClassification",
            "CSFDSKMovieReviewSentimentClassification",
            "SiswatiNewsClassification",
            "SlovakMovieReviewSentimentClassification",
            "SwahiliNewsClassification",
            "DalajClassification",
            "TswanaNewsClassification",
            "IsiZuluNewsClassification",
            "WikiCitiesClustering",
            "MasakhaNEWSClusteringS2S",
            "RomaniBibleClustering",
            "ArXivHierarchicalClusteringP2P",
            "ArXivHierarchicalClusteringS2S",
            "BigPatentClustering.v2",
            "BiorxivClusteringP2P.v2",
            "MedrxivClusteringP2P.v2",
            "StackExchangeClustering.v2",
            "AlloProfClusteringS2S.v2",
            "HALClusteringS2S.v2",
            "SIB200ClusteringS2S",
            "WikiClusteringP2P.v2",
            "SNLHierarchicalClusteringP2P",
            "PlscClusteringP2P.v2",
            "SwednClusteringP2P",
            "CLSClusteringP2P.v2",
            "StackOverflowQA",
            "TwitterHjerneRetrieval",
            "AILAStatutes",
            "ArguAna",
            "HagridRetrieval",
            "LegalBenchCorporateLobbying",
            "LEMBPasskeyRetrieval",
            "SCIDOCS",
            "SpartQA",
            "TempReasonL1",
            "TRECCOVID",
            "WinoGrande",
            "BelebeleRetrieval",
            "MLQARetrieval",
            "StatcanDialogueDatasetRetrieval",
            "WikipediaRetrievalMultilingual",
            "CovidRetrieval",
            "Core17InstructionRetrieval",
            "News21InstructionRetrieval",
            "Robust04InstructionRetrieval",
            "KorHateSpeechMLClassification",
            "MalteseNewsClassification",
            "MultiEURLEXMultilabelClassification",
            "BrazilianToxicTweetsClassification",
            "CEDRClassification",
            "CTKFactsNLI",
            "SprintDuplicateQuestions",
            "TwitterURLCorpus",
            "ArmenianParaphrasePC",
            "indonli",
            "OpusparcusPC",
            "PawsXPairClassification",
            "RTE3",
            "XNLI",
            "PpcPC",
            "TERRa",
            "WebLINXCandidatesReranking",
            "AlloprofReranking",
            "VoyageMMarcoReranking",
            "WikipediaRerankingMultilingual",
            "RuBQReranking",
            "T2Reranking",
            "GermanSTSBenchmark",
            "SICK-R",
            "STS12",
            "STS13",
            "STS14",
            "STS15",
            "STSBenchmark",
            "FaroeseSTS",
            "FinParaSTS",
            "JSICK",
            "IndicCrosslingualSTS",
            "SemRel24STS",
            "STS17",
            "STS22.v2",
            "STSES",
            "STSB",
            "SummEvalSummarization.v2",

        ],
    )

    # WE ALSO NEED TO RUN THESE
    retrieval_to_be_downsampled = [
        "TopiOCQA",
        "MSMARCO-PL",
        "ClimateFEVER",
        "FEVER",
        "HotpotQA",
        "HotpotQA-PL",
        "DBPedia",
        "DBPedia-PL",
        "NeuCLIR2022Retrieval",
        "NeuCLIR2023Retrieval",
        "NeuCLIR2022Retrieval",
        "NeuCLIR2023Retrieval",
        "NQ",
        "NQ-PL",
        "NeuCLIR2022Retrieval",
        "NeuCLIR2023Retrieval",
        "MIRACLRetrieval",
        "RiaNewsRetrieval",
        "Quora-PL",
        "QuoraRetrieval",
    ]

    tasks = [t for t in tasks if t.metadata.name not in retrieval_to_be_downsampled]

    slurm_jobs_folder.mkdir(exist_ok=True)
    files = create_slurm_job_files(
        model_names, tasks, results_folder, slurm_prefix, slurm_jobs_folder
    )
    run_slurm_jobs(files)
