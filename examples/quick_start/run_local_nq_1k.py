import argparse
from flashrag.config import Config
from flashrag.utils import get_dataset
from flashrag.pipeline import SequentialPipeline
from flashrag.prompt import PromptTemplate

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str)
parser.add_argument("--retriever_path", type=str)
parser.add_argument("--gpu_id", type=str, default="3")
args = parser.parse_args()

config_dict = {
    "gpu_id": args.gpu_id,
    "data_dir": "/home/guanjunjie/my_datasets/FlashRAG_datasets/",
    "index_path": "/home/guanjunjie/my_datasets/FlashRAG_datasets/indexes/test_e5/e5_Flat.index",
    "corpus_path": "/home/guanjunjie/my_datasets/FlashRAG_datasets/retrieval-corpus/wiki18_100w_1k.jsonl",
    "model2path": {
        "e5": args.retriever_path,
        "llama3-8B-instruct": args.model_path,
    },
    "generator_model": "llama3-8B-instruct",
    "retrieval_method": "e5",
    "metrics": ["em", "f1", "acc"],
    "retrieval_topk": 5,
    "save_intermediate_data": True,
}

config = Config(config_dict=config_dict)

all_split = get_dataset(config)
test_data = all_split["test"]
prompt_templete = PromptTemplate(
    config,
    system_prompt="Answer the question based on the given document. \
                    Only give me the answer and do not output any other words. \
                    \nThe following are given documents.\n\n{reference}",
    user_prompt="Question: {question}\nAnswer:",
)


pipeline = SequentialPipeline(config, prompt_template=prompt_templete)


output_dataset = pipeline.run(test_data, do_eval=True)
print("---generation output---")
print(output_dataset.pred)


#python simple_pipeline.py --model_path /home/guanjunjie/my_models/Qwen2.5-7B-Instruct --retriever_path /home/guanjunjie/my_models/e5-base-v2
#python run_local_nq_1k.py --gpu_id 3 --model_path /home/guanjunjie/my_models/Llama-3.1-8B-Instruct --retriever_path /home/guanjunjie/my_models/e5-base-v2
