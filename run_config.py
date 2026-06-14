import torch

from mmengine.config import read_base
from opencompass.partitioners import NumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask

custom_imports = dict(imports=['arkvale.chat_arkvale'], allow_failed_imports=False)

with read_base():
    from opencompass.configs.datasets.text2json.text2json import text2json_datasets
    
    from opencompass.configs.datasets.needlebench_v2.needlebench_v2_128k.needlebench_v2_multi_retrieval_128k import (
        needlebench_en_datasets,
    )
    from opencompass.configs.datasets.longbench.longbenchpassage_retrieval_en.longbench_passage_retrieval_en_gen import LongBench_passage_retrieval_en_datasets
    
    from opencompass.configs.datasets.longproc import (
        longproc_datasets,
    )
datasets = needlebench_en_datasets # longproc_datasets # text2json_datasets # needlebench_en_datasets # LongBench_passage_retrieval_en_datasets  

models = [
    dict(
        type='ArkValeChatBot',
        path='meta-llama/Llama-3.1-8B-Instruct',
        batch_size=1,
        run_cfg=dict(num_gpus=1),
        page_size=16,
        token_budget=0.015625 / 2,
        page_budgets=None,
        page_topks=None,
        n_unlimited_layers=0,
        n_sink_pages=2,
        n_win_pages=5,
        group_size=1,
        n_max_bytes=20 * (1 << 30),
        n_max_cpu_bytes=20 * (1 << 30),
        max_seq_len=128 * 1024,
        max_out_len=9216,
        abbr='llama-3.1-8B',
    ),
    dict(
        type='ArkValeChatBot',
        path='meta-llama/Llama-3.2-3B-Instruct',
        batch_size=1,
        run_cfg=dict(num_gpus=1),
        page_size=16,
        token_budget=0.015625 / 2,
        page_budgets=None,
        page_topks=None,
        n_unlimited_layers=0,
        n_sink_pages=2,
        n_win_pages=5,
        group_size=1,
        n_max_bytes=20 * (1 << 30),
        n_max_cpu_bytes=20 * (1 << 30),
        max_seq_len=128 * 1024,
        max_out_len=9216,
        abbr='llama-3.2-3B',
    ),
    dict(
        type='ArkValeChatBot',
        path='Qwen/Qwen3-4B-Instruct-2507',
        batch_size=1,
        run_cfg=dict(num_gpus=1),
        page_size=16,
        token_budget=0.015625 / 2,
        page_budgets=None,
        page_topks=None,
        n_unlimited_layers=0,
        n_sink_pages=2,
        n_win_pages=5,
        group_size=1,
        n_max_bytes=20 * (1 << 30),
        n_max_cpu_bytes=20 * (1 << 30),
        max_seq_len=128 * 1024,
        max_out_len=9216,
        abbr='qwen3-4B',
    ),
]

infer = dict(
    partitioner=dict(
        type=NumWorkerPartitioner,
        num_worker=1,
        num_split=1,
        force_rebuild=True,
        keep_keys=[
            'custom_imports',
            'eval.runner.task.judge_cfg',
            'eval.runner.task.dump_details',
            'eval.given_pred',
            'eval.runner.task.cal_extract_rate',
        ],
    ),
    runner=dict(
        type=LocalRunner,
        max_num_workers=1,
        retry=5,
        task=dict(type=OpenICLInferTask),
    ),
)
