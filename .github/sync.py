import os
import sys
import argparse
from modelscope import HubApi
from huggingface_hub import HfApi

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import gen_dataframe


def sync(mstk: str, hftk: str):
    gen_dataframe()
    for file in ["train.jsonl", "test.jsonl", ".gitattributes"]:
        HubApi().upload_file(
            path_or_fileobj=f"./{file}",
            path_in_repo=file,
            repo_id="monetjoe/cv_backbones",
            repo_type="dataset",
            token=mstk,
        )
        HfApi().upload_file(
            path_or_fileobj=f"./{file}",
            path_in_repo=file,
            repo_id="monetjoe/cv_backbones",
            repo_type="dataset",
            token=hftk,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upd MS studio & HF space")
    parser.add_argument("--mstk", required=True, help="Your ModelScope Access Token")
    parser.add_argument("--hftk", required=True, help="Your HuggingFace Access Token")
    args = parser.parse_args()
    sync(args.mstk, args.hftk)
