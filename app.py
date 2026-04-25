import os
import re
import json
import requests
import gradio as gr
import pandas as pd
from tqdm import tqdm
from bs4 import BeautifulSoup

V_TO_SPLIT = {"IMAGENET1K_V1": "train", "IMAGENET1K_V2": "test"}

EN_US = os.getenv("LANG") != "zh_CN.UTF-8"

EN2ZH = {
    "ImageNet version": "ImageNet 版本",
    "Clean cache": "清理缓存",
    "Status": "状态栏",
    "Download JSON lines": "下载 jsonl",
    "Preview": "预览",
}


def _L(en_txt: str):
    return en_txt if EN_US else f"{en_txt} ({EN2ZH[en_txt]})"


def parse_url(url: str):
    response = requests.get(url)
    html = response.text
    return BeautifulSoup(html, "html.parser")


def special_type(m_ver: str):
    m_type = re.search("[a-zA-Z]+", m_ver).group(0)
    if m_type == "wide" or m_type == "resnext":
        return "resnet"

    elif m_type == "swin":
        return "swin_transformer"

    elif m_type == "inception":
        return "googlenet"

    return m_type


def info_on_dataset(m_ver: str, m_type: str, in1k_span):
    url_span = in1k_span.find_next_sibling("span", {"class": "s2"})
    size_span = url_span.find_next_sibling("span", {"class": "mi"})
    m_url = str(url_span.text[1:-1])
    input_size = int(size_span.text)
    m_dict = {"ver": m_ver, "type": m_type, "input_size": input_size, "url": m_url}
    return m_dict, size_span


def gen_dataframe(url="https://pytorch.org/vision/main/_modules/"):
    torch_page = parse_url(url)
    article = torch_page.find("article", {"id": "pytorch-article"})
    ul = article.find("ul").find("ul")
    in1k_v1, in1k_v2 = [], []
    for li in tqdm(ul.find_all("li"), desc="Crawling cv backbone info..."):
        name = str(li.text)
        if name.__contains__("torchvision.models.") and len(name.split(".")) == 3:
            if name.__contains__("_api") or name.__contains__("feature_extraction"):
                continue

            href = li.find("a").get("href")
            model_page = parse_url(url + href)
            divs = model_page.select("div.viewcode-block")
            for div in divs:
                div_id = str(div["id"])
                if div_id.__contains__("_Weights"):
                    m_ver = div_id.split("_Weight")[0].lower()
                    m_type = special_type(m_ver)
                    in1k_v1_span = div.find(
                        name="span",
                        attrs={"class": "n"},
                        string="IMAGENET1K_V1",
                    )
                    if not in1k_v1_span:
                        continue

                    m_dict, size_span = info_on_dataset(m_ver, m_type, in1k_v1_span)
                    in1k_v1.append(m_dict)
                    in1k_v2_span = size_span.find_next_sibling(
                        name="span",
                        attrs={"class": "n"},
                        string="IMAGENET1K_V2",
                    )
                    if in1k_v2_span:
                        m_dict, _ = info_on_dataset(m_ver, m_type, in1k_v2_span)
                        in1k_v2.append(m_dict)

    dataset = {"IMAGENET1K_V1": in1k_v1, "IMAGENET1K_V2": in1k_v2}
    with open("train.jsonl", "w", encoding="utf-8") as jsonl_file:
        for item in in1k_v1:
            jsonl_file.write(json.dumps(item) + "\n")

    with open("test.jsonl", "w", encoding="utf-8") as jsonl_file:
        for item in in1k_v2:
            jsonl_file.write(json.dumps(item) + "\n")

    return dataset


# outer func
def infer(subset: str):
    status = "Success"
    prewiew = out_json = None
    try:
        cache_json = f"{V_TO_SPLIT[subset]}.jsonl"
        if os.path.exists(cache_json):
            with open(cache_json, "r", encoding="utf-8") as jsonl_file:
                dataset = [json.loads(line) for line in jsonl_file]

        else:
            dataset = gen_dataframe()[subset]

        prewiew = pd.DataFrame(dataset)
        out_json = cache_json

    except Exception as e:
        status = f"{e}"

    return status, prewiew, out_json


# outer func
def sync(subset: str):
    status = "Success"
    try:
        cache_json = f"{V_TO_SPLIT[subset]}.jsonl"
        if os.path.exists(cache_json):
            os.remove(cache_json)

        if os.path.exists(cache_json):
            raise FileExistsError(f"Failed to clean {cache_json}")

    except Exception as e:
        status = f"{e}"

    return status, None


if __name__ == "__main__":
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                subset_opt = gr.Dropdown(
                    label=_L("ImageNet version"),
                    choices=["IMAGENET1K_V1", "IMAGENET1K_V2"],
                    value="IMAGENET1K_V1",
                )
                sync_btn = gr.Button(_L("Clean cache"))

            with gr.Column():
                status_bar = gr.Textbox(label=_L("Status"), buttons=["copy"])
                dld_file = gr.File(label=_L("Download JSON lines"))

        with gr.Row():
            data_frame = gr.Dataframe(label=_L("Preview"))

        subset_opt.change(
            infer,
            inputs=subset_opt,
            outputs=[status_bar, data_frame, dld_file],
        )
        sync_btn.click(sync, inputs=subset_opt, outputs=[status_bar, dld_file])

    demo.launch()
