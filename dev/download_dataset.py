import os
from huggingface_hub import hf_hub_download

# 创建 data 目录
os.makedirs("../data", exist_ok=True)

# 方法一：使用环境变量（推荐）
# os.environ["HTTP_PROXY"] = "socks5h://127.0.0.1:10800"    # HTTP 流量
# os.environ["HTTPS_PROXY"] = "socks5h://127.0.0.1:10800"   # HTTPS 流量

# TinyStories 文件
tiny_files = [
    "TinyStoriesV2-GPT4-train.txt",
    "TinyStoriesV2-GPT4-valid.txt"
]

for filename in tiny_files:
    print(f"Downloading {filename}...")
    hf_hub_download(
        repo_id="roneneldan/TinyStories",
        filename=filename,
        repo_type="dataset",
        local_dir="data",
        # local_dir_use_symlinks=False,  # Downloading to a local directory does not use symlinks anymore.
        #  参数已弃用，无需设置
    )


# OWT 文件（注意：这些是 .gz 压缩包）
owt_files = ["owt_train.txt.gz", "owt_valid.txt.gz"]
owt_files = []

for filename in owt_files:
    print(f"Downloading {filename}...")
    hf_hub_download(
        repo_id="stanford-cs336/owt-sample",
        filename=filename,
        repo_type="dataset",
        local_dir="data",
    )

# 解压 .gz 文件
import gzip
import shutil

for gz_file in owt_files:
    src = f"../data/{gz_file}"
    dst = f"../data/{gz_file[:-3]}"  # 去掉 .gz
    print(f"Decompressing {gz_file}...")
    with gzip.open(src, 'rb') as f_in:
        with open(dst, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    os.remove(src)  # 可选：删除压缩包

print("✅ 所有文件下载并解压完成！")