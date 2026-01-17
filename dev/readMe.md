

```
#!/bin/bash
mkdir -p data
cd data

PROXY="socks5h://127.0.0.1:10808"

curl -L -x "$PROXY" -O https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
curl -L -x "$PROXY" -O https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

curl -L -x "$PROXY" -O https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz

curl -L -x "$PROXY" -O https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

【问题1】

```
Found. Redirecting to https://cas-bridge.xethub.hf.co/...
```
- 说明 **Hugging Face 的数据集链接现在使用了重定向机制（通常是为了支持大文件分发或 XetHub 存储）**，而你的 `curl` 命令**默认不会自动跟随重定向**（尤其是 HTTP 302/301），除非你显式启用。

- 解决方案：加上 `-L` 参数（让 curl 自动跟随重定向）

> `-L, --location`: If the server reports that the requested page has moved to a different location (indicated with a Location: header and a 3xx response code), this option will make curl redo the request on the new place.

【问题2】

没下完中断, 改用官方库下载


【问题3】

uv venv 安装了一个新版本python，写代码时。vscode里写一个open函数没有提示文档 ？

解决办法：
  `python -c "print(open.__doc__)"` 发现正常有文档
  Ctrl+Shift+P → Python: Clear Cache And Reload Window 之后正常
