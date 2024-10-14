# DeepSeek-VL

DeepSeek-VL: Towards Real-World Vision-Language Understanding

## Reference

- [project](https://www.deepseek.com/)
- [arxiv](https://arxiv.org/abs/2403.05525)
- [github](https://github.com/deepseek-ai/DeepSeek-VL)
- [hugging face-1.3b-base](https://huggingface.co/deepseek-ai/deepseek-vl-1.3b-base)
- [hugging face-1.3b-chat](https://huggingface.co/deepseek-ai/deepseek-vl-1.3b-chat)
- [hugging face-7b-base](https://huggingface.co/deepseek-ai/deepseek-vl-7b-base)
- [hugging face-7b-chat](https://huggingface.co/deepseek-ai/deepseek-vl-7b-chat)

## Try

1. clone repo
```
git clone https://github.com/jyoung105/cog-VLM
```

2. move to directory
```
cd ./cog-VLM/deepseek-vl
```

<!-- 3. download weights before deployment
```
cog run script/download-weights
``` -->

3. predict to inference
```
cog predict -i prompt="Describe an image." -i image="FILE_OR_PATH"
```