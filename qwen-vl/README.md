# Qwen-VL

Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution

## Reference

- [project](https://qwenlm.github.io/blog/qwen2-vl/)
- [demo](https://huggingface.co/spaces/Qwen/Qwen2-VL)
- [arxiv](https://arxiv.org/abs/2409.12191)
- [hugging face-2b](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct)
- [hugging face-7b](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct)
- [hugging face-72b](https://huggingface.co/Qwen/Qwen2-VL-72B-Instruct)

## Try

1. clone repo
```
git clone https://github.com/jyoung105/cog-VLM
```

2. move to directory
```
cd ./cog-VLM/qwen-vl
```

<!-- 3. download weights before deployment
```
cog run script/download-weights
``` -->

3. predict to inference
```
cog predict -i prompt="Describe an image." -i image="FILE_OR_PATH"
```