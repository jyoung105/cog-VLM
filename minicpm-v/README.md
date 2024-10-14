# MiniCPM-V

MiniCPM-V: A GPT-4V Level MLLM on Your Phone

## Reference

- [arxiv](https://arxiv.org/abs/2408.01800)
- [github](https://github.com/OpenBMB/MiniCPM-V)
- [hugging face](https://huggingface.co/openbmb/MiniCPM-V-2_6)
- [hugging face-int4](https://huggingface.co/openbmb/MiniCPM-V-2_6-int4)
- [hugging face-gguf](https://huggingface.co/openbmb/MiniCPM-V-2_6-gguf)

## Try

1. clone repo
```
git clone https://github.com/jyoung105/cog-VLM
```

2. move to directory
```
cd ./cog-VLM/minicpm-v
```

<!-- 3. download weights before deployment
```
cog run script/download-weights
``` -->

3. predict to inference
```
cog predict -i prompt="Describe an image." -i image="FILE_OR_PATH"
```