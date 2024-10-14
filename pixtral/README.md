# Pixtral

Pixtral 12B

## Reference

- [project](https://mistral.ai/news/pixtral-12b/)
- [arxiv](https://arxiv.org/abs/2410.07073)
- [hugging face](https://huggingface.co/mistralai/Pixtral-12B-2409)

## Try

1. clone repo
```
git clone https://github.com/jyoung105/cog-VLM
```

2. move to directory
```
cd ./cog-VLM/pixtral
```

<!-- 3. download weights before deployment
```
cog run script/download-weights
``` -->

3. predict to inference
```
cog predict -i prompt="Describe an image." -i image="FILE_OR_PATH"
```