# Phi-vision

Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone

## Reference

- [blog](https://techcommunity.microsoft.com/t5/ai-azure-ai-services-blog/discover-the-new-multi-lingual-high-quality-phi-3-5-slms/ba-p/4225280)
- [arxiv](https://arxiv.org/abs/2404.14219)
- [github](https://github.com/microsoft/Phi-3CookBook)
- [hugging face](https://huggingface.co/microsoft/Phi-3.5-vision-instruct)

## Try

1. clone repo
```
git clone https://github.com/jyoung105/cog-VLM
```

2. move to directory
```
cd ./cog-VLM/phi-vision
```

<!-- 3. download weights before deployment
```
cog run script/download-weights
``` -->

3. predict to inference
```
cog predict -i prompt="Describe an image." -i image="FILE_OR_PATH"
```