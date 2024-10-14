# Molmo

Molmo and PixMo: Open Weights and Open Data for State-of-the-Art Multimodal Models

## Reference

- [project](https://molmo.allenai.org/blog)
- [demo](https://molmo.allenai.org/)
- [arxiv](https://arxiv.org/abs/2409.17146)
- [hugging face-1b-MoE](https://huggingface.co/allenai/MolmoE-1B-0924)
- [hugging face-7b-D](https://huggingface.co/allenai/Molmo-7B-D-0924)
- [hugging face-7b-O](https://huggingface.co/allenai/Molmo-7B-O-0924)
- [hugging face-72b](https://huggingface.co/allenai/Molmo-72B-0924)

## Try

1. clone repo
```
git clone https://github.com/jyoung105/cog-VLM
```

2. move to directory
```
cd ./cog-VLM/molmo
```

<!-- 3. download weights before deployment
```
cog run script/download-weights
``` -->

3. predict to inference
```
cog predict -i prompt="Describe an image." -i image="FILE_OR_PATH"
```