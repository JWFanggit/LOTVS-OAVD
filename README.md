# LOTVS-OAVD
## Train A_CLIP
a.Download the [pretraining clip model](https://pan.baidu.com/s/1DwBFk1Fr5MHdM25eNFRf4g) code:(mkpx)

b.Download the [MM-AU Benchmark](http://www.lotvsmmau.net)

c.Make data structure and write the root_path (The path of dataset) to A_CLIP/configs/A_CLIP.yaml
>[root_path]
>>1,2,3,....
>>[OOD_train.txt]

d.```Run T_CLIP.py```

e.You can save the trained A_CLIP model for second-stage diffusion model training, or directly use the checkpoint we provide for inferencing.
## Inference OAVD
a.Download our [checkpoint](https://pan.baidu.com/s/1FjYzopBbpbfiQtevznDTgA) code:(st5h)

c.```Run train_and_inference.py```
