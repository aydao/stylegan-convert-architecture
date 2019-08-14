# stylegan-convert-architecture

This repository contains a script that can convert any @NVlabs StyleGAN pkl into the analogous architecture that uses vanilla Tensorflow checkpoint, courtesy of @taki0112 ([in this excellent repository](https://github.com/taki0112/StyleGAN-Tensorflow)). Note: this script assumes that the input pkl is a model that has finished progressive growing, and that it will not grow any more, which is the typical case for transfer learning on a StyleGAN.

**Why would anyone be interested in converting this?** For one, the NVlabs code requires using their dataset_tool script which must expand the source dataset by at least 10x. The taki0112 implemenation, however, works directly on raw images. Secondly, the NVlabs code requires their "dnnlib" codebase, while taki0112 simply uses a single clean and readable Tensorflow file. Finally, NVlabs released their StyleGAN code and trained models under an [Attribution-NonCommercial 4.0 International](https://github.com/NVlabs/stylegan/blob/master/LICENSE.txt) license, while taki0112 uses the more permissive [MIT](https://github.com/taki0112/StyleGAN-Tensorflow/blob/master/LICENSE) license.

**Recommended usage:** let's say you've trained a NVlabs StyleGAN and want to transfer/retrain on a considerably larger dataset. Using this script, you can copy over your learned weights and begin training using [this code](https://github.com/taki0112/StyleGAN-Tensorflow) on a dataset up to ~10x larger.
