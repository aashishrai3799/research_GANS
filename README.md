# research_GANS

## 8x.py:
Main file to train the defined architecture.
In file u'll need to provide address of 3 files:
X_train: LR images in nx24x24x3 npy format,
Y_image: HR(8x) images in nx192x192x3 npy format,
Y_train: Labels (will be used in case of classification)

## benchmark.py:
Mainly used to calculate PSNR and SSIM

## blur.py:
Generate augmented images. Augmentation includes basic operations like H-shift, V-shift, Brightness up/down, blurr, etc.

## create_dataset.py
File to generate dataset in npy format (as described above in 8x.py). User need to provide the path to input dataset folder containing different classes.

## detect_images.py
Use MTCNN to detect faces.

## sr_re.py
File used for precidction. User needs to specify the address to the trained checkpoint and testing dataset.

### The rest of the files were not used directly. Maybe sometimes I may have copied some functions from them. To run the model as described in the paper, the above mentioned files are sufficient.

Link to the pre-trained models: https://lmi.fe.uni-lj.si/en/research/fh/

