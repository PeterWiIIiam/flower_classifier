# Flower Classifier
This is a flower classifier for the kaggle competition. The best accuracy is currently 85%.

## How to use the model
### Install required packages 
boto, boto3, botocore, numpy, pillow, tensorflow (1.4.0)
```
sh requirements.sh
```


### Download dataset and trained model
```
python download.py
```
1. Download flower images from the link: 
https://www.kaggle.com/alxmamaev/flowers-recognition
2. Download vgg16_weights.npz from https://www.cs.toronto.edu/~frossard/post/vgg16/
3. Move it under the depository folder
4. Resize and split the image data into 
training and testing set. Create the corresponding TFRecord 
to reduce RAM requirement
```
python build_TFRecord.py    --mode [train, test] \
                            --partition_size [1000]
```

### Train the model
```
python main.py --train True \
               --test False \
               --epoch number_epoch_to_be_trained \
               --learning_rate learning_rate_in_training \
               --checkpoint_dir directory_to_save_model 
```

### Test the model
```
python main.py --train False \
               --test True \
               --epoch number_epoch_to_be_trained \
               --learning_rate learning_rate_in_training \
               --checkpoint_dir model_directory
```
### Use the model to classify images
```
python main.py --input_images_dir input_images'_directory \ 
               --checkpoint_dir model_directory 
```
--input_images_dir flag can receive a directory as well as the path to a single image.

Have fun with flowers!

