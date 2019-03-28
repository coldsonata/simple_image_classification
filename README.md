# image_classification

### For storage and sharing of python files

### Downloading images
To reproduce results, you need to download the images with the following commands.
```py
python flickr_GetUrl.py [TAG] [NUMBER OF IMAGE URLS TO SCRAPE]
```
This downloads the image urls into a csv file. The urls will be attempted to be scrapped at their highest available size.

For our case we will use the tags Sensational and Drab, and the number of images we want is 5000. Note that the number of images we end up downloading will be lesser than our stated amount because of flickr not letting us download them.

Following that we have to use the csv files produced by the previous code to download the images. It can be done in the following way.
```py
python get_images.py [FILE NAME]
```

The images will be downloaded into the folder [FILE NAME]. If the folder does not exist, it will be created.

### Augmenting the images
The number of Drab images we managed to download is much lesser than the number of Sensational images. We use a simple data augmentation technique and randomly flip images in Drab by 180 degrees until we have an equal amount of images in both categories. We use the following python script for it.
```py
python augment_drab.py
```
### Splitting the images into Training, Validation, and Testing images
We can split them with the following script.

```py
python split_data.py
```
### Train the model
Finally, we can train the model. The model used here is NASNetLarge, and trained through transfer learning. There might be a variety of warnings
```py
python train_model.py [MODEL_SAVE_NAME]
```

The model will be saved as [MODEL_SAVE_NAME] in the same folder.

