# Process
### Different models
Different keras models were used to find the best performing one. The VGG16, VGG19, InceptionResNetV2, InceptionV3, and NASNetLarge models were trained.

It was found that the model with the best test accuracy was the NASNetLarge model.

Trying to train the model by unfreezing the weights of the hidden layers yielded similar accuracy as freezing them, and so it was decided to freeze the weights to save training time.

### Different hyper-parameters
Once the best model was narrowed down, different hyper-parameters were briefly tested. Different pooling layers, batch size, and activation functions were tested. 

Because it was a brief test, only GlobalAveragePooling and GlobalMaxPooling were tested for pooling layers. Batch size was tested between 16, 32 and 64, while the Sigmoid and Softmax activation functions were tested.

In the end, it was found that a final layer with GlobalAveragePooling and a Softmax activation function, together with a batch size of 16 yielded the best results.

### Different images
Replacing the Drab tagged images with Dull tagged images yielded a much lower accuracy. This might likely be because the difference between Sensational and Dull images are not as distinct as that of Sensational and Drab images.

### Test Accuracy
Overall, the test acccuracy was around 85% and can be downloaded here: https://drive.google.com/file/d/1k8AX7WQqz3DsuX1L3DgTyvkKSzyeQYbb/view

The model might be further improved through ensemble learning, or a more detailed hyper-parameter grid search.
