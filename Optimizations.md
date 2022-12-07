Batch Normalization
Is used for training very deep NN. It normalizes the contribution to a layer for every mini-batch. It decreases the issue of planning updates across numerous layers.The effect is that its settling the learning rate and decreases the number of training epochs. Overall it helps with the learability of the network.

Data augmentation
With data augmentation we take the data we have, change it up a little bit to get "new" training data. This augmentation could be to mirror or rotate the images. It is also possible to change the colors, as long as the color is not a distinguishing criteria.
With more diverse data, the model has a better opportunity to train and improve the model prediction accuracy.

Dropout
Dropout describes a random dropout of units or feature maps. By dropping for example random neurons in a layer we interfere with the network in a way that it can not "rely" on specific neuros to do the job. Overall it avoids memorizing the training data.

Simpler models
By making the model simpler and using less parameters or stronger inductive biases we make it more sensible to the task at hand. And therefore more likely to not overfit.

Regularization
The regularization adds a penalty for the loss function with the goal to minimize the parameter magnitude.
There are 2 functions - L1 and L2.
The both modify the global loss to bring the optimizer algorithm in the desired direction.

Label smoothing
It is used so the system does not learn a categorization, were there is only 1 right answer. Instead we change to soft probability labels so almost right decisions get a rather high probability close to 1 and also count towards the right solution. Synonymisly, wrong answers get a rather low probability value close to zero instead of just zero.

Early stopping
Early stopping is used to avoid overfitting by stopping the training process as soon as the validation loss is consistently increasing.
