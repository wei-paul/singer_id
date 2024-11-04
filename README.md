# . It's a crucial skill in machine learning. Here's how we can infer overfitting from the results:

Perfect training accuracy (1.0000) vs. lower validation accuracy (0.3500): The model performs flawlessly on training data but much worse on unseen data.

Very low training loss (0.0107) vs. high validation loss (4.5547): The large gap between training and validation loss indicates the model isn't generalizing well.

Overfitting occurs when a model learns the training data too well, including its noise and peculiarities, rather than learning the underlying patterns that generalize to new data. It's like memorizing answers instead of understanding the subject.

To recognize overfitting in the future, look for:

A large gap between training and validation metrics
Training metrics that keep improving while validation metrics plateau or worsen
Near-perfect performance on training data but poor performance on new data



Creating a model right now, stuck on some things. Is there a base template for CNN models for speech recognition and then modify it from there, or create it from scratch. 
how do you tell if your model is too complex?
The idea to approach CNN is the same throughout whatever problem you're trying to solve right? 
Feature extraction. using a filter, learn the "features" and at the same time compress the feature map into smaller and smaller dimensions? 

What have I learned so far after my 4th attempt at understanding/recreating the model:
- Sequencial() is called if your model consists of linear layers (single input, and single output.). So for example Multiple inputs: A model that takes both an image and text as input to classify a product.
Multiple outputs: A model that predicts both the genre and the artist of a song simultaneously.
- dropout is used to prevent overfitting (learning the training data too well), so it randomly drops % of neurons during training
- flatten() flattens the 3D output from the convolutional layers into a 1D vector, preparing it for the dense layers


Playing around with the model:

- when increasing the dropout % due to troubleshooting overfitting. the accuracy completely drops. So does that mean that it's learning STRICTLY and very dependent on the training data (linear learn, not non-linear.)
(accuracy decreases by ~20%, val_loss and val_accuracy stayed the same. so that means the problem is most likely not the dropout)
E
Epoch 199/200
1/1 [==============================] - 0s 36ms/step - loss: 1.8768 - accuracy: 0.3375 - val_loss: 2.3741 - val_accuracy: 0.1500
Epoch 200/200
1/1 [==============================] - 0s 37ms/step - loss: 1.7263 - accuracy: 0.4500 - val_loss: 2.3514 - val_accuracy: 0.2000
- But when I try running again (it was INITIALLY 64%), it dropped all the way to like 50 even though same setup @@
poch 196/200
Epoch 199/200
1/1 [==============================] - 0s 37ms/step - loss: 1.3239 - accuracy: 0.5750 - val_loss: 2.9311 - val_accuracy: 0.2500
Epoch 200/200
1/1 [==============================] - 0s 35ms/step - loss: 1.3579 - accuracy: 0.5500 - val_loss: 2.7667 - val_accuracy: 0.2500
- NEXT I tried adding this "regularizor" and accuracy just dropped. We can somewhat infer from this that The model may have been relying heavily on memorizing training data before.
Epoch 199/200
1/1 [==============================] - 0s 37ms/step - loss: 2.2377 - accuracy: 0.2125 - val_loss: 2.3854 - val_accuracy: 0.0500
Epoch 200/200
1/1 [==============================] - 0s 38ms/step - loss: 2.2195 - accuracy: 0.1875 - val_loss: 2.4206 - val_accuracy: 0.0500
- NEXT I tried, I played around with the batch size, decreased it from 100 to 50. 
Epoch 199/200
2/2 [==============================] - 0s 62ms/step - loss: 0.7274 - accuracy: 0.7875 - val_loss: 2.6743 - val_accuracy: 0.3000
Epoch 200/200
2/2 [==============================] - 0s 31ms/step - loss: 0.6367 - accuracy: 0.8125 - val_loss: 2.8470 - val_accuracy: 0.3000
- AFTER understanding how batch size work, and decide to train on 10 samples at a time x10 times for an epoch
Epoch 199/200
8/8 [==============================] - 0s 7ms/step - loss: 0.1188 - accuracy: 0.9750 - val_loss: 3.9542 - val_accuracy: 0.2500
Epoch 200/200
8/8 [==============================] - 0s 7ms/step - loss: 0.3420 - accuracy: 0.8625 - val_loss: 4.1553 - val_accuracy: 0.2000
*it seems like the accuracy shot up here. but the actual training data is still ass, so let's try adding back the regularizor thing
Epoch 199/200
8/8 [==============================] - 0s 7ms/step - loss: 2.2964 - accuracy: 0.1250 - val_loss: 2.4113 - val_accuracy: 0.0000e+00
Epoch 200/200
8/8 [==============================] - 0s 8ms/step - loss: 2.2940 - accuracy: 0.1250 - val_loss: 2.4110 - val_accuracy: 0.0000e+00
- Okay, so CLEARLY regularizor is a no go here, OR I need to decrease the regularizor here, from 0.01 to 0.0001
Epoch 199/200
8/8 [==============================] - 0s 10ms/step - loss: 0.7895 - accuracy: 0.7250 - val_loss: 2.5845 - val_accuracy: 0.1500
Epoch 200/200
8/8 [==============================] - 0s 7ms/step - loss: 0.7978 - accuracy: 0.7375 - val_loss: 2.5659 - val_accuracy: 0.3000
-  Okay, so now, it seems like it had fix the TERRIBLE accuracy which I think it means underfitting, but now let's modify the regularizor a bit more to 0.00005
Epoch 199/200
8/8 [==============================] - 0s 7ms/step - loss: 0.2428 - accuracy: 0.9375 - val_loss: 3.2108 - val_accuracy: 0.4500
Epoch 200/200
8/8 [==============================] - 0s 7ms/step - loss: 0.2689 - accuracy: 0.9250 - val_loss: 3.2703 - val_accuracy: 0.5000
- DEFINITELY improving. but my val_loss is still bad. Going to try adding batch normalization. and maybe another Conv2D layer.
(3 conv2d layer, flatten, 2 dense layer.)
Epoch 199/200
8/8 [==============================] - 0s 9ms/step - loss: 0.0535 - accuracy: 1.0000 - val_loss: 4.3164 - val_accuracy: 0.2000
Epoch 200/200
8/8 [==============================] - 0s 9ms/step - loss: 0.0542 - accuracy: 1.0000 - val_loss: 4.2739 - val_accuracy: 0.2000
- Bruh. okay. so my training data is absolutely insane, it learns training data SUPER well, but clearly still not learning feature. Going to try adjusting learning rate. 
Epoch 199/200
8/8 [==============================] - 0s 9ms/step - loss: 0.6075 - accuracy: 0.9625 - val_loss: 7.0935 - val_accuracy: 0.2000
Epoch 200/200
8/8 [==============================] - 0s 9ms/step - loss: 0.6328 - accuracy: 0.9875 - val_loss: 7.1424 - val_accuracy: 0.2000
- lol. okay val loss just went the opposite direction. going to decrease the learning rate. (default 0.001)
Epoch 199/200
8/8 [==============================] - 0s 9ms/step - loss: 0.7104 - accuracy: 0.8750 - val_loss: 2.3049 - val_accuracy: 0.3500
Epoch 200/200
8/8 [==============================] - 0s 9ms/step - loss: 0.7876 - accuracy: 0.8000 - val_loss: 2.3527 - val_accuracy: 0.3500
- didn't do much to the validations, but made the training data worse. 
**AFTER SEVERAL ADJUSTMENTS TO THE MODEL, made no difference in regards to making the val_accuracy better, or val_loss better. Decided to do data augmentation: applies time stretching, pitch shifting, and noise addition to the MFCC features. The prepare_dataset function is modified to include an augment parameter. When augment is True, it adds an augmented version of each audio file to the dataset, effectively doubling the size of your training data.
Epoch 199/200
16/16 [==============================] - 0s 14ms/step - loss: 0.2636 - accuracy: 0.9438 - val_loss: 7.7558 - val_accuracy: 0.1250
Epoch 200/200
16/16 [==============================] - 0s 13ms/step - loss: 0.2953 - accuracy: 0.9563 - val_loss: 1.4834 - val_accuracy: 0.6750
- it seemed to improve my validation by quite a bit. so let's continue to make data augmentation adjustments
16/16 [==============================] - 0s 9ms/step - loss: 0.1988 - accuracy: 0.9625 - val_loss: 0.3822 - val_accuracy: 0.9250
Epoch 200/200
16/16 [==============================] - 0s 11ms/step - loss: 0.1685 - accuracy: 0.9750 - val_loss: 0.6621 - val_accuracy: 0.7250
- GETTING THERE, so what I did was I modified the number of "features" in the Mfcc extracted from 20 > 13. When tested the epoch (which I didn't save, there were some weird >14 val_loss, which was super weird. I then adjusted thought about "generalizing" it more, since we changed the 20 > 13, so in a way, we shouldn't need that many for filter count right? 32, 64, 128, (reverse for dense layer)).
Epoch 199/200
16/16 [==============================] - 0s 7ms/step - loss: 0.1911 - accuracy: 0.9750 - val_loss: 0.6111 - val_accuracy: 0.8250
Epoch 200/200
16/16 [==============================] - 0s 9ms/step - loss: 0.2581 - accuracy: 0.9312 - val_loss: 0.9591 - val_accuracy: 0.7750
- Okay it seems like it's getting better and better, but still not ideal? going to try randomly just deleting one dense layer
Epoch 199/200
16/16 [==============================] - 0s 7ms/step - loss: 0.0591 - accuracy: 1.0000 - val_loss: 0.3608 - val_accuracy: 0.9000
Epoch 200/200
16/16 [==============================] - 0s 7ms/step - loss: 0.0563 - accuracy: 1.0000 - val_loss: 0.4158 - val_accuracy: 0.9000
- WHOA, okay, not sure... why this works, not sure how changing the last dropout rate from 0.25 to 0.4 would work
16/16 [==============================] - 0s 7ms/step - loss: 0.1078 - accuracy: 0.9688 - val_loss: 1.6765 - val_accuracy: 0.7000
Epoch 200/200
16/16 [==============================] - 0s 7ms/step - loss: 0.0729 - accuracy: 0.9875 - val_loss: 3.5928 - val_accuracy: 0.4750
- OKAY trash, changing back the dropout rate. 
Epoch 199/200
16/16 [==============================] - 0s 7ms/step - loss: 0.0591 - accuracy: 1.0000 - val_loss: 0.3608 - val_accuracy: 0.9000
Epoch 200/200
16/16 [==============================] - 0s 7ms/step - loss: 0.0563 - accuracy: 1.0000 - val_loss: 0.4158 - val_accuracy: 0.9000
- Now i'm curious so, playing around with adding strides to my conv2dlayers
Epoch 199/200
16/16 [==============================] - 1s 39ms/step - loss: 0.1392 - accuracy: 0.9937 - val_loss: 0.5987 - val_accuracy: 0.8500
Epoch 200/200
16/16 [==============================] - 1s 33ms/step - loss: 0.1176 - accuracy: 0.9937 - val_loss: 0.6693 - val_accuracy: 0.8500
- Didn't really change much, okay let's add padding on top of it then. 
Epoch 193/200
16/16 [==============================] - 1s 39ms/step - loss: 0.1542 - accuracy: 1.0000 - val_loss: 0.5602 - val_accuracy: 0.9000
Epoch 194/200
16/16 [==============================] - 1s 52ms/step - loss: 0.1766 - accuracy: 0.9875 - val_loss: 0.5236 - val_accuracy: 0.8750
- Also didn't change much HOWEVER, for some reason adding padding + stride gets rid of random >5 loss or something. Let's try 0.5, 0.3, 0.3 for dropout
Epoch 199/200
16/16 [==============================] - 1s 62ms/step - loss: 0.2057 - accuracy: 0.9875 - val_loss: 0.7473 - val_accuracy: 0.8250
Epoch 200/200
16/16 [==============================] - 1s 47ms/step - loss: 0.2697 - accuracy: 0.9688 - val_loss: 0.6972 - val_accuracy: 0.8250
- Still nothing really has changed but by now my model is super stable, don't see any random >5 losses. What happens if I got rid of batch normalization
Epoch 199/200
16/16 [==============================] - 1s 68ms/step - loss: 2.2938 - accuracy: 0.1187 - val_loss: 2.3921 - val_accuracy: 0.0250
Epoch 200/200
16/16 [==============================] - 1s 65ms/step - loss: 2.2921 - accuracy: 0.1187 - val_loss: 2.3917 - val_accuracy: 0.0250
-OOPS okay. Bringing it back and then modifying regularizer 0.00005 > 0.0005
Epoch 199/200
16/16 [==============================] - 1s 60ms/step - loss: 0.6986 - accuracy: 0.9500 - val_loss: 1.2892 - val_accuracy: 0.7750
Epoch 200/200
16/16 [==============================] - 1s 55ms/step - loss: 0.6699 - accuracy: 0.9625 - val_loss: 4.2070 - val_accuracy: 0.2750
- Okay, I guess regularizer also affects random sporadic spike of val_loss. Changing it back. 
Epoch 199/200
16/16 [==============================] - 1s 73ms/step - loss: 0.2369 - accuracy: 0.9688 - val_loss: 0.8520 - val_accuracy: 0.8500
Epoch 200/200
16/16 [==============================] - 1s 55ms/step - loss: 0.1646 - accuracy: 0.9937 - val_loss: 0.8707 - val_accuracy: 0.8500


* Things to add to my model
- Regularizor
- Data augmentation??
- Different dropout rates

ask about dropout rates, 

ask about regularizor,

ask about to which degree your model should look like,
