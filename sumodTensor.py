import tensorflow as tf

mnist = tf.keras.datasets.mnist 

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.Sequential() 
model.add(tf.keras.layers.Flatten()) 
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu)) 
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu)) 
model.add(tf.keras.layers.Dense(10,activation=tf.nn.softmax))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy']) #parameter for training of the model

model.fit(x_train, y_train, epochs=3) #training the models


val_loss, val_acc = model.evaluate(x_test, y_test) #evaluating the efficiency
print(val_loss, val_acc)




import matplotlib.pyplot as plt
plt.imshow(x_train[0], cmap = plt.cm.binary)
plt.show()
print(x_train[0])


model.save('number_reader.model') #saving a model
new_model = tf.keras.models.load_model('number_reader.model') #load the model


predictions = new_model.predict([x_test])


print(predictions)


import numpy as np
print(np.argmax(predictions[0]))



plt.imshow(x_test[0])
plt.show()

