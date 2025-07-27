import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

class ABU(layers.Layer):
    def __init__(self):
        super(ABU, self).__init__()
        self.alpha = tf.Variable(initial_value=[0.25, 0.25, 0.25, 0.25],
                                 trainable=True, dtype=tf.float32)

    def call(self, x):
        alpha = tf.nn.softmax(self.alpha)
        relu = tf.nn.relu(x)
        elu = tf.nn.elu(x)
        swish = tf.nn.swish(x)
        tanh = tf.nn.tanh(x)
        return (alpha[0] * relu +
                alpha[1] * elu +
                alpha[2] * swish +
                alpha[3] * tanh)

def create_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), padding='same', input_shape=(32, 32, 3)),
        layers.BatchNormalization(),
        ABU(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        ABU(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        ABU(),

        layers.Flatten(),
        layers.Dense(128),
        ABU(),
        layers.Dropout(0.4),

        layers.Dense(10, activation='softmax')
    ])
    return model

optimizers = {
    #'SGD': tf.keras.optimizers.SGD(),
    #'Momentum': tf.keras.optimizers.SGD(momentum=0.9),
    #'RMSProp': tf.keras.optimizers.RMSprop(),
    'Adam': tf.keras.optimizers.Adam()
}

results = {}
best_model = None
best_acc = 0.0
best_name = ""

for name, optimizer in optimizers.items():
    print(f'\nTraining with {name} optimizer...')
    model = create_model()
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train,
                        epochs=15,
                        batch_size=64,
                        validation_data=(x_test, y_test),
                        verbose=1)

    results[name] = history

    val_acc = max(history.history['val_accuracy'])
    if val_acc > best_acc:
        best_acc = val_acc
        best_model = model
        best_name = name

best_model.save('best_cifar10_model_with_abu.h5')
print(f"\nBest model ({best_name}) saved as 'best_cifar10_model_with_abu.h5'.")

plt.figure(figsize=(12, 6))
for name, history in results.items():
    plt.plot(history.history['val_accuracy'], label=name)
plt.title('Optimizer Comparison on CIFAR-10 with ABU Activation')
plt.xlabel('Epochs')
plt.ylabel('Validation Accuracy')
plt.ylim([0, 1])
plt.legend()
plt.grid()
plt.show()

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

def display_sample_prediction(model, x_test, y_test):
    index = np.random.randint(0, x_test.shape[0])
    sample_image = x_test[index]
    true_label = np.argmax(y_test[index])

    sample_image_reshaped = np.expand_dims(sample_image, axis=0)
    prediction = model.predict(sample_image_reshaped)
    predicted_label = np.argmax(prediction)

    plt.imshow(sample_image)
    plt.title(f'True: {class_names[true_label]}, Predicted: {class_names[predicted_label]}')
    plt.axis('off')
    plt.show()

display_sample_prediction(best_model, x_test, y_test)
