from platform import python_version
python_version()
import tensorflow as tf
from tensorflow import keras
tf.__version__, keras.__version__

class ResNet_Model():
    def __init__(self):
        from tensorflow.keras.applications.resnet50 import ResNet50
        input_t = keras.Input(shape=(128, 128, 3))
        self.base_model = ResNet50(weights='imagenet', 
                         include_top=False,
                         input_tensor=input_t
        )
        
        self.model = keras.models.Sequential()
        self.model.add(self.base_model)
        self.model.add(keras.layers.Flatten())
        self.model.add(keras.layers.BatchNormalization())
        self.model.add(keras.layers.Dense(128, activation='relu', kernel_regularizer='l2'))
        self.model.add(keras.layers.Dropout(0.5))
        self.model.add(keras.layers.BatchNormalization())
        self.model.add(keras.layers.Dense(64, activation='relu', kernel_regularizer='l2'))
        self.model.add(keras.layers.Dropout(0.5))
        self.model.add(keras.layers.BatchNormalization())
        self.model.add(keras.layers.Dense(2, activation='softmax'))
        
    def train_model(self, learn_rate, loss_type, epoch_amount,
                    train_images,train_labels,valid_images,valid_labels):
        LEARNING_RATE = learn_rate
        LOSS = loss_type
        EPOCHS = epoch_amount
        self.model.compile(
            loss=LOSS,
            optimizer=tf.keras.optimizers.RMSprop(learning_rate=LEARNING_RATE),
            metrics=['accuracy']
        )
        self.history = self.model.fit(
            train_images,
            train_labels,
            epochs=EPOCHS,
            validation_data=(valid_images, valid_labels)
        )

    def save_model(self,filepath):
        self.model.save(filepath)
    
    def load_pretrained_model(self,filepath):
        self.model.load(filepath)

    def get_history(self):
        return self.history


