from src.experiments.BaseExperiment import BaseExperiment
from src.utils.model_utils import F1MetricsCallback
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model


class MobileNetV3LargeAugumentedExperiment(BaseExperiment):
    """
    Augmented experiment with data augmentation and early stopping.
    """

    def __init__(self, config: dict):
        super().__init__("MobileNetV3Large_augumented", config)

    def train(self, X_train, y_train, X_val, y_val, f1_callback):
        """
        Train model with data augmentation and early stopping.

        Returns:
            Training history object
        """
        print("\n" + "=" * 60)
        print("TRAINING MODEL WITH DATA AUGMENTATION")
        print("=" * 60)

        # Data augmentation
        datagen = ImageDataGenerator(
            rotation_range=15,
            horizontal_flip=True,
            vertical_flip=True
        )
        train_generator = datagen.flow(X_train, y_train,
                                       batch_size=self.config['batch_size'])

        # Early stopping
        early_stop = EarlyStopping(
            monitor='val_accuracy',
            patience=6,
            restore_best_weights=True,
            verbose=1
        )

        history = self.model.fit(
            train_generator,
            epochs=self.config['epochs'],
            validation_data=(X_val, y_val),
            callbacks=[f1_callback, early_stop],
            verbose=1
        )

        return history

    def create_model(self, num_classes: int):
        input_shape = (self.config['image_size'], self.config['image_size'], 3)

        print("Creating MobileNetV3Large model...")
        base_model = MobileNetV3Large(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )

        x = Flatten()(base_model.output)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.45)(x)
        output = Dense(num_classes, activation='softmax')(x)

        model = Model(inputs=base_model.input, outputs=output)
        model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )

        self.model = model
