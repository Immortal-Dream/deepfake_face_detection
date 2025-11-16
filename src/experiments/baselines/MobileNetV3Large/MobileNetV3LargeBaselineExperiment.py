from src.experiments.BaseExperiment import BaseExperiment
from src.utils.model_utils import F1MetricsCallback
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model

class MobileNetV3LargeBaselineExperiment(BaseExperiment):
    """
    Baseline experiment without data augmentation.
    """

    def __init__(self, config: dict):
        super().__init__("MobileNetV3Large_baseline", config)

    def train(self, X_train, y_train, X_val, y_val, f1_callback):
        """
        Train model without data augmentation.

        Returns:
            Training history object
        """
        print("\n" + "="*60)
        print("TRAINING BASELINE MODEL (No Augmentation)")
        print("="*60)

        history = self.model.fit(
            X_train, y_train,
            epochs=self.config['epochs'],
            batch_size=self.config['batch_size'],
            validation_data=(X_val, y_val),
            callbacks=[f1_callback],
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

