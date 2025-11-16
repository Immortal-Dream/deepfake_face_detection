from src.experiments.BaseExperiment import BaseExperiment
from src.utils.model_utils import F1MetricsCallback

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