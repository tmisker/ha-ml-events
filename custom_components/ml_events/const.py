"""Constants for the ML Events integration."""

DOMAIN = "ml_events"
CONF_DETECTOR_NAME = "detector_name"
CONF_INCLUDED_SENSORS = "included_sensors"
CONF_AUTO_SELECT_SENSORS = "auto_select_sensors"
CONF_TRAINING_WINDOW = "training_window_minutes"
CONF_MIN_EXAMPLES = "min_examples"
CONF_MODEL_TYPE = "model_type"

# Service constants
SERVICE_LABEL_EVENT = "label_event"
SERVICE_TRAIN_MODEL = "train_model"
SERVICE_CLEAR_LABELS = "clear_labels"

# Default values
DEFAULT_TRAINING_WINDOW = 30  # minutes
DEFAULT_MIN_EXAMPLES = 5
DEFAULT_MODEL_TYPE = "random_forest"
DEFAULT_POLLING_INTERVAL = 60  # seconds
DEFAULT_AUTO_SELECT_SENSORS = True

# State attributes
ATTR_CONFIDENCE = "confidence"
ATTR_LAST_TRAINED = "last_trained"
ATTR_EXAMPLES_COUNT = "examples_count"
ATTR_TOP_FEATURES = "top_features"
ATTR_MODEL_ACCURACY = "model_accuracy"

# Internal storage
DATA_MODELS = "models"
DATA_TRAINED_DETECTORS = "trained_detectors"
DATA_DETECTOR_CONFIG = "detector_config"
DATA_LABELS = "event_labels"
DATA_ML_SERVICE = "ml_service"