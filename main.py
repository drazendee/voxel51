import os
import time

# # Set environment variables before importing FiftyOne
# os.environ["FIFTYONE_DATABASE_DIR"] = "/tmp/db/"
# os.environ["FIFTYONE_DEFAULT_DATASET_DIR"] = "/tmp/db/"
# os.environ["FIFTYONE_DATASET_ZOO_DIR"] = "/tmp/db_zoo/"
# os.environ["FIFTYONE_MODEL_ZOO_DIR"] = "/tmp/zoo_model/"

import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone import ViewField as F

# Ensure FiftyOne reads the configurations from the environment variables
fo.config.database_dir = os.environ["FIFTYONE_DATABASE_DIR"]
fo.config.default_dataset_dir = os.environ["FIFTYONE_DEFAULT_DATASET_DIR"]
fo.config.dataset_zoo_dir = os.environ["FIFTYONE_DATASET_ZOO_DIR"]
fo.config.model_zoo_dir = os.environ["FIFTYONE_MODEL_ZOO_DIR"]

# Load dataset
dataset = foz.load_zoo_dataset("quickstart")

# Launch the FiftyOne app
session = fo.launch_app(dataset, port=5151, remote=True)

# Show the session
session.show()

# Update the session view
session.view = (
    dataset
    .sort_by("uniqueness", reverse=True)
    .limit(25)
    .filter_labels("predictions", F("confidence") > 0.5)
)
while True:
    time.sleep(10)