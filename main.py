import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone import ViewField as F

fo.config.database_dir = "/tmp/zoo/db"
fo.config.default_dataset_dir = "/tmp/zoo/db"
fo.config.dataset_zoo_dir = "/tmp/zoo/"
fo.config.model_zoo_dir = "/tmp/zoo_model"

# Load dataset
dataset = foz.load_zoo_dataset("quickstart")


# Launch the FiftyOne app
session = fo.launch_app(dataset, address='0.0.0.0', port=5151, remote=True)

# Show the session
session.show()

# Update the session view
session.view = (
    dataset
    .sort_by("uniqueness", reverse=True)
    .limit(25)
    .filter_labels("predictions", F("confidence") > 0.5)
)