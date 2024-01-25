import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone import ViewField as F


dataset = foz.load_zoo_dataset("quickstart")

session = fo.launch_app(dataset, address='127.0.0.1')

session.show()


session.view = (
    dataset
    .sort_by("uniqueness", reverse=True)
    .limit(25)
    .filter_labels("predictions", F("confidence") > 0.5)
)