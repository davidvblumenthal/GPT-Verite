from accelerate.tracking import WandBTracker
from typing import Optional

import wandb

class WandBCustomTracker(WandBTracker):

    name = "wandb_resume_run"

    def store_init_configuration(self, values: dict):

        wandb.config.update(values, allow_val_change=True)
        logger.debug("Stored initial configuration hyperparameters to WandB")