"""
    Defines classes for the tasks to run item gen on.
    Conceptually, a task is a creativity datasets that the LLM both solves and generates items for.
    Tasks must consist of:
    1. A prompt template for each component of the pipeline.
    2. A series of methods for scoring the responses that can be passed to the constraint optimizer.
"""

from abc import ABC, abstractmethod
from typing import List

from config import config
import imp


class AbstractTask(ABC):
    @abstractmethod
    def RunScorers(self) -> List:
        pass


class CPS(AbstractTask):
    # add custom modules under the class def
    roberta_scorer = imp.load_source(
        "RLPS_RoBERTa",
        "/home/aml7990/Code/creativity-item-generation/scorers/RLPS_RoBERTa.py",
    )

    def __init__(self) -> None:
        super().__init__()
        self.task_id = "CPS"

    # run scoring for the task
    # to ensure correct behavior, round must always be passed
    def RunScorers(self, i: int) -> None:
        self.roberta_scorer.predict_with_model(
            config["itemResponseOriginalityModelDir"],
            config["itemResponseGenOutputFile"],
            "originality",
            config[
                "itemResponseGenOutputFile"
            ],  # we need to chop off most columns from the first instance, so send another copy to save to
            i,  # round
        )


class Consequences(AbstractTask):
    def __init__(self) -> None:
        super().__init__()
        self.task_id = "consequences"

    # run scoring for the task
    # to ensure correct behavior, round must always be passed
    def RunScorers(self, i: int) -> None:
        pass
