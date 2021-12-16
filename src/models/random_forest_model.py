from sklearn.ensemble import RandomForestClassifier

from src.models.base_model import BaseModel


class RandomForestModel(BaseModel):
    def __init__(self, max_depth: int = 4, random_state: int = 0):
        self.max_depth = max_depth
        self.random_state = random_state

        super().__init__(
            model=RandomForestClassifier(
                max_depth=self.max_depth,
                random_state=self.random_state
            )
        )
