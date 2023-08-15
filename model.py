from pydantic import BaseModel
from typing import List


class Questionnaire(BaseModel):
    answers: List[int]
    estimated_time: List[int]
    gender: int
    engnat: int
    age: int
