from typing import Generic, TypeVar
from pydantic import BaseModel
from pydantic.generics import GenericModel

T= TypeVar('T')
class ResponseModel(GenericModel, Generic[T]):
    status: str
    message: str
    data: T

    class Config:
        orm_mode = True