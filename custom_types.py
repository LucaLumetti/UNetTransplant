from typing import List, Literal, Union

Split = Union[Literal["train", "val", "test"], List[Literal["train", "val", "test"]]]
