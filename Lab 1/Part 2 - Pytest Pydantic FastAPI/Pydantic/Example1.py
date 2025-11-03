from datetime import datetime 
from typing import List, Optional 
from pydantic import BaseModel 

class User(BaseModel):
    id: int
    name: str = 'John Doe'
    signup_ts: Optional[datetime] = None
    friends: List[int] = []

external_data = {
    'id': 'imane',
    'signup_ts': '2019-06-01 12:22',
    'friends': [1, 2, '5'],
}

user = User(**external_data) 

print(user.id)
#> 123
print(repr(user.signup_ts))
#> datetime.datetime(2019, 6, 1, 12, 22)
print(user.friends)
#> [1, 2, 5]
print(user.dict())