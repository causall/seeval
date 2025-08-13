from typing import TypeVar,Generic, Callable, ParamSpec, Protocol, Type, TypeVarTuple, Mapping,Any, TypedDict, Unpack, runtime_checkable

B = ParamSpec('B')
T = TypeVar('T')
R = TypeVar('R')

class Baz(TypedDict):
    name: str
    age: int

class Forward(Protocol[T,R]):
    def forward(self, x: T) -> R:
        ...

class ForwardImpl():
    def forward(self, x: Baz):
        print(x)
        return 12

def foo(foobar:Forward[T,R], x: T):
    foobar.forward(x)

foo(ForwardImpl(), {"name": "John", "age": 30})
"""
 Call = Callable[[T],R]

def bar(x: Baz) -> int:
    print(x)
    return 9

def foo(foobar:Callable[[T],R], x: T):
    foobar(x)

foo(bar,{"name": "John", "age": 30})
"""
