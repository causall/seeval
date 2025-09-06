from typing import TypeVar, Generic, Callable, ParamSpec, Protocol, Type, TypeVarTuple, Mapping, Any, TypedDict, Unpack, runtime_checkable
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as plt3d
from seevals import utils
import numpy as np
import pandas as pd

B = ParamSpec('B')
T = TypeVar('T')
R = TypeVar('R')


class Baz(TypedDict):
    name: str
    age: int


class Forward(Protocol[T, R]):
    def forward(self, x: T) -> R:
        ...


class ForwardImpl():
    def forward(self, x: Baz):
        print(x)
        return 12


def foo(foobar: Forward[T, R], x: T):
    foobar.forward(x)


foo(ForwardImpl(), {"name": "John", "age": 30})


x = utils.calc_hoeffding_error(
    num_samples=10, upper_bound=1, lower_bound=0.0, confidence=0.90)
xx = utils.calc_hoeffding_error(
    num_samples=10, upper_bound=5, lower_bound=1, confidence=0.95)
xxx = utils.calc_hoeffding_error(
    num_samples=10, upper_bound=5, lower_bound=1, confidence=0.99)
print(x, xx/5, xxx/5)

x = utils.calc_serfling_error(
    num_samples=10, population_size=20, upper_bound=2, lower_bound=0, confidence=0.95)
xx = utils.calc_serfling_error(
    num_samples=10, population_size=14, upper_bound=2, lower_bound=0, confidence=0.95)
xxx = utils.calc_serfling_error(
    num_samples=4, population_size=10, upper_bound=2, lower_bound=0, confidence=0.99)


fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
X, Y = np.meshgrid(range(2, 22, 2), range(2, 22, 2))
mask = X > Y
# the following needs to be an np array is that correct

z = np.array([utils.calc_serfling_error(
    num_samples=int(x), population_size=int(y), upper_bound=2, lower_bound=0, confidence=0.95) for x, y in zip(X.ravel(), Y.ravel())])
Z = z.reshape(X.shape)

ax.plot_surface(X, Y, Z)
ax.set_xlabel('num_samples')
ax.set_ylabel('population_size')
ax.set_zlabel('error')
ax.set_title('Serfling Error')
mask = np.nonzero(Z <= 0.5)
# print(list(filter(lambda x: x[0] != x[1], zip(X[mask], Y[mask], Z[mask]))))
# print(f"num_samples: {X[mask]}, population_size: {Y[mask]}")
# plt.show()
filtered_results = list(
    filter(lambda x: x[0] != x[1], zip(X[mask], Y[mask], Z[mask])))
for samples, pop, error in filtered_results:
    print(f"({int(samples)}, {int(pop)}): {error:.4f}")


"""
 Call = Callable[[T],R]

def bar(x: Baz) -> int:
    print(x)
    return 9

def foo(foobar:Callable[[T],R], x: T):
    foobar(x)

foo(bar,{"name": "John", "age": 30})
"""
df = pd.DataFrame(
    {'samples': X.ravel(), 'population': Y.ravel(), 'error': (Z/2.0).ravel()})

filtered_df = df[
    (df['samples'] != df['population']) &
    df['error'].notna()
].sort_values(['samples', 'population']).reset_index(drop=True)
pd.set_option('display.float_format', '{:.6f}'.format)
print(filtered_df[filtered_df['error'] <= 0.25])
