import numpy as np
a = np.array([10, 30, 20, 4, 30, 51, 7, 2, 4, 40, 100])
print(a[4])
print(a[3:])
print(a[3:7])
print(a[1::4])

print()
print(np.zeros(5))
print()
print(np.ones((5, 5)))

print()
print(np.linspace(3, 10, 5))
print()

b = np.array([['x', 'y', 'z'], ['a', 'b', 'c']])
print(b)
print(b.ndim)

print()
c = [12, 4, 10, 40, 2]

print(np.sort(c))

print(np.arange(25))

print(np.full((3,5),10))

print(np.diag([0,3,9,10]))