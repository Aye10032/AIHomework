import numpy as np

np.random.seed(0)
arr11 = 5 - np.arange(1, 13).reshape(4, 3)

print(arr11)

# 计算所有元素的和
print('-------------')
print(arr11.sum())

# 计算每一列的和
print(np.sum(arr11, axis=0))

# 对每一个元素求累积和
print('-------------')
print(arr11.cumsum())

# 对每一列求累积和
print(np.cumsum(arr11, axis=0))

# 计算每一行的累计积
print('-------------')
print(np.cumprod(arr11, axis=1))

# 计算所有元素的最小值
print('-------------')
print(arr11.min())

# 计算每一列的最大值
print(np.max(arr11, axis=0))

# 计算所有元素的均值
print('-------------')
print(arr11.mean())

# 计算每一行的均值
print(np.mean(arr11, axis=1))

# 计算所有元素的中位数
print('-------------')
print(np.median(arr11))

# 计算每一列中的中位数
print(np.median(arr11, axis=0))

# 计算所有元素的方差
print('-------------')
print(arr11.var())

# 计算每一行的标准差
print(np.std(arr11, axis=1))
