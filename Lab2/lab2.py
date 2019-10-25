import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sc
import pandas as pd
import seaborn
from scipy.interpolate import interp1d

#NUM1
lambd = 5
size_ex_100 = 100
size_ex_1000 = 1000

al = 0
sigma = 1.75
size_norm_1000 = 1000
size_norm_100 = 100

ex_100 = np.random.exponential(lambd, size_ex_100)
ex_1000 = np.random.exponential(lambd, size_ex_1000)
norm_100 = np.random.normal(al, sigma, size_norm_100)
norm_1000 = np.random.normal(al, sigma, size_norm_1000)

s_ex_100 = sum(ex_100)/size_ex_100
s_ex_1000 = sum(ex_1000)/size_ex_1000
s_norm_100 = sum(norm_100)/size_norm_100
s_norm_1000 = sum(norm_1000)/size_norm_1000

d_ex_100 = np.var(ex_100)
d_ex_1000 = np.var(ex_1000)
d_norm_100 = np.var(norm_100)
d_norm_1000 = np.var(norm_1000)

m_ex_100 = sc.expon.fit(ex_100)
m_ex_1000 = sc.expon.fit(ex_1000)
m_norm_100 = sc.norm.fit(norm_100)
m_norm_1000 = sc.norm.fit(norm_1000)

k_ex_100 = np.quantile(ex_100, q=[0.5, 0.99])
k_ex_1000 = np.quantile(ex_1000, q=[0.5, 0.99])
k_norm_100 = np.quantile(norm_100, q=[0.5, 0.99])
k_norm_1000 = np.quantile(norm_1000, q=[0.5, 0.99])

print('Нормальное распределение 100:','\n','Выборочное среднее:', s_norm_100, '\n',
      'Дисперсия:', d_norm_100, '\n', 'Математическое ожидание:', m_norm_100[0], '\n',
      'Квантиль 0,5:', k_norm_100[0],  '\n', 'Квантиль 0,99:', k_norm_100[1])
print()
print('Нормальное распределение 1000:','\n','Выборочное среднее:', s_norm_1000, '\n',
      'Дисперсия:', d_norm_1000, '\n', 'Математическое ожидание:', m_norm_1000[0], '\n',
      'Квантиль 0,5:', k_norm_1000[0],  '\n', 'Квантиль 0,99:', k_norm_1000[1])
print()
print('Экспоненциальное распределение 100:','\n','Выборочное среднее:', s_ex_100, '\n',
      'Дисперсия:', d_ex_100, '\n', 'Математическое ожидание:', m_ex_100[0], '\n',
      'Квантиль 0,5:', k_ex_100[0],  '\n', 'Квантиль 0,99:', k_ex_100[1])
print()
print('Экспоненциальное распределение 1000:','\n','Выборочное среднее:', s_ex_1000, '\n',
      'Дисперсия:', d_ex_1000, '\n', 'Математическое ожидание:', m_ex_1000[0], '\n',
      'Квантиль 0,5:', k_ex_1000[0],  '\n', 'Квантиль 0,99:', k_ex_1000[1])

fig1, ax1 = plt.subplots()
count, x, ignored = plt.hist(ex_100, 100, density=True)
ax1.set_xlabel("Гистограмма: Экспоненциальное распределение 100")
plt.show()
fig1, ax1 = plt.subplots()
count, x, ignored = plt.hist(ex_1000, 100, density=True)
ax1.set_xlabel("Гистограмма: Экспоненциальное распределение 1000")
plt.show()
fig2, ax2 = plt.subplots()
count, x, ignored = plt.hist(norm_100, 100, density=True)
ax2.set_xlabel("Гистограмма: Нормальное распределение 100")
plt.show()
fig2, ax2 = plt.subplots()
count, x, ignored = plt.hist(norm_1000, 100, density=True)
ax2.set_xlabel("Гистограмма: Нормальное распределение 1000")
plt.show()

series_ex1 = pd.Series(ex_100)
series_ex2 = pd.Series(np.random.exponential(lambd, 500))
series_ex3 = pd.Series(np.random.exponential(lambd, 1000))
fig3, ax3 = plt.subplots()
ax3.set_xlabel("Функция распределения: Экспоненциальное распределение")
ax4 = ax3.twinx()
ax4.set_xlim((ax4.get_xlim()[0], series_ex1.max()))
ax4.set_xlim((ax4.get_xlim()[0], series_ex2.max()))
ax4.set_xlim((ax4.get_xlim()[0], series_ex3.max()))
n, bins, patches = ax4.hist(
    series_ex1, cumulative=1, histtype='step', bins=100, color='tab:orange')
n, bins, patches = ax4.hist(
    series_ex2, cumulative=1, histtype='step', bins=100, color='tab:green')
n, bins, patches = ax4.hist(
    series_ex3, cumulative=1, histtype='step', bins=100, color='tab:blue')
plt.show()

series_norm1 = pd.Series(norm_1000)
series_norm2 = pd.Series(np.random.normal(al, sigma, 500))
series_norm3 = pd.Series(np.random.normal(al, sigma, 100))
fig4, ax5 = plt.subplots()
ax5.set_xlabel("Функция распределения: Нормальное распределение")
ax6 = ax5.twinx()
ax6.set_xlim((ax6.get_xlim()[0], series_norm1.max()))
ax6.set_xlim((ax6.get_xlim()[0], series_norm2.max()))
ax6.set_xlim((ax6.get_xlim()[0], series_norm3.max()))
n, bins, patches = ax6.hist(
    series_norm1, cumulative=1, histtype='step', bins=100, color='tab:orange')
n, bins, patches = ax6.hist(
    series_norm2, cumulative=1, histtype='step', bins=100, color='tab:green')
n, bins, patches = ax6.hist(
    series_norm3, cumulative=1, histtype='step', bins=100, color='tab:blue')
plt.show()

fig5, ax7 = plt.subplots()
ax7.set_xlabel("Функция распределения теоретическая: Экспоненциальное распределение 100")
cdf_ex = sc.expon.cdf(ex_100, lambd)
x_new = np.linspace(ex_100.min(), ex_100.max(), 500)
f = interp1d(ex_100, cdf_ex, kind='quadratic')
y_smooth = f(x_new)
plt.plot(x_new, y_smooth)
plt.show()

fig5, ax7 = plt.subplots()
ax7.set_xlabel("Функция распределения теоретическая: Экспоненциальное распределение 1000")
cdf_ex = sc.expon.cdf(ex_1000, lambd)
x_new = np.linspace(ex_1000.min(), ex_1000.max(), 500)
f = interp1d(ex_1000, cdf_ex, kind='quadratic')
y_smooth = f(x_new)
plt.plot(x_new, y_smooth)
plt.show()

fig6, ax7 = plt.subplots()
ax7.set_xlabel("Функция распределения теоретическая: Нормальное распределение 100")
cdf_norm = sc.norm.cdf(norm_100, loc=al, scale=sigma)
x_new = np.linspace(norm_100.min(), norm_100.max(), 500)
f = interp1d(norm_100, cdf_norm, kind='quadratic')
y_smooth = f(x_new)
plt.plot(x_new, y_smooth)
plt.show()

fig6, ax7 = plt.subplots()
ax7.set_xlabel("Функция распределения теоретическая: Нормальное распределение 1000")
cdf_norm = sc.norm.cdf(norm_1000, loc=al, scale=sigma)
x_new = np.linspace(norm_1000.min(), norm_1000.max(), 500)
f = interp1d(norm_1000, cdf_norm, kind='quadratic')
y_smooth = f(x_new)
plt.plot(x_new, y_smooth)
plt.show()

fig8, ax10 = plt.subplots()
seaborn.distplot(ex_100, hist = False, label='100')
seaborn.distplot(np.random.exponential(lambd, 500), hist = False, label='500')
seaborn.distplot(np.random.exponential(lambd, 1000), hist = False, label='1000')
ax10.set_xlabel("Плотность распределения: Экспоненциальное распределение")
plt.show()

fig7, ax9 = plt.subplots()
seaborn.distplot(norm_1000, hist = False, label='1000')
seaborn.distplot(np.random.normal(al, sigma, 500), hist = False, label='500')
seaborn.distplot(np.random.normal(al, sigma, 100), hist = False, label='100')
ax9.set_xlabel("Плотность распределения: Нормальное распределение")
plt.show()

fig9, ax11 = plt.subplots()
pdf_ex = sc.expon.pdf(ex_100, lambd)
x_new = np.linspace(ex_100.min(), ex_100.max(), 500)
f = interp1d(ex_100, pdf_ex, kind='quadratic')
y_smooth = f(x_new)
plt.plot(x_new, y_smooth)
ax11.set_xlabel("Функция распределения теоретическая: Экспоненциальное распределение 100")
plt.show()

fig9, ax11 = plt.subplots()
pdf_ex = sc.expon.pdf(ex_1000, lambd)
x_new = np.linspace(ex_1000.min(), ex_1000.max(), 500)
f = interp1d(ex_1000, pdf_ex, kind='quadratic')
y_smooth = f(x_new)
plt.plot(x_new, y_smooth)
ax11.set_xlabel("Функция распределения теоретическая: Экспоненциальное распределение 1000")
plt.show()

fig10, ax12 = plt.subplots()
pdf_norm = sc.norm.pdf(norm_100, loc=al, scale=sigma)
x_new = np.linspace(norm_100.min(), norm_100.max(), 500)
f = interp1d(norm_100, pdf_norm, kind='quadratic')
y_smooth = f(x_new)
plt.plot(x_new, y_smooth)
ax12.set_xlabel("Функция распределения теоретическая: Нормальное распределение 100")
plt.show()

fig10, ax12 = plt.subplots()
pdf_norm = sc.norm.pdf(norm_1000, loc=al, scale=sigma)
x_new = np.linspace(norm_1000.min(), norm_1000.max(), 500)
f = interp1d(norm_1000, pdf_norm, kind='quadratic')
y_smooth = f(x_new)
plt.plot(x_new, y_smooth)
ax12.set_xlabel("Функция распределения теоретическая: Нормальное распределение 1000")
plt.show()


#NUM2
def distance(x1,y1,x2,y2):
    dis = 0
    dis = ((x2-x1)**2 +(y2-y1)**2)**0.5
    return dis

a = 10
b = 30
arr = []
arr_100 = []
arr_1000 = []
arr_10000 = []
fig11, ax13 = plt.subplots()
A = np.random.uniform(-5,5,(1,2))
B = A + (0, b)
C = A + (a, 0)
D = A + (a, b)
plt.plot([A[0][0],B[0][0],D[0][0],C[0][0],A[0][0]], [A[0][1],B[0][1],D[0][1],C[0][1],A[0][1]], color = 'black', linewidth = 3)
x_min, x_max, y_min, y_max = A[0][0], D[0][0], A[0][1], D[0][1]

X_1 = np.random.uniform(x_min,x_max,(1,100))
Y_1 = np.random.uniform(y_min,y_max,(1,100))
ax13.set_xlabel("Выборка 100")
for i in range(len(X_1)):
    plt.scatter(X_1[i], Y_1[i])
plt.show()

fig12, ax14 = plt.subplots()
plt.plot([A[0][0],B[0][0],D[0][0],C[0][0],A[0][0]], [A[0][1],B[0][1],D[0][1],C[0][1],A[0][1]], color = 'black', linewidth = 3)
X_2 = np.random.uniform(x_min,x_max,(1,1000))
Y_2 = np.random.uniform(y_min,y_max,(1,1000))
ax14.set_xlabel("Выборка 1000")
for i in range(len(X_2)):
    plt.scatter(X_2[i], Y_2[i])
plt.show()
fig13, ax14 = plt.subplots()
X_3 = np.random.uniform(x_min,x_max,(1,10000))
Y_3 = np.random.uniform(y_min,y_max,(1,10000))
plt.plot([A[0][0],B[0][0],D[0][0],C[0][0],A[0][0]], [A[0][1],B[0][1],D[0][1],C[0][1],A[0][1]], color = 'black', linewidth = 3)
ax14.set_xlabel("Выборка 10000")
for i in range(len(X_3)):
    plt.scatter(X_3[i], Y_3[i])
plt.show()

for i in range(1, len(X_1[0]), 2):
    d = distance(X_1[0][i], Y_1[0][i], X_1[0][i-1], Y_1[0][i-1])
    arr_100.append(d)
for i in range(1,len(X_2[0]),2):
    d = distance(X_2[0][i], Y_2[0][i], X_2[0][i-1], Y_2[0][i-1])
    arr_100.append(d)
for i in range(1,len(X_3[0]),2):
    d = distance(X_3[0][i], Y_3[0][i], X_3[0][i-1], Y_3[0][i-1])
    arr_10000.append(d)

arr.extend(arr_100)
arr.extend(arr_1000)
arr.extend(arr_10000)
sred = sum(arr) / float(len(arr))
print(len(arr))

fig4, ax5 = plt.subplots()
seaborn.distplot(arr, hist = False, label='100')
ax5.set_xlabel("Функция распределения и Плотность распределения расстояний между точками")
series_norm1 = pd.Series(arr)
ax6 = ax5.twinx()
ax6.set_xlim((ax6.get_xlim()[0], series_norm1.max()))
n, bins, patches = ax6.hist(
    arr, cumulative=1, histtype='step', bins=100, color='tab:orange')
plt.show()

size_norm_10000 = 10000
norm_100 = np.random.normal(al, sigma, size_norm_100)
norm_1000 = np.random.normal(al, sigma, size_norm_1000)
norm_10000 = np.random.normal(al, sigma, size_norm_10000)


fig5, ax7 = plt.subplots()
seaborn.distplot(norm_100, hist = False, label='100')
seaborn.distplot(norm_1000, hist = False, label='1000')
seaborn.distplot(norm_10000, hist = False, label='10000')
ax7.set_xlabel("Плотность распределения")

fig6, ax8 = plt.subplots()
ax8.set_xlabel("Функция распределения")
series_norm2 = pd.Series(norm_100)
series_norm3 = pd.Series(norm_1000)
series_norm4 = pd.Series(norm_10000)
ax9 = ax8.twinx()
ax9.set_xlim((ax9.get_xlim()[0], series_norm2.max()))
ax9.set_xlim((ax9.get_xlim()[0], series_norm3.max()))
ax9.set_xlim((ax9.get_xlim()[0], series_norm4.max()))
n, bins, patches = ax9.hist(
    series_norm2, cumulative=1, histtype='step', bins=100, color='tab:orange')
n, bins, patches = ax9.hist(
    series_norm3, cumulative=1, histtype='step', bins=100, color='tab:red')
n, bins, patches = ax9.hist(
    series_norm4, cumulative=1, histtype='step', bins=100, color='tab:green')
plt.show()

#----------------------------NOTE-------------------------------
fig5, ax7 = plt.subplots()
cdf_ex = sc.expon.cdf(ex, lambd)
ax7.set_xlabel("Функция распределения теоретическая: Экспоненциальное распределение")
for i in range(len(cdf_ex)):
    plt.scatter(ex[i], cdf_ex[i])
plt.show()

fig6, ax8 = plt.subplots()
cdf_norm = sc.norm.cdf(norm, loc=al, scale=sigma)
ax8.set_xlabel("Функция распределения теоретическая: Нормальное распределение")
for i in range(len(cdf_norm)):
    plt.scatter(norm[i], cdf_norm[i])
plt.show()


fig9, ax11 = plt.subplots()
pdf_ex = sc.expon.pdf(ex, lambd)
ax11.set_xlabel("Плотность распределения теоретическая: Экспоненциальное распределение")
for i in range(len(pdf_ex)):
     plt.scatter(ex[i], pdf_ex[i])
plt.show()
fig10, ax12 = plt.subplots()
pdf_norm = sc.norm.pdf(norm, loc=al, scale=sigma)
ax12.set_xlabel("Плотность распределения теоретическая: Нормальное распределение")
for i in range(len(pdf_norm)):
    plt.scatter(norm[i], pdf_norm[i])
plt.show()