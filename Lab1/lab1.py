import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

#NUM4
a = np.array([[2,1,0,1],[1,-3,2,4],[-5,0,-1,-7],[1,-6,2,6]])
b = np.array([8,9,-5,0])
ans1 = np.linalg.solve(a,b)
print("Ответ на номер 4:", ans1)

#NUM5
c = np.array([7,45,12,23])
ans2 = np.roots(c)
print("Ответ на номер 5:", ans2)

x = np.linspace(-8,5,100)
y = 7*x**3 + 45*x**2 + 12*x + 23
fig, ax = plt.subplots()
ax.plot(x, y, color="blue", label="y(x)")
ax.set_xlabel("Задание номер 5")
ax.legend()
plt.show()

#NUM6
d = np.random.uniform(0,10,(10, 2))
fig1, ax1 = plt.subplots()
for i in d:
    plt.scatter(i[0],i[1])
ax1.set_xlabel("Задание номер 6")
plt.show()

#NUM7
al = 0.3
sigma = 1.75
size = 1000
e = np.random.normal(al, sigma, size)
fig2, ax2 = plt.subplots()
count, x, ignored = plt.hist(e, 32, density=True)
ax2.set_xlabel("Задание номер 7")
plt.show()

print("Ответ на номер 7: Математическое ожидание = ", sum(e)/size)
print("Ответ на номер 7: Дисперсия = ", np.var(e))

#NUM8
fig3, ax3 = plt.subplots()
side = np.random.uniform(1,3)
A = np.random.uniform(-3,3,(1,2))
B = A + (0, side)
C = A + (side, 0)
D = A + (side, side)
plt.plot([A[0][0],B[0][0],D[0][0],C[0][0],A[0][0]], [A[0][1],B[0][1],D[0][1],C[0][1],A[0][1]], color = 'black', linewidth = 3)
x_min, x_max, y_min, y_max = A[0][0], D[0][0], A[0][1], D[0][1]
X = stats.truncnorm.rvs((x_min - al) / sigma, (x_max - al) / sigma, loc=al, scale=sigma, size = 10)
Y = stats.truncnorm.rvs((y_min - al) / sigma, (y_max - al) / sigma, loc=al, scale=sigma, size = 10)
ax3.set_xlabel("Задание номер 8")
for i in range(len(X)):
    plt.scatter(X[i], Y[i])
plt.show()
