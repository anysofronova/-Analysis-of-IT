import numpy as np
import matplotlib.pyplot as plt
import math
import seaborn

al = 0.3
sigma = 1.75
size = 1000
R = 10
side = 20

#Todo-------------------------1-------------------------

fig1, ax1 = plt.subplots()
plt.xlabel("x")
plt.ylabel("y")
plt.grid()  #сетка
A = np.random.uniform(0,5,(1,2))
B = A + (0, side)
C = A + (side, 0)
D = A + (side, side)
plt.plot([A[0][0],B[0][0],D[0][0],C[0][0],A[0][0]], [A[0][1],B[0][1],D[0][1],C[0][1],A[0][1]], color = 'black', linewidth = 3)
x_min, x_max, y_min, y_max = A[0][0], D[0][0], A[0][1], D[0][1]
O_1 = (x_max-x_min)/2 + x_min
O_2 = (y_max-y_min)/2 + y_min
circle1 = plt.Circle((O_1, O_2), R, fill=False, linewidth = 3, clip_on = False)
ax1.add_artist(circle1)
plt.axis("equal")
plt.xlim(-5,35)
plt.ylim(-5,35)
X = np.random.uniform(x_min, x_max, size = size)
Y = np.random.uniform(y_min, y_max, size = size)
ax1.set_xlabel("Выборка через квадрат")
for i in range(len(X)):
    #plt.scatter(X[i], Y[i])
    if ((O_1-X[i])**2 + (O_2-Y[i])**2) < R**2:
        plt.scatter(X[i], Y[i])
plt.show()

#Todo-------------------------2-------------------------

fig2, ax2 = plt.subplots()
plt.xlabel("x")
plt.ylabel("y")
plt.grid()  #сетка
O = np.random.uniform(0,5,(1,2))
circle2 = plt.Circle((O[0][0], O[0][1]), R, fill=False, linewidth = 3, clip_on = False)
ax2.add_artist(circle2)
ax2.set_xlabel("Выборка через угол и расстояние")
plt.axis("equal")
x = []
y = []
for i in range(size):
    angle = np.random.uniform(0, 360)
    r = math.sqrt(np.random.uniform(0,1)) * R
    t_x = O[0][0] + (r * math.cos(angle))
    t_y = O[0][1] + (r * math.sin(angle))
    x.append(t_x)
    y.append(t_y)
    plt.scatter(t_x, t_y)
plt.show()

def v_s(arr):
    return sum (arr) / size

def disp(arr):
    return np.var(arr)

def distance(x1,y1,x2,y2):
    dis = 0
    dis = ((x2-x1)**2 +(y2-y1)**2)**0.5
    return dis

print('Выборочное среднее для выборки 1:', v_s(X), ",", v_s(Y), '\n',
      'Выборочное среднее для выборки 2:', v_s(x), ",", v_s(y), '\n',
      'Дисперсия для выборки 1:', disp(X), ",", disp(Y),'\n',
      'Дисперсия для выборки 2:', disp(x), ",", disp(y),'\n')

DIS_1 = []
dis_1 = []
for i in range(len(X)):
    d = distance(X[i], Y[i], 20, 0)
    DIS_1.append(d)
for i in range(len(x)):
    d = distance(x[i], y[i], 20, 0)
    dis_1.append(d)
fig3, ax3 = plt.subplots()
seaborn.distplot(DIS_1, hist = False, label='1')
seaborn.distplot(dis_1, hist = False, label='2')
ax3.set_xlabel("Плотность распределения между точкой и фиксированной точкой")
plt.show()

DIS_2 = []
dis_2 = []
for i in range(1, len(X), 2):
    d = distance(X[i], Y[i], X[i - 1], Y[i - 1])
    DIS_2.append(d)
for i in range(1, len(x), 2):
    d = distance(x[i], y[i], x[i - 1], y[i - 1])
    dis_2.append(d)
fig4, ax4 = plt.subplots()
seaborn.distplot(DIS_2, hist = False, label='1')
seaborn.distplot(dis_2, hist = False, label='2')
ax4.set_xlabel("Плотность распределения между 2-мя точками выборки")
plt.show()