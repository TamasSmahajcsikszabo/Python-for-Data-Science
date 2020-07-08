import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
fig, ax_lst = plt.subplots(2,2) # faceting

plt.show()


# plotting is on np.array objects
import pandas
a = pandas.DataFrame(np.random.rand(4,5), columns = list('abcde')) # the imput

# converting it into an np.array
a_asndarray = a.values

x = np.linspace(0,2,200)

plt.plot(x,x, label = 'linear')
plt.plot(x,x ** 2, label = 'quadratic')
plt.plot(x,x ** 3, label = 'cubic')
plt.title('simple plot')
plt.xlabel('x values')
plt.ylabel('f(x)')
plt.legend()
plt.show()

### simple plot
plt.plot([1,4,5,7], [9,8,2,3])
plt.show()

plt.plot([1,4,5,7], [9,8,2,3], 'bo') # concatenated format + type argument
plt.axis([0,10,0,20]) # setting the axis
plt.show()

t = np.arange(0., 5., 0.2)

plt.plot(t , t, 'r-', t, t**2, 'bs',t, t**3, 'g^')
plt.show()

## accessing variables by name

data = {'a' : np.arange(50),
        'c' : np.random.randint(0, 50, 50),
        'd' : np.random.randn(50)}

data['b'] = data['a'] + 10 * np.random.randn(50)
data['d'] = np.abs(data['d']) * 100

plt.scatter('a', 'b', c = 'c', s = 'd', alpha = 1/2, data = data)
plt.show()

## plotting with categorical data
names = ['group_a', 'group_b', 'group_c']
values = [1, 10, 100]

plt.figure(1, figsize = (9,3))
plt.subplot(131)
plt.bar(names,values)
plt.subplot(132)
plt.scatter(names,values)
plt.subplot(133)
plt.plot(names,values)
plt.suptitle('Categorical Plotting')
plt.show()

## LINE PROPERTIES
setp() # for setting parameters

### MULTIPLE FIGURES AND AXES
def f(t):
    return np.exp(-t) * np.cos(2*np.pi*t)

t1 = np.arange(0.0, 5.0, 0.1)
t2 = np.arange(0.0, 5.0, 0.02)


plt.figure(1)
plt.subplot(211)
plt.plot(t1, f(t1), 'bo', t2, f(t2), 'k')

plt.subplot(212) # number sets: n_col, n_row, n_plot; subplot(1,2,2) is a valid
plt.plot(t2, np.cos(2*np.pi*t2), 'r--')
plt.show()

# figure clearing: clf()
# current axis clearing cla()
# release memory after plot creation with close()

### adding text to the plot
mu, sigma = 100, 15
x = mu + sigma * np.random.randn(10000)

n, bins, patches = plt.hist(x, 50, density = 1, facecolor = 'g', alpha = 0.75)
plt.xlabel('Smarts')
plt.ylabel('Probability')
plt.title('Hisogram of IQ')
plt.text(60, 0.025, r'$\mu = 100,\ \sigma = 15$', color = 'blue') # r indicates raw text information
plt.axis([40, 160, 0, 0.03])
plt.grid(True)
plt.show()

### annotating with annotate()
ax = plt.subplot(111)

t = np.arange(0.0, 5.0, 0.01)
s = np.cos(2*np.pi*t)
line, = plt.plot(t, s, lw=2)

plt.annotate('local max', xy=(2, 1), xytext=(3, 1.5),
             arrowprops=dict(facecolor='black', shrink=0.05),
             )

plt.ylim(-2, 2)
plt.show()

## logarithmis and other nonlinear axes
plt.xscale('log')