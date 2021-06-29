import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def func_log(x, a, b, c, d):
	temp = b*x + c
	return a * np.log(np.where(temp <= 0, 0.0001, temp)) + d


def func_gp(x, a, b, c, d):
	return (a * (1 - np.power(b, x)) / (1 - b)) - c


history = {
	'0.5': [
		0.4200655519962311,
		0.6110789179801941,
		0.6781938672065735,
		0.7107771635055542,
		0.7240163087844849
	],
	'0.1':  []
}

epochs = np.array([
	0.5582422614097595,
	0.6038148999214172,
	0.6296037435531616,
	0.6504266262054443,
	0.6625966429710388,
	0.6721488237380981,
	0.6850582361221313,
	0.6922223567962646,
	0.6994564533233643,
	0.7062109112739563,
	0.7118563055992126,
	0.715543270111084,
	0.7184608578681946,
	0.7238864302635193
])

n = epochs.shape[0]
up_to_epochs = 65
train_x = np.arange(n)
func = func_log

params, other = curve_fit(
	func,
	train_x,
	epochs,
	p0=[0, 0.5, 0.5, 1],
	bounds=(
		[-10000, -1, -10000, -10000],
		[10000, 1, 10000, 10000],
	)
)
pred_x = np.arange(up_to_epochs)
pred = func(pred_x, *params)

print(params)
plt.plot(train_x, epochs, 'bo', label='epochs')
plt.plot(pred_x, pred, '-', label='fit')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
