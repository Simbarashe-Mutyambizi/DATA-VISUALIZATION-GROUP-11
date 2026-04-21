import matplotlib.pyplot as plt
import numpy as np

vals = [0.629]
names = ["Trips/Hour"]
x = np.arange(len(vals))            # [0]
bar_width = 0.2                     # make smaller

fig, ax = plt.subplots()
ax.bar(x, vals, width=bar_width, color="C0")
ax.set_xticks(x)
ax.set_xticklabels(names)

# restrict x-limits so bar is centered and narrow on the axis
ax.set_xlim(-0.5, 0.5)              # adjust to taste; narrower range -> bar appears narrower
ax.set_ylim(0, 0.7)
ax.set_ylabel("R²")
ax.set_title("R²(test) Bike")
plt.show()