import random

import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt


SPAWN_PROB = [0, 0.005, 0.02, 0.05]

CLEANUP_VIEW_SIZE = 7

thresholdDepletion = 0.4
thresholdRestoration = 0.0
wasteSpawnProbability = 0.5
appleRespawnProbability = 0.05



waste_densities = np.linspace(0,1,100)
apple_spawn_prob, waste_spawn_prob = [], []

for waste_density in waste_densities:

    # waste_density = 0
    # if self.potential_waste_area > 0:
    #     waste_density = 1 - self.compute_permitted_area() / self.potential_waste_area

    # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    # exit()

    # if theres too much waste
    if waste_density >= thresholdDepletion:
        apple_spawn_prob.append(0)
        waste_spawn_prob.append(0)
    # else compute the probabilities based on how much waste
    else:
        waste_spawn_prob.append(wasteSpawnProbability)
        if waste_density <= thresholdRestoration:
            apple_spawn_prob.append(appleRespawnProbability)
        else:
            spawn_prob = (
                1
                - (waste_density - thresholdRestoration)
                / (thresholdDepletion - thresholdRestoration)
            ) * appleRespawnProbability
            apple_spawn_prob.append(spawn_prob)


plt.plot(waste_densities, apple_spawn_prob, label='Apples')
plt.plot(waste_densities, waste_spawn_prob, label='Waste')
plt.xlabel("Waste Density %", fontsize=22)
plt.ylabel("Probability of Spawn (each tile)", fontsize=22)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend()
# plt.show()
plt.savefig("results/spawn_func.png",bbox_inches='tight', dpi=100)