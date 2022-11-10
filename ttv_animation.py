import rebound
from transit_times import transits
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import os
import imageio

# Trappist-1 simulation
sim = rebound.Simulation()
sim.units = ("day", "AU", "Msun")
# trappist1
sim.add(m=0.0898,  x=-6.270115450709344e-6, y=8.835624225733689e-22, z=-2.580856314896234e-8, vx=8.768152952693422e-9, vy=-7.344243028008174e-6, vz=3.1565958683313974e-11)
# trappist1b
sim.add(m=4.126e-6, x=0.01154062487574975, y=4.5545326868616914e-18, z=5.535729797317407e-5, vx=-0.0002969391071305844, vy=0.047955756680831484, vz=-1.4242384066266517e-6)
# trappist1c
sim.add(m=3.928e-6, x=0.015700140283041567, y=-7.805162864628216e-18, z=6.117228628527248e-5, vx=-5.783068796913491e-5, vy=0.041265366429277785, vz=-2.2534327619555574e-7)
# trappist1d
sim.add(m=1.17e-6, x=0.02223189424784782, y=-4.7697382095340744e-18, z=3.992427906922342e-5, vx=-0.0002858774854722535, vy=0.03459300535789464, vz=-5.135603651742169e-7)
# trappist1e
sim.add(m=2.08e-6, x=0.029401499049070842, y=1.040903005716132e-17, z=0.00010573491687572098, vx=4.842812901966987e-5, vy=0.02999034454017728, vz=1.7415373439952337e-7)
# trappist1f
sim.add(m=3.12e-6, x=0.03853976420772517, y=3.470377206974679e-18, z=0.00017328927824027067, vx=0.00026293850693193815, vy=0.026244869459408154, vz=1.1822640329190357e-6)
# trappist1g
sim.add(m=3.967e-6, x=0.04681477443782413, y=-1.9080907458050414e-17, z=0.00021049667920635862, vx=4.858590539154705e-5, vy=0.023832760708238972, vz=2.1845819197543384e-7)
# trappist1h
sim.add(m=9.79e-7, x=0.061945750141904586, y=-1.9081750249014255e-17, z=0.0002110082846530544, vx=-0.0001168669207994127, vy=0.02070983162471822, vz=-3.980884634390229e-7)
sim.integrator = "whfast"
sim.dt = 0.001
sim.move_to_com()
sim.integrate(1000)
sim.t = 0
sim.save("checkpoint.bin")

# Calculate the transit times
sim = rebound.Simulation("checkpoint.bin")
trans = transits(sim, method="newtons", accuracy=1e-7, tmax=1000.0)
# Calculate the ttvs
ttvs = np.empty(len(trans), dtype=object)
for i in range(len(trans)):
    nonzero_transits = trans[i][np.nonzero(trans[i])]
    # Linear least squares fit
    N = len(nonzero_transits)
    A = np.vstack([np.ones(N), range(N)]).T
    c, m = np.linalg.lstsq(A, nonzero_transits, rcond=-1)[0]
    # Subtract the fit line from the nonzero transits to get the ttvs
    ttvs[i] = nonzero_transits - m * np.array(range(N)) - c

# Generating each frame of the animation
sim = rebound.Simulation("checkpoint.bin")
for i in range(1000):
    sim.integrate(sim.t+1)
    # Orbital plot
    fig, ax = rebound.OrbitPlot(sim, color=True,unitlabel="[AU]",xlim=[-0.08,0.08],ylim=[-0.08,0.08])
    fig.set_figheight(5)
    fig.set_figwidth(5)
    ax.set_title("Trappist-1", fontsize=14)
    ax.hlines(y=0, xmin=0, xmax=0.08, color="black", label="line of sight")
    ax.annotate(f"t = {i} days", xy=(-0.075, 0.067))
    ax.legend()
    plt.savefig(f"Trappist1Simulation/orbitplot{i}.png",bbox_inches='tight')
    fig.clear()
    plt.close()
    # TTV plot
    fig2, axs2 = plt.subplots(2,1, figsize=(5,5))
    nonzero_d = trans[2][np.nonzero(trans[2])]
    nonzero_e = trans[3][np.nonzero(trans[3])]
    nonzero_f = trans[4][np.nonzero(trans[4])]
    nonzero_g = trans[5][np.nonzero(trans[5])]
    d_trans = nonzero_d[np.where(nonzero_d < sim.t)[0]]
    e_trans = nonzero_e[np.where(nonzero_e < sim.t)[0]]
    f_trans = nonzero_f[np.where(nonzero_f < sim.t)[0]]
    g_trans = nonzero_g[np.where(nonzero_g < sim.t)[0]]
    axs2[0].scatter(d_trans, ttvs[2][:len(d_trans)], color="violet")
    axs2[0].scatter(e_trans, ttvs[3][:len(e_trans)], color="goldenrod")
    axs2[0].legend(["Trappist-1d", "Trappist-1e"], loc="upper right")
    axs2[0].set_title("Transits", fontsize=14)
    axs2[0].set_xlim(0,1000)
    axs2[1].set_xlim(0,1000)
    axs2[1].scatter(f_trans, ttvs[4][:len(f_trans)], color="darkslategrey")
    axs2[1].scatter(g_trans, ttvs[5][:len(g_trans)], color="royalblue")
    axs2[1].legend(["Trappist-1f", "Trappist-1g"], loc="upper right")
    axs2[1].set_xlabel("Time (days)", fontsize=11)
    axs2[0].set_ylim(-0.07,0.07)
    axs2[1].set_ylim(-0.06,0.06)
    axs2[1].set_ylabel("TTV (days)", fontsize=11)
    axs2[1].yaxis.set_label_coords(-.15, 1.1)
    plt.savefig(f"ttvplots/ttvplot{i}.png",bbox_inches='tight')
    fig2.clear()
    plt.close()

# Merge each frame of the orbital and ttv plots
for i in range(1000):
    image1 = Image.open(f"Trappist1Simulation/orbitplot{i}.png")
    image2 = Image.open(f"ttvplots/ttvplot{i}.png")
    new_image = Image.new('RGB',(image1.size[0]+image2.size[0], image1.size[1]), (250,250,250))
    new_image.paste(image1,(0,0))
    new_image.paste(image2,(image1.size[0],0))
    if i < 10:
        new_image.save(f"Combined/merged_image00{i}.png","PNG")
    elif i < 100:
        new_image.save(f"Combined/merged_image0{i}.png","PNG")
    else:
        new_image.save(f"Combined/merged_image{i}.png","PNG")

# Use the frames of the merged plots to create a gif
png_dir = os.getcwd() + "/Combined"
images = []
for file_name in sorted(os.listdir(png_dir)):
    if file_name.endswith('.png'):
        file_path = os.path.join(png_dir, file_name)
        images.append(imageio.imread(file_path))
imageio.mimsave('trappist1gif.gif', images)