import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pandas as pd

import matplotlib.animation as animation

fig, ax = plt.subplots()

# df = pd.read_csv('/home/walml/repos/euclid-morphology/upload/master_catalog_during_2024_07_24_euclid_challenge.csv')
df = pd.read_csv('/home/walml/repos/euclid-morphology/upload/master_catalog_during_2024_07_30_euclid_challenge_testing.csv')
df.sort_values(by='segmentation_area', inplace=True, ascending=False)
# print(df.iloc[0])
print(len(df))

# scat = ax.imshow(Image.open(df.iloc[30]['jpg_loc_composite']))


# def animate(i):
#     ax.imshow(Image.open(df.iloc[30+i]['jpg_loc_composite'])), 
#     ax.axis('off')
#     return ax,

# x = np.arange(0, 20)

# ani = animation.FuncAnimation(fig, animate, repeat=True,
#                                     frames=len(x) - 1, interval=50)

                                    
# writer = animation.PillowWriter(fps=5,
#                                 metadata=dict(artist='Mike Walmsley')
# )
#                                 #,
#                                 # b#itrate=3600)
# ani.save('euclid.gif', writer=writer)


def make_gif():
    frames = [Image.open(image) for image in df['jpg_loc_composite_resized'][1501:1561]]
    frame_one = frames[0]
    frame_one.save("euclid_gif.gif", format="GIF", append_images=frames,
               save_all=True, duration=200, loop=0)
make_gif()
# plt.show()