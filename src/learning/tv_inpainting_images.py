import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces

# Load the dataset
faces = fetch_olivetti_faces()
images = faces.images

# Set up a grid to plot the 40 unique people
fig, axes = plt.subplots(4, 10, figsize=(15, 6),
                         subplot_kw={'xticks':[], 'yticks':[]},
                         gridspec_kw=dict(hspace=0.1, wspace=0.1))

# Loop through and grab the first image of every 10-image block
for i, ax in enumerate(axes.flat):
    # i * 10 gives us index 0, 10, 20, 30... which are the first photos of each subject
    ax.imshow(images[i * 10], cmap='gray')
    ax.set_title(f"Subject {i}", fontsize=10)

plt.show()