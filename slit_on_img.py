#%%
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import xarray as xr
# %%
fnames = glob('*.jpeg')
len(fnames)
# %%
img = Image.open(fnames[0])
img = img.convert('F')
# %%
width, height = img.size
xc = width//2
yc = height//2
# %%
plt.imshow(img)
plt.scatter(xc,yc)
# %%
angles = np.linspace(0,180,height)
# %%
ds = xr.DataArray(
    img,
    coords = {
        'x':('x',angles),
        'y': ('y',angles)
    }
)
# %%
ds.plot.imshow(size = 1,aspect = 'equal')
# %%
plt.figure(figsize=(8, 8))
ax = plt.subplot()
ds.plot(ax=ax)
ax.set_aspect('equal')

# %%
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load the image

image_path = glob('*.jpeg')[0]
img = Image.open(image_path)

# Define the image size
img_size = img.size  # (width, height)
img_center = (img_size[0] // 2, img_size[1] // 2)  # Center of the image

# Define the slit parameters
fov_diameter_km = 300  # Given FOV in km
fov_degrees = 180       # Full field of view in degrees
slit_thickness = 0.1    # Thickness of the slit in degrees
slit_length = 20        # Half-length of the slit in degrees (full length = 40 degrees)

# Convert degrees to pixels
deg_to_pix = img_size[0] / fov_degrees  # Pixels per degree
slit_thickness_pix = slit_thickness * deg_to_pix
slit_length_pix = slit_length * deg_to_pix

# Create slit coordinates
x1, x2 = img_center[0] - slit_thickness_pix / 2, img_center[0] + slit_thickness_pix / 2
y1, y2 = img_center[1] - slit_length_pix, img_center[1] + slit_length_pix

# Plot the image and slit
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(img)
ax.add_patch(plt.Rectangle((x1, y1), slit_thickness_pix, 2 * slit_length_pix, 
                           color='red', alpha=0.7, label="Slit"))

# Formatting
ax.set_xlim(0, img_size[0])
ax.set_ylim(img_size[1], 0)  # Invert y-axis for proper orientation
ax.set_xticks([])
ax.set_yticks([])
ax.set_title("All-Sky Image with Slit Overlay")

plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# Load the image
def plot_slit_on_allsky(allskyimg_path):
    image_path = allskyimg_path
    img = Image.open(image_path)
    img_size = img.size  # (width, height)
    img_center = (img_size[0] // 2, img_size[1] // 2)

    # FOV and Projection Parameters
    fov_degrees = 180  # Full FOV of the image
    slit_thickness = 0.01  # Degrees
    slit_length = 20  # Degrees on both sides from the center (40 deg total)

    # Convert degrees to radians
    fov_radians = np.radians(fov_degrees)
    slit_thickness_rad = np.radians(slit_thickness)
    slit_length_rad = np.radians(slit_length)

    # Fisheye projection function
    def angular_to_pixel(theta, phi, img_center, img_size):
        """Convert sky coordinates (theta, phi) to image pixel coordinates."""
        # Compute normalized radius (r) from center based on fisheye projection
        r = (theta / (fov_radians / 2)) * (img_size[0] / 2)
        # Convert to Cartesian image coordinates
        x = img_center[0] + r * np.sin(phi)
        y = img_center[1] - r * np.cos(phi)  # Negative y because image coordinates are flipped
        return x, y
   
    # Generate slit coordinates in angular space (theta, phi)
    num_points = 100  # Number of points to draw the slit
    theta_vals = np.linspace(-slit_length_rad, slit_length_rad, num_points)  # Range along slit
    phi_vals = np.zeros_like(theta_vals)  # Centered along N-S axis (phi = 0)

    # Convert slit coordinates to pixels
    slit_pixels = np.array([angular_to_pixel(theta, phi, img_center, img_size) for theta, phi in zip(theta_vals, phi_vals)])

    # Plot the image
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img)

    # Draw the slit line
    ax.plot(slit_pixels[:, 0], slit_pixels[:, 1], color='red', linewidth=2, label="Slit")

    # Formatting
    ax.set_xlim(0, img_size[0])
    ax.set_ylim(img_size[1], 0)  # Flip y-axis
    ax.set_xticks([])
    ax.set_yticks([])
    # ax.set_title("All-Sky Image with Corrected Slit Overlay")

    plt.show()

# %%
fnames = glob('*.jpeg')
np.sort(fnames)
# %%
plot_slit_on_allsky(fnames[1])
# %%
