#%%
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import xarray as xr
import os
from tqdm import tqdm


def plot_slit_on_allsky(allskyimg_path, save_path='./', plot_show=False):
    """plots the slit postion on the all sky images from IRF.

    Args:
        allskyimg_path (_type_): path to directory with raw all sky images (.jpeg).
        save_path (str, optional): path to save directory. Defaults to './'.
        plot_show (bool, optional): if True, plot will show. Defaults to False.

    Returns:
        _type_: saves a .png image with the slit position on the all sky image.
    """    
    DPI = 100
    image_path = allskyimg_path
    img = Image.open(image_path)
    img_size = img.size  # (width, height)
    img_center = (img_size[0] // 2, img_size[1] // 2)

    # FOV and Projection Parameters
    fov_degrees = 180  # Full FOV of the image
    slit_thickness = 0.01  # Degrees
    slit_length = 10  # Degrees on both sides from the center (20 deg total)

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
    num_points = 100 # Number of points to draw the slit
    theta_vals = np.linspace(-slit_length_rad, slit_length_rad, num_points)  # Range along slit
    phi_vals = np.zeros_like(theta_vals)  # Centered along N-S axis (phi = 0)

    # Convert slit coordinates to pixels
    slit_pixels = np.array([angular_to_pixel(theta, phi, img_center, img_size) for theta, phi in zip(theta_vals, phi_vals)])

    # Plot the image
    fig, ax = plt.subplots(figsize=(img_size[0]/DPI,img_size[-1]/DPI))
    ax.imshow(img)
    # Draw the slit line
    ax.plot(slit_pixels[:, 0], slit_pixels[:, 1], color='red', linewidth=2, label="Slit",markersize = 0.5 )
        
    # Formatting
    ax.set_aspect('equal')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0) #Remove padding around the image
    ax.set_xlim(0, img_size[0])
    ax.set_ylim(img_size[1], 0)  # Flip y-axis
    ax.set_xticks([])

    ax.set_yticks([])
    #save plot
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, os.path.basename(image_path).replace('.jpeg', '.png')), dpi=DPI)
    if plot_show: plt.show()
    else: plt.close(fig)
    
    

# %%
if __name__ == '__main__':
    fdir = 'imgs'
    fpath = os.path.join(fdir,'raw/*.jpeg')
    fnames = glob(fpath)
    np.sort(fnames)
    print(len(fnames))

    for i in tqdm(fnames):
        plot_slit_on_allsky(i, os.path.join(fdir,'with_slit'))
