import numpy as np
import matplotlib.pyplot as plt
import cv2

image = cv2.imread('data/discharge_print_128x128.png')

# cv2 opens in BGR format for some reason
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


plt.imshow(image_rgb)
plt.show()


image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
image_gray = image_gray.astype(np.float32)
height, weight = image_gray.shape
plt.imshow(image_gray,cmap="gray")
plt.show()


import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def create_mask(binary_image):
    _, mask = cv2.threshold(binary_image, 1, 255, cv2.THRESH_BINARY)
    return mask.astype(bool)

def create_cond_mask(mask, depth=3):
    kernel = np.ones((depth, depth), np.uint8)
    dilated_mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    cond_mask = dilated_mask ^ mask
    return cond_mask

def generate_mask_binary_image(binary_image, depth=3):
    mask = create_mask(binary_image)
    cond_mask = create_cond_mask(mask, depth)
    return mask, cond_mask

binary_mask = cv2.imread('data/ima01_mask.bmp', cv2.IMREAD_GRAYSCALE)

mask, cond_mask = generate_mask_binary_image(binary_mask, depth=3)

masked_image_gray = image_gray.copy()
masked_image_gray[mask] = 255

combined_visualization = cv2.cvtColor(image_gray.astype(np.uint8), cv2.COLOR_GRAY2RGB)
combined_visualization[mask] = [255, 255, 255]
combined_visualization[cond_mask] = [255, 0, 0]

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(binary_mask, cmap='gray')
plt.title("Binary Mask")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(combined_visualization)
plt.title("Image + Mask on white + Conditioning on red")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(masked_image_gray, cmap='gray', vmin=0, vmax=255)
plt.title('Final Masked Input')
plt.axis('off')

plt.tight_layout()
plt.show()


masked_img_gray = np.where(mask, 255, image_gray)
plt.imshow(masked_img_gray, cmap="gray", vmin=0, vmax=255)


texture_mean = masked_img_gray.mean(where=~mask)
count = masked_img_gray[~mask].size
print("Texture mean =",texture_mean)
print("number of pixels =",count)


t = (image_gray - texture_mean)/((count)** (1/2))
t1 = (image_gray - texture_mean)

spot=np.where(mask, 0, t)



fft_spot = np.fft.fft2(spot)
plt.imshow(fft_spot.real, cmap="gray")


cov = np.fft.ifft2(np.abs(fft_spot) ** 2).real
plt.imshow(cov, cmap="gray")



height, width = cov.shape

# Padding was needed for large masks. It causes big distances between pixels and index errors. To solve this, we pad the image
padded_h, padded_w = height * 2, width * 2
spot_padded = np.zeros((padded_h, padded_w), dtype=np.float32)
spot_padded[:height, :width] = spot

fft_spot_padded = np.fft.fft2(spot_padded)
cov_padded = np.fft.ifft2(np.abs(fft_spot_padded)**2).real
# We need to shift from the left-up side to the center using fftshift
cov_shifted = np.fft.fftshift(cov_padded)

pixels_c_coords = np.argwhere(cond_mask)
N_c = len(pixels_c_coords)
A = np.zeros((N_c, N_c), dtype=np.float32)


# The cv matrix is built in function of h, not (i,j). So we need to calculate each h and then take the correct pixel
for i in range(N_c):
    for j in range(N_c):
        y_i, x_i = pixels_c_coords[i]
        y_j, x_j = pixels_c_coords[j]
        delta_y = y_j - y_i
        delta_x = x_j - x_i
        A[i, j] = cov_shifted[padded_h // 2 + delta_y, padded_w // 2 + delta_x]





## Solve the linear problem:
# A * ψ₁ = b

# b = (u - v-)*

cond_array = np.where(cond_mask,image_gray,0)

# Making it a vector
cond_array = cond_array.flatten()

# Removing the mask elements
cond_array = cond_array[cond_array != 0]

b = cond_array - texture_mean
kriging_w_1 = (np.linalg.pinv(A) @ b).real

kriging_w_1.shape



pixels_m_coords = np.argwhere(mask)
N_m = len(pixels_m_coords)
Gamma_mc = np.zeros((N_m, N_c), dtype=np.float32)

for i in range(N_m):
    for j in range(N_c):
        y_i, x_i = pixels_m_coords[i]
        y_j, x_j = pixels_c_coords[j]
        delta_y = y_j - y_i
        delta_x = x_j - x_i
        Gamma_mc[i, j] = cov_shifted[padded_h // 2 + delta_y, padded_w // 2 + delta_x]

kriging_component_M = (Gamma_mc @ kriging_w_1).real
kriging_component_matrix = np.zeros_like(image_gray)
kriging_component_matrix[mask] = kriging_component_M

plt.imshow(kriging_component_matrix, cmap='gray')
plt.title("Kriging Component")
plt.colorbar()
plt.show()




# Generate a gaussian matrix with the padded size
W_padded = np.random.randn(padded_h, padded_w).astype(np.float32)
fft_W_padded = np.fft.fft2(W_padded)
F_padded = np.fft.ifft2(fft_spot_padded * fft_W_padded).real

# Remove the padding
F = F_padded[:height, :width]

F_c = F[cond_mask]
kriging_w_2 = (np.linalg.pinv(A) @ F_c).real # Solve for ψ₂

F_star_M = (Gamma_mc @ kriging_w_2).real
F_M = F[mask]

innovation_vector = F_M - F_star_M
innovation_component_matrix = np.zeros_like(image_gray)
innovation_component_matrix[mask] = innovation_vector



inpainted_region = texture_mean + kriging_component_matrix + innovation_component_matrix

final_image = np.where(mask, inpainted_region, image_gray)
final_image = np.clip(final_image, 0, 255).astype(np.uint8)


fig, axes = plt.subplots(1, 4, figsize=(20, 5))

axes[0].imshow(masked_img_gray, cmap='gray', vmin=0, vmax=255)
axes[0].set_title("Masked Image")

axes[1].imshow(kriging_component_matrix, cmap='gray')
axes[1].set_title("Kriging Component")

axes[2].imshow(innovation_component_matrix, cmap='gray')
axes[2].set_title("Innovation Component")

axes[3].imshow(final_image, cmap='gray', vmin=0, vmax=255)
axes[3].set_title("Final result")

for ax in axes:
    ax.axis('off')

plt.tight_layout()
plt.show()