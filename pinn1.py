import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import os

N = 100
num_samples = 30
num_sensors = 16
ambient = 25

os.makedirs("smooth_chip/images", exist_ok=True)
os.makedirs("smooth_chip/sensors", exist_ok=True)

def create_laplacian(N):
    size = N*N
    main = -4*np.ones(size)
    side = np.ones(size)
    diagonals = [main, side, side, side, side]
    offsets = [0, -1, 1, -N, N]
    return sp.diags(diagonals, offsets, format="csr")

L = create_laplacian(N)

# Fixed realistic sensor layout
sensor_positions = [
    (20,20), (20,50), (20,80),
    (50,20), (50,50), (50,80),
    (80,20), (80,50), (80,80),
    (10,50), (90,50),
    (50,10), (50,90),
    (30,30), (70,70), (50,30)
]

for idx in range(num_samples):

    x = np.arange(N)
    y = np.arange(N)
    X, Y = np.meshgrid(x, y)

    # Smooth Gaussian hotspot
    x0 = np.random.randint(30, 70)
    y0 = np.random.randint(30, 70)
    sigma = np.random.uniform(10, 20)
    intensity = np.random.uniform(200, 400)

    Q = intensity * np.exp(-((X-x0)**2 + (Y-y0)**2)/(2*sigma**2))

    # Add global left-right gradient
    gradient = np.linspace(0, 100, N)
    gradient_field = np.tile(gradient, (N,1))

    b = -Q.flatten()
    T = spla.spsolve(L, b)
    T = T.reshape(N,N)

    T = T - T.min()
    T = ambient + (T / T.max()) * 75

    # Add smooth gradient
    T = 0.7*T + 0.3*gradient_field

    # Add slight sensor-like speckle noise
    noise = np.random.normal(0, 0.5, (N,N))
    T += noise

    # Save image
    plt.imshow(T, cmap='jet')
    plt.axis('off')
    plt.savefig(f"smooth_chip/images/sample_{idx}.png",
                bbox_inches='tight', pad_inches=0)
    plt.close()

    # Extract sensors
    sensor_values = np.array([T[r,c] for r,c in sensor_positions])

    np.savez(
        f"smooth_chip/sensors/sample_{idx}.npz",
        coords=sensor_positions,
        values=sensor_values
    )

print("Smooth chip-style dataset complete.")