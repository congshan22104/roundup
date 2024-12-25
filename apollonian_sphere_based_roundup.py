import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Parameters
num_uavs = 4                # Number of UAVs
capture_distance = 1.0      # Distance to maintain around the target
max_speed = 15             # Maximum speed of UAVs
step_time = 0.1             # Time step for simulation
sim_steps = 200             # Total steps for simulation
safety_distance = 2.0       # Minimum safe distance to avoid collisions

# Initialize target
np.random.seed(42)
target_pos = np.random.rand(3) * 40 - 20  # Random position in range [-10, 10, -10]
target_vel = np.random.rand(3) * 20 - 10    # Random velocity in range [-10, 10, -10]

def update_target_velocity():
    """Update the target's velocity to simulate a zigzag escape path."""
    global target_vel
    zigzag_factor = 10.0  # Increase for more abrupt turns
    axis = np.random.choice([0, 1, 2])  # Randomly choose an axis to zigzag
    target_vel[axis] += (np.random.rand() - 0.5) * zigzag_factor
    target_vel = np.clip(target_vel, -10, 10)  # Limit velocity to a max of 10

# Fixed UAV positions (float type to avoid type casting issues)
uav_positions = np.array([
    [-30.0, -30.0, -30.0],
    [-30.1, -30.1, -30.0],
    [-30.0, -30.1, -30.1],
    [-30.1, -30.0, -30.1]
])

# Calculate Apollonian Sphere center and radius for a UAV
def apollonian_sphere(target_pos, uav_pos, k):
    if abs(1 - k**2) < 1e-6:  # Avoid division by zero
        return target_pos, 1.0  # Default center and radius
    center = target_pos + (uav_pos - target_pos) / (1 - k**2)
    radius = np.linalg.norm(target_pos - uav_pos) * k / abs(1 - k**2)
    return center, radius

# Calculate k dynamically
def calculate_k(target_vel, uav_pos, target_pos):
    uav_to_target_dist = np.linalg.norm(target_pos - uav_pos)
    if uav_to_target_dist == 0:
        return 1  # Avoid division by zero, assume equal speed ratio
    target_speed = np.linalg.norm(target_vel)
    uav_speed = max_speed if uav_to_target_dist > capture_distance else target_speed
    return uav_speed / target_speed if target_speed > 0 else 1

# Assign UAVs to positions during the initial phase (independent Apollonian Spheres)
def assign_uav_positions_initial(target_pos, uav_positions):
    assigned_positions = []
    for uav_pos in uav_positions:
        k = calculate_k(target_vel, uav_pos, target_pos)
        center, radius = apollonian_sphere(target_pos, uav_pos, k)
        # Adjust the assigned position to move toward the target direction
        direction_to_target = target_pos - uav_pos
        direction_to_target /= np.linalg.norm(direction_to_target)  # Normalize direction
        assigned_position = center + direction_to_target * radius

        assigned_positions.append(assigned_position)
    return np.array(assigned_positions)

# Generate evenly distributed points on a sphere
def generate_sphere_points(center, radius, num_points):
    indices = np.arange(0, num_points, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices / num_points)  # Elevation angles
    theta = np.pi * (1 + 5**0.5) * indices         # Azimuthal angles (golden angle)

    x = radius * np.sin(phi) * np.cos(theta) + center[0]
    y = radius * np.sin(phi) * np.sin(theta) + center[1]
    z = radius * np.cos(phi) + center[2]

    return np.column_stack((x, y, z))

# Assign UAVs to evenly spaced positions on a shared Apollonian Sphere
def assign_uav_positions_capture(target_pos, uav_positions):
    radius = 5.0  # Fixed radius for simplicity
    sphere_points = generate_sphere_points(target_pos, radius, num_uavs)

    # Assign positions based on proximity
    assigned_positions = []
    available_positions = sphere_points.tolist()
    for uav_pos in uav_positions:
        distances = [np.linalg.norm(np.array(pos) - uav_pos) for pos in available_positions]
        closest_idx = np.argmin(distances)
        assigned_positions.append(available_positions.pop(closest_idx))

    return np.array(assigned_positions)


# Avoid collisions between UAVs and with the target
def avoid_collisions(uav_positions, target_pos):
    for i in range(num_uavs):
        # Check distance to target
        direction_to_target = target_pos - uav_positions[i]
        distance_to_target = np.linalg.norm(direction_to_target)
        if distance_to_target < safety_distance:
            # Move away from the target
            uav_positions[i] -= 0.5 * direction_to_target / distance_to_target

        # Check distance to other UAVs
        for j in range(num_uavs):
            if i != j:
                direction_to_uav = uav_positions[j] - uav_positions[i]
                distance_to_uav = np.linalg.norm(direction_to_uav)
                if distance_to_uav < safety_distance:
                    # Move away from the other UAV
                    uav_positions[i] -= 0.5 * direction_to_uav / distance_to_uav

    return uav_positions

# Update UAV positions to converge to assigned positions
def update_uavs_cooperative(assigned_positions, uav_positions):
    for i in range(num_uavs):
        direction = assigned_positions[i] - uav_positions[i]
        distance = np.linalg.norm(direction)
        if distance > 0:
            velocity = max_speed * direction / distance
            uav_positions[i] += velocity * step_time
    return avoid_collisions(uav_positions, target_pos)


# Check if all UAVs are in the capture phase
def in_capture_phase(uav_positions, target_pos, threshold=7.0):
    distances = np.linalg.norm(uav_positions - target_pos, axis=1)
    return np.all(distances <= threshold)

# Check if capture is successful
def is_capture_successful(target_pos, uav_positions, capture_distance):
    distances = np.linalg.norm(uav_positions - target_pos, axis=1)
    return np.all(distances <= capture_distance + 0.1)

# Set up 3D plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim([-50, 50])
ax.set_ylim([-50, 50])
ax.set_zlim([-50, 50])
ax.set_box_aspect([1, 1, 1])

# Plot elements
target_dot, = ax.plot([], [], [], 'ro', label="Target")
uav_dots, = ax.plot([], [], [], 'bo', label="UAVs")

# Animation update function
def update(frame):
    global target_pos, target_vel, uav_positions

    # Update target position
    # Update target position and velocity
    update_target_velocity()
    target_pos += target_vel * step_time

    # Decide phase: Initial or Capture
    if in_capture_phase(uav_positions, target_pos):
        assigned_positions = assign_uav_positions_capture(target_pos, uav_positions)
    else:
        assigned_positions = assign_uav_positions_initial(target_pos, uav_positions)

    # Update UAV positions cooperatively with dynamic speed adjustment
    uav_positions = update_uavs_cooperative(assigned_positions, uav_positions)

    # Check if capture is successful
    if is_capture_successful(target_pos, uav_positions, capture_distance):
        print(f"Capture successful at frame {frame}")
        ani.event_source.stop()

    # Update plots
    target_dot.set_data([target_pos[0]], [target_pos[1]])
    target_dot.set_3d_properties([target_pos[2]])
    uav_dots.set_data(uav_positions[:, 0], uav_positions[:, 1])
    uav_dots.set_3d_properties(uav_positions[:, 2])
    return [target_dot, uav_dots]

# Create animation
ani = FuncAnimation(fig, update, frames=sim_steps, interval=50, blit=False)
plt.legend()
plt.show()
