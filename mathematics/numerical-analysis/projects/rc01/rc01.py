#%%
#Libraries and class
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from scipy.optimize import fsolve

@dataclass
class Constants:
    l1: float
    l2: float
    l3: float
    gamma: float
    x1: float
    x2: float
    y2: float
    p1: float
    p2: float
    p3: float
    
#%%
# Define the function for f(θ)
def f(theta, constants):
    """
    Calculates a value based on the given angle theta and constants object.

    Parameters:
    theta (float): The angle in radians.
    constants (Constants): An object containing the necessary constants.

    Returns:
    float: The calculated result.
    """
    l1, l2, l3 = constants.l1, constants.l2, constants.l3
    gamma = constants.gamma
    x1, x2, y2 = constants.x1, constants.x2, constants.y2
    p1, p2, p3 = constants.p1, constants.p2, constants.p3

    a2 = l3 * np.cos(theta) - x1 
    b2 = l3 * np.sin(theta)
    a3 = l2 * np.cos(theta + gamma) - x2
    b3 = l2 * np.sin(theta + gamma) - y2
    d = 2 * (a2 * b3 - b2 * a3)
    
    n1 = b3 * (p2**2 - p1**2 - a2**2 - b2**2) - b2 * (p3**2 - p1**2 - a3**2 - b3**2)
    n2 = -a3 * (p2**2 - p1**2 - a2**2 - b2**2) + a2 * (p3**2 - p1**2 - a3**2 - b3**2)
    
    return n1**2 + n2**2 - p1**2 * d**2

#%%
# Question 4a
constants = Constants(
    l1=2, 
    l2=np.sqrt(2),
    l3=np.sqrt(2),
    gamma=np.pi / 2, 
    x1=4, 
    x2=0, 
    y2=4, 
    p1=np.sqrt(5),
    p2=np.sqrt(5), 
    p3=np.sqrt(5)
)

theta = np.pi / 4

# Evaluate the function
result = f(theta, constants)
print(f'f(θ=π/4) = {result}')

#%%
# Generate theta values and compute f(theta) for plotting
theta_values = np.linspace(-np.pi, np.pi, 400)
results = [f(theta, constants) for theta in theta_values]

# Plotting f(theta)
plt.figure(figsize=(10, 6))
plt.plot(theta_values, results, label=r'$f(\theta)$', linewidth=2)
plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
plt.axvline(-np.pi/4, color='red', linestyle=':', linewidth=2, label=r'$\theta = -\pi/4$')
plt.axvline(np.pi/4, color='red', linestyle=':', linewidth=2, label=r'$\theta = \pi/4$')
plt.xlabel(r'$\theta$', fontsize=14)
plt.ylabel(r'$f(\theta)$', fontsize=14)
plt.title(r'Function of $f(\theta)$ for Stewart Platform', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)
plt.show()

#%%
# Create Helper Functions
def get_x_y(theta, constants):
    """
    Returns the coordinates x and y for the given angle theta and constants object.

    Parameters:
    theta (float): The angle in radians.
    constants (Constants): An object containing the necessary constants.
    
    Returns:
    tuple: The coordinates (x, y).
    """
    l1, l2, l3 = constants.l1, constants.l2, constants.l3
    gamma = constants.gamma
    x1, x2, y2 = constants.x1, constants.x2, constants.y2
    p1, p2, p3 = constants.p1, constants.p2, constants.p3

    a2 = l3 * np.cos(theta) - x1 
    b2 = l3 * np.sin(theta)
    a3 = l2 * np.cos(theta + gamma) - x2
    b3 = l2 * np.sin(theta + gamma) - y2
    
    d = 2 * (a2 * b3 - b2 * a3)
    n1 = b3 * (p2**2 - p1**2 - a2**2 - b2**2) - b2 * (p3**2 - p1**2 - a3**2 - b3**2)
    n2 = -a3 * (p2**2 - p1**2 - a2**2 - b2**2) + a2 * (p3**2 - p1**2 - a3**2 - b3**2)
    
    x = n1 / d
    y = n2 / d
    
    return x, y


def get_points(x, y, theta, constants):
    """
    Returns the points of the triangle based on x, y, and theta using the constants object.

    Parameters:
    x (float): The x-coordinate.
    y (float): The y-coordinate.
    theta (float): The angle in radians.
    constants (Constants): An object containing the necessary constants.

    Returns:
    list: A list containing the points (l1_point, l2_point, l3_point) of the triangle.
    """
    l1, l2, l3 = constants.l1, constants.l2, constants.l3
    gamma = constants.gamma

    # Base point of the triangle (first vertex)
    l1_point = (x, y)
    
    # Second vertex of the triangle
    l2_x = x + (l3 * np.cos(theta))
    l2_y = y + (l3 * np.sin(theta))
    l2_point = (np.round(l2_x, 3), np.round(l2_y))  # Rounded to 3 decimal places for clarity
    
    # Third vertex of the triangle
    l3_x = x + (l2 * np.cos(theta + gamma))
    l3_y = y + (l2 * np.sin(theta + gamma))
    l3_point = (np.round(l3_x), np.round(l3_y))  # Rounded to 3 decimal places for clarity
    
    return [l1_point, l2_point, l3_point]

def get_anchor_points(constants):
    """
    Returns the anchor points for the strut anchor points using the constants object.

    Parameters:
    constants (Constants): An object containing the necessary constants (x1, x2, y2).

    Returns:
    list: A list containing tuples with the coordinates of the anchor points.
    """
    x1, x2, y2 = constants.x1, constants.x2, constants.y2
    
    return [(0, 0), (x1, 0), (x2, y2)]

def plot_triangle(ax, points, anchor_points, x_limits=None, y_limits=None, x_step=None, y_step=None):
    """
    Plots a triangle given the points and anchor points on the provided axis.

    Parameters:
    ax: The axis on which to plot the triangle.
    points: The points of the triangle (list of 3 points).
    anchor_points: The anchor points (list of 2 or more points).
    x_limits (tuple, optional): Tuple specifying the x-axis limits (x_min, x_max).
    y_limits (tuple, optional): Tuple specifying the y-axis limits (y_min, y_max).
    x_step (float, optional): Step size for the x-axis grid.
    y_step (float, optional): Step size for the y-axis grid.

    Returns:
    None
    """
    points = np.array(points)
    anchor_points = np.array(anchor_points)
    
    # Extract x and y coordinates for the triangle points
    x_coords = points[:, 0]
    y_coords = points[:, 1]
    
    # Close the triangle by appending the first point at the end
    x_closed = np.append(x_coords, x_coords[0])
    y_closed = np.append(y_coords, y_coords[0])
    
    # Plot the triangle with red lines
    ax.plot(x_closed, y_closed, 'r-', linewidth=3.5)
    
    # Plot blue dots at the triangle vertices
    ax.plot(x_coords, y_coords, 'bo', markersize=8)
    
    # Plot lines from anchor points to triangle points
    for i, anchor in enumerate(anchor_points):
        if i < len(points):  # Ensure we stay within bounds
            ax.plot([anchor[0], points[i, 0]], [anchor[1], points[i, 1]], 'b-', linewidth=1.5)
    
    # Plot blue dots at the anchor points
    ax.plot(anchor_points[:, 0], anchor_points[:, 1], 'bo', markersize=8)
    
    # Set axis labels
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    
    # Set x-axis limits if provided
    if x_limits is not None:
        ax.set_xlim(x_limits)
    
    # Set y-axis limits if provided
    if y_limits is not None:
        ax.set_ylim(y_limits)
    
    # Set grid step increments if limits are provided
    if x_step is not None and x_limits is not None:
        ax.set_xticks(np.arange(x_limits[0], x_limits[1] + x_step, x_step))  # Adjust x-axis ticks
    if y_step is not None and y_limits is not None:
        ax.set_yticks(np.arange(y_limits[0], y_limits[1] + y_step, y_step))  # Adjust y-axis ticks
    
    # Add grid for better visualization
    ax.grid(True)

#%%
# Create Plot
theta = np.pi / 4
theta_negative = -np.pi / 4

# First triangle points
x, y = get_x_y(theta_negative, constants)  # Use constants object here
points1 = get_points(x, y, theta_negative, constants)  # Use constants object
anchor_points = get_anchor_points(constants)  # Use constants object for anchor points

# Second triangle points
x, y = get_x_y(theta, constants)  # Use constants object here
points2 = get_points(x, y, theta, constants)  # Use constants object

# Create subplots for side-by-side triangles
fig, axes = plt.subplots(1, 2, figsize=(11, 5))

# Plot the first triangle on the first axis with custom axis limits
plot_triangle(axes[0], points1, anchor_points, x_limits=(-0.25, 4.25), y_limits=(-0.25, 4.25))

# Plot the second triangle on the second axis with custom axis limits
plot_triangle(axes[1], points2, anchor_points, x_limits=(-0.25, 4.25), y_limits=(-0.25, 4.25))

# Adjust layout to avoid overlap
plt.tight_layout()

# Display the concatenated plots
plt.show()
#%%
constants = Constants(
    l1=3, 
    l2=3 * np.sqrt(2), 
    l3=3, 
    gamma=np.pi / 4,  # GAMMA in lowercase
    x1=5, 
    x2=0, 
    y2=6,  # Updated Y2 as per your request
    p1=5, 
    p2=5, 
    p3=3
)

theta_values = np.linspace(-np.pi, np.pi, 400)

plt.figure(figsize=(10, 6))
plt.plot(theta_values, f(theta_values, constants), label=r'$f(\theta)$')  # Default linewidth
plt.axhline(0, color='black', linestyle='--', linewidth=0.8)  # Default linewidth
plt.xlabel(r'$\theta$', fontsize=14)
plt.ylabel(r'$f(\theta)$', fontsize=14)
plt.title(r'Function of $f(\theta)$ for Stewart Platform', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)
plt.show()

#%%

# Function to find roots using fsolve
def find_roots(constants, initial_guesses):
    """
    Finds roots of f(theta) using different initial guesses.
    
    Parameters:
    constants (Constants): Object containing the necessary constants.
    initial_guesses (list or array): List of initial guesses for fsolve to start from.
    
    Returns:
    list: A list of unique roots.
    """
    # Use fsolve to find the roots based on the provided initial guesses
    roots = []
    for guess in initial_guesses:
        root = fsolve(f, guess, args=(constants))[0]
        roots.append(root)
    
    # Round the roots to avoid numerical precision issues and remove duplicates
    roots = np.round(roots, decimals=6)
    unique_roots = np.unique(roots)
    
    return unique_roots

# Provide initial guesses for fsolve to find the roots
initial_guesses = [-1, -np.pi / 3, np.pi / 3, .5, 2]  # Customize this list

# Find and print the roots using the custom initial guesses
roots = find_roots(constants, initial_guesses)
print(f"The roots of f(theta)in the interval are : {roots}")

# Function to calculate the length of the struts
def calculate_strut_lengths(points, anchor_points):
    lengths = []
    for i in range(3):
        length = np.sqrt((points[i][0] - anchor_points[i][0])**2 + (points[i][1] - anchor_points[i][1])**2)
        lengths.append(length)
    return lengths

# %%
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes = axes.flatten()

anchor_points = get_anchor_points(constants)

# Loop through up to four roots
for i, theta in enumerate(roots[:4]):  # Limit to the first four roots
    x, y = get_x_y(theta, constants)
    points = get_points(x, y, theta, constants)
    
    # Plot the triangle in the corresponding subplot with custom limits
    plot_triangle(axes[i], points, anchor_points, x_limits=(-2.5, 7.5), y_limits=(-2, 7), x_step=2.5, y_step=2)
    axes[i].set_title(rf"$\theta$ = {theta}")

    # Calculate and verify strut lengths
    lengths = calculate_strut_lengths(points, anchor_points)
    print(f"For root {np.round(theta, 3)}, strut lengths are: {np.round(lengths)}")
    print(f"Expected: p1={constants.p1}, p2={constants.p2}, p3={constants.p3}\n")

# Turn off any unused subplots if fewer than four roots
for j in range(len(roots), 4):
    axes[j].axis('off')

# Adjust layout
plt.tight_layout()
plt.show()
# %%
constants = Constants(
    l1=3, 
    l2=3 * np.sqrt(2), 
    l3=3, 
    gamma=np.pi / 4, 
    x1=5, 
    x2=0, 
    y2=6,  
    p1=5, 
    p2=7, 
    p3=3
)

theta_values = np.linspace(-np.pi, np.pi, 400)

plt.figure(figsize=(10, 6))
plt.plot(theta_values, f(theta_values, constants), label=r'$f(\theta)$')  
plt.axhline(0, color='black', linestyle='--', linewidth=0.8)  
plt.xlabel(r'$\theta$', fontsize=14)
plt.ylabel(r'$f(\theta)$', fontsize=14)
plt.title(r'Function of $\theta$ for Stewart Platform', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)
plt.show()

#%%

# Provide initial guesses for fsolve to find the roots
initial_guesses = [-.7, -.4, .01, .4, .9, 2.5 ]  # Customize this list

# Find and print the roots using the custom initial guesses
roots = find_roots(constants, initial_guesses)
print(f"The roots of f(θ) in the interval are : {roots}")

# Set up the 2x3 grid for plotting the six poses
fig, axes = plt.subplots(2, 3, figsize=(15, 10))  # Create a 2x3 grid
axes = axes.flatten()  # Flatten the 2D array of axes for easier access

# Get the anchor points
anchor_points = get_anchor_points(constants)

# Loop through the six roots and plot each pose
for i, theta in enumerate(roots[:6]):  # Limit to the first six roots
    x, y = get_x_y(theta, constants)
    points = get_points(x, y, theta, constants)
    # Plot the triangle in the corresponding subplot with custom limits
    plot_triangle(axes[i], points, anchor_points, x_limits=(-5.5, 5.5), y_limits=(-.5, 10), )
    axes[i].set_title(rf"$\theta$ = {theta}")

    # Calculate and verify strut lengths
    lengths = calculate_strut_lengths(points, anchor_points)
    print(f"For root {np.round(theta, 3)}, strut lengths are: {np.round(lengths)}")
    print(f"Expected: p1={constants.p1}, p2={constants.p2}, p3={constants.p3}\n")

# Turn off any unused subplots (though in this case, we should have exactly 6)
for j in range(len(roots), 6):
    axes[j].axis('off')

# Adjust layout
plt.tight_layout()
plt.show()


# %%
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

# Set a threshold for considering a valid root (how close to zero we want f(theta) to be)
ROOT_THRESHOLD = 1e-6

# Function to find roots for a given p2, and check if they are valid
def find_roots_for_p2(p2_value, constants, initial_guesses, ax=None):
    """
    Adjusts p2 in the constants object, finds the roots, and returns the number of unique roots.
    Also plots f(theta) for the current p2 value on the provided axis.
    """
    # Update p2 in constants
    constants.p2 = p2_value
    
    # Generate theta values and compute f(theta)
    theta_values = np.linspace(-np.pi, np.pi, 400)
    f_values = [f(theta, constants) for theta in theta_values]
    
    # Plot f(theta) for the current p2 value on the provided axis
    ax.plot(theta_values, f_values, label=fr'$f(\theta), p_2 = {p2_value:.3f}$')
    ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
    ax.set_xlabel(r'$\theta$', fontsize=14)
    ax.set_ylabel(r'$f(\theta)$', fontsize=14)
    ax.set_title(fr'$p_2 = {p2_value:.3f}$')
    ax.legend(fontsize=10)
    ax.grid(True)
    
    # Find the roots for the given p2 value
    roots = []
    for guess in initial_guesses:
        root = fsolve(f, guess, args=(constants))[0]
        
        # Check if the found root is valid (i.e., f(root) is close to zero)
        if abs(f(root, constants)) < ROOT_THRESHOLD:
            roots.append(root)
    
    # Convert to numpy array and round the roots to avoid precision issues
    roots = np.round(np.array(roots), decimals=6)
    unique_roots = np.unique(roots)
    
    # Print the number of valid roots and the roots themselves
    print(f"p2 = {p2_value:.3f}: Found {len(unique_roots)} valid roots: {unique_roots}")
    
    return unique_roots

# Function to iterate over possible p2 values and append plots in a grid (wrap after 3)
def find_p2_with_two_roots(constants, initial_guesses, p2_start=-1):
    """
    Iterates over possible p2 values starting at p2_start, plots f(theta), and prints the number of roots.
    The plots wrap after 3 per row.
    
    Parameters:
    - constants: The Constants object.
    - initial_guesses: List of initial guesses for root finding.
    - p2_start: Starting value of p2.
    """
    p2 = p2_start
    plot_count = 0
    max_plots_per_row = 3  # Wrap after 3 plots per row
    total_plots = 6  # You can change this to adjust the number of total plots
    
    # Calculate the number of rows needed (wrap after 3)
    num_rows = (total_plots + max_plots_per_row - 1) // max_plots_per_row
    
    # Create a figure with a 3xN grid
    fig, axes = plt.subplots(num_rows, max_plots_per_row, figsize=(15, num_rows * 5))
    axes = axes.flatten()  # Flatten the 2D array of axes for easier access
    fig.subplots_adjust(hspace=0.3, wspace=0.3)  # Adjust the space between subplots
    
    # Iterate to plot p2 and find roots
    while plot_count < total_plots:
        # Plot for the current p2 value and check the roots
        unique_roots = find_roots_for_p2(p2, constants, initial_guesses, ax=axes[plot_count])
        
        if len(unique_roots) == 2:  # Check if there are exactly 2 unique roots
            print(f"Found p2={p2} with two distinct roots: {unique_roots}")
            plt.tight_layout()
            plt.show()
            return p2, unique_roots
        
        # Increment p2 and plot the next iteration
        p2 += 1
        plot_count += 1
    
    # Show the final figure with all appended plots
    plt.tight_layout()
    plt.show()

# Example constants (with p2 placeholder)
constants = Constants(
    l1=3, 
    l2=3 * np.sqrt(2), 
    l3=3, 
    gamma=np.pi / 4, 
    x1=5, 
    x2=0, 
    y2=6,  
    p1=5, 
    p2=None,  # To be found
    p3=3
)

# Initial guesses for root finding
initial_guesses = [-np.pi/2, 0, np.pi/2]

# Start p2 at 0 and increment by 1 each time, looking for exactly 2 roots
find_p2_with_two_roots(constants, initial_guesses, p2_start=0)


# %%
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

# Set a threshold for considering a valid root (how close to zero we want f(theta) to be)
ROOT_THRESHOLD = 1e-6

# Function to find roots for a given p2, and check if they are valid
def find_roots_for_p2(p2_value, constants, initial_guesses, ax=None):
    """
    Adjusts p2 in the constants object, finds the roots, and returns the number of unique roots.
    Also plots f(theta) for the current p2 value on the provided axis.
    """
    # Update p2 in constants
    constants.p2 = p2_value
    
    # Generate theta values and compute f(theta)
    theta_values = np.linspace(-np.pi, np.pi, 400)
    f_values = [f(theta, constants) for theta in theta_values]
    
    # Plot f(theta) for the current p2 value on the provided axis
    ax.plot(theta_values, f_values, label=fr'$f(\theta), p_2 = {p2_value:.3f}$')
    ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
    ax.set_xlabel(r'$\theta$', fontsize=14)
    ax.set_ylabel(r'$f(\theta)$', fontsize=14)
    ax.set_title(fr'$p_2 = {p2_value:.3f}$')
    ax.legend(fontsize=10)
    ax.grid(True)
    
    # Find the roots for the given p2 value
    roots = []
    for guess in initial_guesses:
        root = fsolve(f, guess, args=(constants))[0]
        
        # Check if the found root is valid (i.e., f(root) is close to zero)
        if abs(f(root, constants)) < ROOT_THRESHOLD:
            roots.append(root)
    
    # Convert to numpy array and round the roots to avoid precision issues
    roots = np.round(np.array(roots), decimals=6)
    unique_roots = np.unique(roots)
    
    # Print the number of valid roots and the roots themselves
    print(f"p2 = {p2_value:.3f}: Found {len(unique_roots)} valid roots: {unique_roots}")
    
    return unique_roots

# Function to iterate over possible p2 values and append plots in a grid (wrap after 3)
def find_p2_with_two_roots(constants, initial_guesses, p2_start=-1, total_plots=6):
    """
    Iterates over possible p2 values starting at p2_start, plots f(theta), and prints the number of roots.
    The plots wrap after 3 per row.
    
    Parameters:
    - constants: The Constants object.
    - initial_guesses: List of initial guesses for root finding.
    - p2_start: Starting value of p2.
    - total_plots: Number of plots to show before stopping.
    """
    p2 = p2_start
    plot_count = 0
    max_plots_per_row = 3  # Wrap after 3 plots per row
    
    # Calculate the number of rows needed (wrap after 3)
    num_rows = (total_plots + max_plots_per_row - 1) // max_plots_per_row
    
    # Create a figure with a 3xN grid
    fig, axes = plt.subplots(num_rows, max_plots_per_row, figsize=(15, num_rows * 5))
    axes = axes.flatten()  # Flatten the 2D array of axes for easier access
    fig.subplots_adjust(hspace=0.3, wspace=0.3)  # Adjust the space between subplots
    
    # Iterate to plot p2 and find roots
    while plot_count < total_plots:
        # Plot for the current p2 value and check the roots
        unique_roots = find_roots_for_p2(p2, constants, initial_guesses, ax=axes[plot_count])
        
        if len(unique_roots) == 2:  # Check if there are exactly 2 unique roots
            print(f"Found p2={p2} with two distinct roots: {unique_roots}")
        
        # Increment p2 and plot the next iteration
        p2 += 1
        plot_count += 1
    
    # Show the final figure with all appended plots
    plt.tight_layout()
    plt.show()

# Example constants (with p2 placeholder)
constants = Constants(
    l1=3, 
    l2=3 * np.sqrt(2), 
    l3=3, 
    gamma=np.pi / 4, 
    x1=5, 
    x2=0, 
    y2=6,  
    p1=5, 
    p2=None,  # To be found
    p3=3
)

# Initial guesses for root finding
initial_guesses = [-np.pi/2, 0, np.pi/2]

# Start p2 at -1 and increment by 1 each time, looking for exactly 2 roots
find_p2_with_two_roots(constants, initial_guesses, p2_start=-1, total_plots=6)

# %%
import numpy as np
from scipy.optimize import fsolve

# Set a threshold for considering a valid root
ROOT_THRESHOLD = 1e-13

def find_roots(constants, initial_guesses):
    """
    Finds roots of f(θ) using different initial guesses and the fsolve method.
    
    Parameters:
    constants (Constants): Object containing the necessary constants.
    initial_guesses (list or array): List of initial guesses for fsolve to start from.
    
    Returns:
    list: A list of unique roots that are close to zero (based on the ROOT_THRESHOLD).
    """
    # Create an empty list to store the roots found    
    roots = []
    # Iterate over each initial guess and find the root using fsolve
    for guess in initial_guesses:
        root = fsolve(f, guess, args=(constants), xtol=1e-12)[0]  # Find root for each guess
        
        # Check if the found root is valid (i.e., f(root) is close to zero)
        if abs(f(root, constants)) < ROOT_THRESHOLD:
            roots.append(root)  # Append the found root to the list if it satisfies the threshold
    
    # Return only unique roots to avoid duplicates
    unique_roots = np.unique(roots)
    return unique_roots

# Define the function to dynamically adjust initial guesses and iterate over p2 values
def find_p2_intervals(constants, p2_min, p2_max, p2_step):
    """
    Iterates over p2 values and adjusts the number of initial guesses dynamically
    to find the intervals where there are 0, 2, 4, or 6 roots.
    
    Parameters:
    constants: The Constants object.
    p2_min: Minimum p2 value to start the search.
    p2_max: Maximum p2 value to end the search.
    p2_step: The step size to increment p2 values.
    
    Returns:
    None. Prints the number of roots for each p2 value.
    """
    # Iterate over p2 values
    p2_values = np.arange(p2_min, p2_max + p2_step, p2_step)
    
    for p2 in p2_values:
        # Update the constant p2 in the constants object
        constants.p2 = p2

        # Adjust initial guesses based on the expected number of roots:
        if p2 < 5:
            # Expect fewer roots, use only 3 initial guesses
            initial_guesses = [-np.pi / 4, 0, np.pi / 4]
        elif p2 == 5:
            # Expect 4 roots around p2 = 5
            initial_guesses = [-np.pi / 2, -np.pi / 4, 0, np.pi / 4]
        elif p2 >= 7:
            # For p2 around 7, use the known set of initial guesses for 6 roots
            initial_guesses = [-.7, -.4, .01, .4, .9, 2.5]
        else:
            # General case for higher roots, use more guesses
            initial_guesses = [-np.pi / 2, -np.pi / 4, 0, np.pi / 4, np.pi / 2]
        
        # Find the roots for the current p2 value
        roots = find_roots(constants, initial_guesses)
        num_roots = len(roots)  # Get the number of roots found
        
        # Print the results
        print(f"p2 = {p2:.4f}, Number of Roots: {num_roots}, Roots: {roots}")

# Example constants (use your existing constants)
constants = Constants(
    l1=3, 
    l2=3 * np.sqrt(2), 
    l3=3, 
    gamma=np.pi / 4, 
    x1=5, 
    x2=0, 
    y2=6,  
    p1=5, 
    p2=None,  # To be set in the loop
    p3=3
)

# Define the range of p2 values and the step size
p2_min = 3.5  # Start around a lower value where fewer roots are expected
p2_max = 7.5  # Go past the value where you know there are 6 roots
p2_step = 0.001  # Use small increments to check for changes in the number of roots

# Call the function to find the intervals and dynamically adjust initial guesses
find_p2_intervals(constants, p2_min, p2_max, p2_step)


# %%
