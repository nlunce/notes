import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import quad
from functools import lru_cache
import time

class ParametricCurve:
    def __init__(self, x_func, y_func, dx_dt_func, dy_dt_func, t_min=0, t_max=1):
        self.x = x_func
        self.y = y_func
        self.dx_dt = dx_dt_func
        self.dy_dt = dy_dt_func
        self.t_min = t_min
        self.t_max = t_max

    def integrand(self, t):
        return np.sqrt(self.dx_dt(t)**2 + self.dy_dt(t)**2)

    @lru_cache(maxsize=None)
    def compute_arc_length(self, s):
        arc_length, _ = quad(self.integrand, self.t_min, s)
        return arc_length

    def total_length(self):
        return self.compute_arc_length(self.t_max)

    def newton_find_t(self, target_length, initial_guess=None, tol=1e-8, max_iter=100):
        if initial_guess is None:
            initial_guess = (self.t_min + self.t_max) / 2
        t = initial_guess
        for _ in range(max_iter):
            f_t = self.compute_arc_length(t) - target_length
            f_prime_t = self.integrand(t)
            if abs(f_t) < tol:
                return t
            if f_prime_t == 0:
                raise ZeroDivisionError("Derivative zero encountered in Newton's method.")
            t -= f_t / f_prime_t
            t = max(self.t_min, min(self.t_max, t))
        raise RuntimeError("Newton's method did not converge.")

    def bisection_find_t(self, target_length, a=None, b=None, tol=1e-8, max_iter=100):
        if a is None:
            a = self.t_min
        if b is None:
            b = self.t_max
        fa = self.compute_arc_length(a) - target_length
        fb = self.compute_arc_length(b) - target_length
        if fa * fb > 0:
            raise ValueError("Bisection method fails: function has same signs at interval endpoints.")
        for _ in range(max_iter):
            t = (a + b) / 2
            f_t = self.compute_arc_length(t) - target_length
            if abs(f_t) < tol:
                return t
            if fa * f_t < 0:
                b = t
                fb = f_t
            else:
                a = t
                fa = f_t
        return t

    def equipartition(self, n, method='newton'):
        total_length = self.total_length()
        segment_length = total_length / n
        partition_points = [self.t_min]
        for i in range(1, n):
            target_length = i * segment_length
            initial_guess = partition_points[-1]
            if method == 'newton':
                t_i = self.newton_find_t(target_length, initial_guess)
            elif method == 'bisection':
                t_i = self.bisection_find_t(target_length, a=initial_guess)
            else:
                raise ValueError("Method must be 'newton' or 'bisection'.")
            partition_points.append(t_i)
        partition_points.append(self.t_max)
        return partition_points
    
    
# Define the first curve
def x1(t):
    return 0.5 + 0.3 * t + 3.9 * t**2 - 4.7 * t**3

def y1(t):
    return 1.5 + 0.3 * t + 0.9 * t**2 - 2.7 * t**3

def dx1_dt(t):
    return 0.3 + 2 * 3.9 * t - 3 * 4.7 * t**2

def dy1_dt(t):
    return 0.3 + 2 * 0.9 * t - 3 * 2.7 * t**2

# Create an instance of ParametricCurve for the first curve
curve1 = ParametricCurve(x1, y1, dx1_dt, dy1_dt, t_min=0, t_max=1)

# Define the second curve
A = 0.4
a = 3
f = np.pi / 2
c = 0.5
B = 0.3
b = 4
D = 0.5
t_max2 = 2 * np.pi

def x2(t):
    return A * np.sin(a * t + f) + c

def y2(t):
    return B * np.sin(b * t) + D

def dx2_dt(t):
    return A * a * np.cos(a * t + f)

def dy2_dt(t):
    return B * b * np.cos(b * t)

# Create an instance of ParametricCurve for the second curve
curve2 = ParametricCurve(x2, y2, dx2_dt, dy2_dt, t_min=0, t_max=t_max2)

def style_plot(ax, x_limits, y_limits, adjust_tick_labels=True):

    ax.grid(True, linestyle='-', alpha=0.2, color='gray')
    ax.set_xticks(np.arange(x_limits[0], x_limits[1] + 0.5, 0.5))
    ax.set_yticks(np.arange(y_limits[0], y_limits[1] + 0.5, 0.5))

    if adjust_tick_labels:
        ax.set_xticklabels(['' if x == 0 else f"{x:.1f}" for x in ax.get_xticks()])
        ax.set_yticklabels(['' if y == 0 else f"{y:.1f}" for y in ax.get_yticks()])

    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

    ax.tick_params(axis='both', which='major', length=6, width=1, colors='black', direction='out')
    ax.tick_params(axis='both', which='minor', length=3, width=1, colors='black', direction='out')

    ax.set_xlim(x_limits)
    ax.set_ylim(y_limits)
    ax.set_aspect('equal')

def plot_curve(curve, n_points=500, n_equipartition=None, title='Parametric Curve', x_limits=(-1.5, 1.5), y_limits=(-0.5, 2)):
    t_vals = np.linspace(curve.t_min, curve.t_max, n_points)
    x_vals = curve.x(t_vals)
    y_vals = curve.y(t_vals)

    plt.figure(figsize=(8, 8), facecolor='white')
    ax = plt.gca()

    plt.plot(x_vals, y_vals, color="#2196F3", linewidth=2.5, zorder=3)

    # Plot equipartition points if requested
    if n_equipartition is not None:
        partition_points = curve.equipartition(n_equipartition)
        x_points = [curve.x(t) for t in partition_points]
        y_points = [curve.y(t) for t in partition_points]
        plt.scatter(x_points, y_points, color="#1565C0", s=40, zorder=4)

    style_plot(ax, x_limits, y_limits)
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.show()

def animate_curve(curve, n_frames=100, filename='animation.mp4', x_limits=(-1.5, 1.5), y_limits=(-0.5, 2), xticks=None, yticks=None):
    # Original speed parameterization
    t_values_original = np.linspace(curve.t_min, curve.t_max, n_frames)
    x_original = curve.x(t_values_original)
    y_original = curve.y(t_values_original)

    # Constant speed parameterization
    t_values_constant = curve.equipartition(n_frames)
    x_constant = [curve.x(t) for t in t_values_constant]
    y_constant = [curve.y(t) for t in t_values_constant]

    # Set up the figure
    fig = plt.figure(figsize=(16, 8), facecolor='white')
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    # Plot the curves
    ax1.plot(x_original, y_original, color="#2196F3", linewidth=2.5, zorder=3)
    ax2.plot(x_constant, y_constant, color="#2196F3", linewidth=2.5, zorder=3)

    # Initialize the points
    point1, = ax1.plot([], [], 'o', color="#1565C0", markersize=8, zorder=4)
    point2, = ax2.plot([], [], 'go', markersize=8, zorder=4)

    style_plot(ax1, x_limits, y_limits, adjust_tick_labels=True)
    style_plot(ax2, x_limits, y_limits, adjust_tick_labels=True)

    if xticks is not None:
        ax1.set_xticks(xticks)
        ax2.set_xticks(xticks)
        ax1.set_xticklabels([f"{x:.1f}" for x in xticks])
        ax2.set_xticklabels([f"{x:.1f}" for x in xticks])
    if yticks is not None:
        ax1.set_yticks(yticks)
        ax2.set_yticks(yticks)
        ax1.set_yticklabels([f"{y:.1f}" for y in yticks])
        ax2.set_yticklabels([f"{y:.1f}" for y in yticks])

    ax1.set_title("Original Speed", fontsize=12, fontweight='bold', pad=15)
    ax2.set_title("Constant Speed", fontsize=12, fontweight='bold', pad=15)
    plt.tight_layout()

    # Update functions
    def update(frame):
        # Update original speed point
        point1.set_data(x_original[:frame], y_original[:frame])
        # Update constant speed point
        point2.set_data(x_constant[:frame], y_constant[:frame])
        return point1, point2

    ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=200, blit=True)
    # Save the animation
    ani.save(filename, writer='ffmpeg')
    plt.show()

def compare_performance(curve, target_length):
    start_time_bisection = time.time()
    bisection_result = curve.bisection_find_t(target_length)
    bisection_time = time.time() - start_time_bisection

    start_time_newton = time.time()
    newton_result = curve.newton_find_t(target_length)
    newton_time = time.time() - start_time_newton

    print(f"Bisection Method Result: {bisection_result:.9f} Time: {bisection_time:.9f} seconds")
    print(f"Newton's Method Result: {newton_result:.9f} Time: {newton_time:.9f} seconds")

# Compute total arc length of the first curve
total_length1 = curve1.total_length()
print(f"Arc Length from t=0 to t=1: {total_length1:.6f}")

# Compare performance of root-finding methods on the first curve
print("\nComparing Performance of Bisection and Newton's Methods:")
compare_performance(curve1, total_length1 / 2)

# Plot the first curve with equipartition points (n=4 and n=20)
plot_curve(curve1, n_points=500, n_equipartition=4, title='Curve with 4 Equipartition Points')
plot_curve(curve1, n_points=500, n_equipartition=20, title='Curve with 20 Equipartition Points')

# Animate the first curve
animate_curve(curve1, n_frames=25, filename='combined_animation.mp4')

# Adjusted ticks
xticks = np.arange(0, 1.1, 0.2)
yticks = np.arange(0, 1.1, 0.2)

# Animate the second curve
animate_curve(curve2, n_frames=200, filename='custom_path_animation.mp4', x_limits=(0, 1), y_limits=(0, 1), xticks=xticks, yticks=yticks)
