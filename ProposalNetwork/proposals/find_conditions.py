import sys
import numpy as np
import matplotlib.pyplot as plt

# To find txt files , needs to be run in proposal before final center point is found
"""
with open("conditions.txt", "a") as f:
    # Redirect the standard output to the file
    sys.stdout = f
    # Your print statements here
    print(f"{int(y_range_px[1].item())},{((gt_cube.center[1].item()-torch.min(y).item())/y_width.item()):.2f}")
# Reset the standard output
sys.stdout = sys.__stdout__
"""

# Conditions
x_cond = []
y_cond = []

# Read data from the file
with open("conditions_z.txt", "r") as file:
    for line in file:
        # Split each line by comma
        parts = line.strip().split(",")
        # Extract x and y values
        x_val = float(parts[0].strip())
        y_val = float(parts[1].strip())
        # Append x and y values to the arrays
        x_cond.append(x_val)
        y_cond.append(y_val)


# System of equations in matrix form
A = np.array([[x, 1] for x in x_cond])


# Solve for coefficients
coefficients, _, _, _ = np.linalg.lstsq(A, y_cond, rcond=None)
print(coefficients)

# Reset the standard output
sys.stdout = sys.__stdout__

plt.figure()
plt.scatter(x_cond,y_cond)
plt.savefig("z_values_to_find.png", dpi=300, bbox_inches='tight')
plt.close()


