import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import splev, splprep


def main():
    # Assuming ctrl_points is already loaded
    ctrl_points = np.load("cortex_outer.npy")

    # Split ctrl_points into num_splines layers
    num_splines = np.unique(ctrl_points[:, 2]).size
    splines_points = np.array_split(ctrl_points, num_splines)

    # Function to blend two splines
    def blend_splines(tckp1, tckp2, blend_factor):
        # Evaluate splines at 100 points
        t_values = np.linspace(0, 1, 100)

        # Evaluate the splines at t_values
        eval_spline1 = splev(t_values, tckp1)
        eval_spline2 = splev(t_values, tckp2)

        # Blend the two splines
        blended_spline = (
            t_values,
            (1 - blend_factor) * np.array(eval_spline1[1])
            + blend_factor * np.array(eval_spline2[1]),
            1,  # Assuming linear splines, so degree k=1
        )

        return blended_spline

    # Interpolate each layer contained in splines_points into a B-spline
    splines = []
    blend_factor = 0.8  # You can adjust this factor for blending

    for i in range(1, len(splines_points)):
        spline_layer1 = splines_points[i - 1]
        spline_layer2 = splines_points[i]

        # Interpolate spline_layer1
        tckp1, _ = splprep(spline_layer1.T, k=1, s=0)

        # Interpolate spline_layer2
        tckp2, _ = splprep(spline_layer2.T, k=1, s=0)

        # Blend the two splines
        blended_spline = blend_splines(tckp1, tckp2, blend_factor)

        splines.append(blended_spline)

    # Plot the original control points without label or with a non-empty label
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    blended_xyz = []
    for spline in splines:
        xyz = splev(spline[0], spline)
        blended_xyz.append(xyz)

    blended_xyz = np.array(blended_xyz)
    blended_z = np.linspace(0, np.max(ctrl_points[:, 2]), blended_xyz.shape[0])

    # plot
    ax.scatter(
        blended_xyz[:, 0],
        blended_xyz[:, 1],
        blended_z,
        c="red",
        label="_nolegend_",
    )
    plt.show()


if __name__ == "__main__":
    main()
