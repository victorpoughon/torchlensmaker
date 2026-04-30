# This file is part of Torch Lens Maker
# Copyright (C) 2024-present Victor Poughon
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from matplotlib.axes import Axes
from matplotlib.patches import Circle


def plot_airy_disk(
    ax: Axes,
    wavelength: float,
    aperture: float,
    center: tuple[float, float],
) -> None:
    """Draw the Airy disk radius as a dashed circle on a spot diagram Axes.

    The Airy disk radius (first dark ring of the diffraction pattern) is:

        r = 1.22 * wavelength / (2 * NA)

    where NA is the numerical aperture in image space.

    Args:
        ax: matplotlib Axes to draw on (must be a spot diagram cell)
        wavelength: wavelength of light, in the same length units as the plot
        aperture: numerical aperture (NA) in image space
        center: (y, z) center of the circle in plot coordinates — pass the
                cell center returned by the spot diagram, or (0, 0) for
                an on-axis field point
    """
    r = 1.22 * wavelength / (2 * aperture)
    cy, cz = center
    # Plot axes are (z, y), so the circle center is (cz, cy)
    circle = Circle((cz, cy), r, fill=False, linestyle="--", color="white", linewidth=1)
    ax.add_patch(circle)
