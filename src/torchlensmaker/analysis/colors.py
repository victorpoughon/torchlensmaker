import colorcet as cc
import matplotlib as mpl

from typing import TypeAlias

# Color theme
color_valid = "#ffa724"
color_blocked = "red"
color_focal_point = "red"

# Default colormap*
LinearSegmentedColormap: TypeAlias = mpl.colors.LinearSegmentedColormap
default_colormap = cc.cm.CET_I2
