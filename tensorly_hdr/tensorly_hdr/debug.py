from bokeh.layouts import column, row
from bokeh.plotting import figure, show, output_notebook
from bokeh.models import Slider, ColumnDataSource, CustomJS
from bokeh.palettes import Category10

#temp
from bokeh.plotting import figure, show
from bokeh.palettes import Viridis256
import random
import numpy as np

# Tensor dimensions
xp_size = 10
time_size = 15
mz_size = 5

# Build 3D tensor [xp][time][mz]
tensor = [[[random.random() for _ in range(mz_size)] 
            for _ in range(time_size)] 
            for _ in range(xp_size)]

tensor_full = np.array(tensor)  # Convert to NumPy array for easier manipulation

# Convert to numpy array (optional but safe)
image_array = [np.array(tensor[:][:][mz_index]) for mz_index in range(mz_size)]

source = ColumnDataSource(data=dict(
    image=[image_array[0]] # Flatten the first slice for initial display
))

# Create figure
p = figure(
    title="Elution profile image",
    x_range=(0, xp_size),
    y_range=(0, time_size),
    width=400,
    height=300,
    x_axis_label="Experiment index",
    y_axis_label="Time index"
)

# Draw image (NumPy or pure list both acceptable)
p.image(image='image', x=0, y=0, dw=xp_size, dh=time_size, palette=Viridis256, source=source)

# Slider to control mz_index
slider = Slider(start=0, end=mz_size - 1, value=0, step=1, title="mz_index")

# CustomJS callback — inject the tensor directly as a JS array
callback = CustomJS(args=dict(source=source, im_array=image_array, slider=slider), code=f"""
    const mz_index = slider.value;

    //source.data.image[0] = im_array[mz_index];  // Update the image data with the selected mz_index
    source.data.image[0] = tensor[mz_index];  // Update the image data with the selected mz_index
    source.change.emit();
""")

slider.js_on_change('value', callback)

# Display everything
show(column(p, slider))
