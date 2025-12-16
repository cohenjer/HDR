
import numpy as np
import tensorly as tl
import matplotlib
import matplotlib.pyplot as plt

# Generating toy image
n1 = 30
n2 = 40
r = 3
sig = 0.2  # 0.15 # 0.05

# Components will be Gaussian distributions, here is the macro to generate them
def gauss(x, m, sig, thresh=0.1):
    # max is 1
    out = np.exp(-(x-m)**2/sig**2)
    out[out < thresh] = 0
    return out

# Generating the data as a mixture of separable Gaussians
x = np.linspace(0, n1-1, n1) # 1D abcisse
y = np.linspace(0, n2-1, n2)
W = np.zeros([n1, r])
W[:,0] = gauss(x, 5, 3)
W[:,1] = gauss(x, 15, 5)
W[:,2] = gauss(x, 25, 10)
H = np.zeros([n2, r])
H[:, 0] = gauss(y, 10, 5)
H[:, 1] = gauss(y, 20, 6)
H[:, 2] = gauss(y, 25, 5)
trueNMF = (None, [W, H])

Y = W@H.T  # NMF model
rng = np.random.default_rng(1246)
Yn = Y + sig*rng.random((n1, n2))  # corruption with gaussian noise

# initialization 
W0 = 0*W + 0.5*rng.random(W.shape)
H0 = 0*H + 0.5*rng.random(H.shape)

font = {'size'   : 16}
matplotlib.rc('font', **font)

# Storing output factors for various regularization in dictionaries
We = dict()
He = dict()

# Chose the grid values for the hyperparameter 
lambset = [0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.7, 0.75, 0.78, 0.8, 1, 1.2, 1.5, 2, 3, 4, 5]

import copy
for lamb in lambset:
    # Computing the sparse NMF with Tensorly (the algorithm is HALS, which is state of the art for this problem)
    out = tl.decomposition.non_negative_parafac_hals(Yn, r, sparsity_coefficients=[lamb, lamb], init=copy.deepcopy((None, [W0, H0])))
    # Estimated factors
    We[lamb] = out[1][0]
    He[lamb] = out[1][1]
    # Set estimates as new initialization for the next iteration (warm start)
    W0 = We[lamb]
    H0 = He[lamb]
    
    # rescale so that when a column of W is 0, so is the corresponding column of H
    norms = tl.max(We[lamb], axis=0)
    for q in range(r):
        if norms[q] > 1e-8:
            We[lamb][:, q] = We[lamb][:, q]/norms[q]
            He[lamb][:, q] = He[lamb][:, q]*norms[q]
        else:
            We[lamb][:, q] = We[lamb][:, q]*0
            He[lamb][:, q] = He[lamb][:, q]*0



from bokeh.plotting import figure, show, output_notebook
from bokeh.models import ColumnDataSource, Slider, CustomJS
from bokeh.layouts import column
import numpy as np

output_notebook()

# ----- Sample Data -----
r = 3
x = np.linspace(0, 10, 200)
lambset = [0, 1, 2, 3, 4]

# Create a 3D array: shape (len(lambset), len(x), r)
We_data = np.stack([np.array([np.sin((i+1)*x/(lam+1)) for i in range(r)]).T for lam in lambset])

# Initial data for lambda = 0
initial_index = 0
sources = [ColumnDataSource(data=dict(x=x, y=We_data[initial_index, :, i])) for i in range(r)]

# Figure setup
p = figure(title=f"Regularization = {lambset[initial_index]}", width=700, height=400,
           x_axis_label="X", y_axis_label="Y")

lines = [p.line('x', 'y', source=sources[i], line_width=2, legend_label=f"Line {i+1}")
         for i in range(r)]

# Flatten the data to send to JavaScript
# Format: We_data_js[lambda index][line index][y values]
We_data_flat = [[[float(We_data[lam_idx, i, j]) for i in range(len(x))] for j in range(r)]
                for lam_idx in range(len(lambset))]

# JavaScript callback for the slider
callback = CustomJS(args=dict(sources=sources,
                              all_data=We_data_flat,
                              title=p.title,
                              lambset=lambset),
    code="""
    const lambda_index = cb_obj.value;
    for (let i = 0; i < sources.length; i++) {
        sources[i].data['y'] = all_data[lambda_index][i];
        sources[i].change.emit();
    }
    title.text = "Regularization = " + lambset[lambda_index];
""")

# Slider widget
slider = Slider(start=0, end=len(lambset)-1, value=0, step=1, title="Regularization")
slider.js_on_change('value', callback)

# Show layout
show(column(p, slider))

# --------------
## Plotly version
#import plotly.graph_objects as go
#fig = go.Figure()

## Add traces for the first regularization value (lambset[0])
#for i in range(r):
    #fig.add_trace(go.Scatter(
        #x=x,
        #y=We[0][:, i],  # Data for the first lambda value
        #mode='lines',
        #name=f'Line {i+1}'
    #))

## Create steps for the slider
#steps = []
#for lam in lambset:
    #step = dict(
        #method='update',
        #args=[{'y': [We[lam][:, i] for i in range(r)]},
              #{'title': f"Regularization = {lam}"}],
        #label=str(lam)
    #)
    #steps.append(step)

## Add slider to the layout
#sliders = [dict(
    #active=0,  # Start with the first regularization value
    #currentvalue={"prefix": "Regularization: "},
    #pad={"t": 50},
    #steps=steps
#)]

## Update layout with slider and labels
#fig.update_layout(
    #sliders=sliders,
    #title="Interactive Regularization Plot",
    #xaxis_title="X",
    #yaxis_title="Y"
#)

## Render the plot in the browser (or Jupyter if you're using it)
#fig.show()

# ---------------
## Matplotlib version

#from matplotlib.widgets import Slider
#fig, ax = plt.subplots()
#fig.subplots_adjust(bottom=0.25)
#plots = ax.plot(We[0])  # list of three line

##?
#ax_reg = fig.add_axes([0.25, 0.1, 0.65, 0.03])

## create the sliders
#samp = Slider(
    #ax=ax_reg,
    #label="Regularization",
    #valmin=0,
    #valmax=5,
    #valinit=0,
    #orientation="horizontal",
    #valstep=lambset,
    #color="green"
#)

#def update(val):
    #lamb = samp.val
    #for i in range(r):
        #plots[i].set_ydata(We[lamb][:, i])
    #fig.canvas.draw_idle()


#samp.on_changed(update)  #runs code update with input the slider value when the slider is moved

#plt.show()
