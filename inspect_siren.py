from matplotlib.cm import get_cmap
from PIL import Image
from flask import Flask
from flask import send_file, request, current_app
from io import BytesIO
import argparse
import torch
import numpy as np
import re
import threading
from siren import Siren
from skimage import color
from sklearn.decomposition import PCA
import umap

parser = argparse.ArgumentParser(
    description="Spins up a server which serves a webpage that allows us to inspect the activations of a SIREN network."
)
parser.add_argument("siren_path", type=str, help="Path to a siren state dict .pt file")
parser.add_argument(
    "--w0_initial",
    type=float,
    default=30,
    help="w0_initial parameter for the siren net, which cannot be recovered from the state dict",
)
parser.add_argument(
    "--img_width", type=int, default=768, help="Width of the output image"
)
parser.add_argument(
    "--img_height", type=int, default=512, help="height of the output image"
)

args = parser.parse_args()

#####################################################################
## Create the SIREN network
#####################################################################
state_dict = torch.load(args.siren_path)
# Get the num layers
layer_weight_re = re.compile(r"net\.(\d+)\.linear\.weight")
max_layer_index = -1
for key in state_dict.keys():
    match = layer_weight_re.match(key)
    if match:
        layer_index = int(match.group(1))
        max_layer_index = max(max_layer_index, layer_index)

num_layers = max_layer_index + 1
# get the dim_hidden
dim_hidden = state_dict["net.0.linear.weight"].shape[0]

# make the siren network
siren = Siren(
    dim_in=2,
    dim_hidden=dim_hidden,
    dim_out=3,
    num_layers=num_layers,
    w0_initial=args.w0_initial,
)

siren.load_state_dict(state_dict)

#####################################################################
## Create input coordinates
#####################################################################
coordinates = (
    torch.ones((args.img_height, args.img_width)).nonzero(as_tuple=False).float()
)
# Normalize coordinates to lie in [-.5, .5]
coordinates = coordinates / (args.img_width - 1) - 0.5
# Convert to range [-1, 1]
coordinates *= 2
coordinates = coordinates.reshape(args.img_height, args.img_width, 2)

#####################################################################
## Globals and utility functions
#####################################################################

thumbnail_size = (96, 64)  # width, height

cmap = get_cmap("coolwarm")

last_used_activations = None
last_siren = None
last_layer = None
last_coords = None
semaphore = threading.Semaphore()


def memoized_get_activations_from(siren, layer, coords):
    """ "Memoized" wrapper around siren.get_activations_from().

    Saves the last result and returns it if we're doing the same query again"""
    global last_used_activations, last_siren, last_layer, last_coords
    with semaphore:
        same_as_last_call = (
            siren is last_siren
            and layer == last_layer
            and torch.all(coords == last_coords)
        )
        if not same_as_last_call:
            last_used_activations = siren.get_activations_from(layer, coords)
            last_siren = siren
            last_layer = layer
            last_coords = coords
        return last_used_activations


def fast_umap(activations):
    # avoid performing umap across all the data. Might be faster??
    # instead, perform umap on some sample of the data,
    # and extrapolate where the rest of the points might be placed.
    # take a sampling of the activations
    # perform umap on that sampling
    # express the other points as linear combos of the umapped points somehow
    # for each point:
    # take its dot product with each umapped point
    # all_points: (n, 28)
    # sampled_points: (m, 28)
    # sampled_points_colors: (m, 3)
    # all_points_dot_products_with_samples (n, m) = all_points @ sampled_points.T
    # take softmax of this matrix: (n, m)
    # all_points_colors (n, 3): softmax_similarity_to_sampled_points @ sampled_points_colors
    # take the softmax of this dot product
    # set point's color to a linear combo of the umapped points' colors
    # then their color can be interpolated according to umapped points
    pass


def create_layer_map(siren, coords, layer, size=None):
    height, width, channels = coords.shape
    assert channels == 2
    coords = coords.reshape(-1, 2)
    activations = memoized_get_activations_from(siren, layer, coords)
    activations = activations.detach().cpu().numpy()
    acts = PCA(n_components=3).fit_transform(activations)
    acts = acts - acts.min(axis=0, keepdims=True)
    acts = acts / acts.max(axis=0, keepdims=True)
    acts_img = (acts.reshape(height, width, 3) * 255).astype("uint8")
    # TODO transform coords from LAB to RGB color space

    pil_map = Image.fromarray(acts_img)
    if size is not None:
        pil_map = pil_map.resize(size, resample=Image.BOX)
    return pil_map


def create_activation_map(siren, coords, layer, feature, size=None):
    """returns a visualization of the activation map of this feature.

    args:
            siren: the SIREN network to inspect
            coords: input coordinates. of shape (h, w, 2)
            layer: which layer we're visualizing
            feature: the # of the feature in that layer
            size: size of the output image (width, height)
    """
    # TODO: handle visualizations of the bias, this is the last feature.
    height, width, channels = coords.shape
    assert channels == 2
    coords = coords.reshape(-1, 2)
    activations = memoized_get_activations_from(siren, layer, coords)
    if feature == activations.shape[1]:
        activation = torch.ones_like(activations[:, 0])
    else:
        activation = activations[:, feature]
    activation = activation.reshape(height, width).detach().cpu().numpy()
    colorized = None
    is_last_layer = siren.num_layers == layer
    if is_last_layer:
        assert feature < 3  # feature should be one of r,g,b channels
        colorized = np.zeros((height, width, 3), dtype=activation.dtype)
        colorized[:, :, feature] = np.clip(activation, 0, 1)
    else:
        normed_activation = (activation + 1) / 2
        colorized = cmap(normed_activation)[:, :, :3]  # shape (height, width, 3)
    pil_map = Image.fromarray((colorized * 255).astype("uint8"))
    if size is not None:
        pil_map = pil_map.resize(size, resample=Image.BOX)
    return pil_map


def to_py_list(tensor):
    return tensor.detach().cpu().numpy().astype(float).tolist()


def get_weight_data(siren):
    """returns a jsonifiable representation of the siren's weights and biases,
    to be passed to the webpage.
    """
    # TODO: drop biases, incorporate into weights. Last "feature" in each layer is the bias
    weight_data = []  # list of weight matrices

    for index, layer in enumerate(list(siren.net) + [siren.last_layer]):
        weights = layer.linear.weight  # shape: (dim_out, dim_in)
        biases = layer.linear.bias  # shape: (dim_out)
        weights = torch.concat(
            [weights, biases[:, None]], dim=1
        )  # shape: (dim_out, dim_in+1)
        # if this isn't the penultimate layer,
        # add a row of zeros so the weights are now of shape (dim_out+1, dim_in+1)
        # these represent the "weights" that go into the "bias feature" in the next layer
        if index < len(siren.net):
            fake_bias_weights = torch.zeros((1, weights.shape[1]))
            weights = torch.concat([weights, fake_bias_weights], dim=0)

        weight_data.append(to_py_list(weights))

    return {"weights": weight_data}


#####################################################################
## Flask server
#####################################################################


def serve_pil_image(pil_img):
    img_io = BytesIO()
    pil_img.save(img_io, "PNG")
    img_io.seek(0)
    return send_file(img_io, mimetype="image/jpeg")


app = Flask(__name__)


@app.route("/")
def main_page():
    return current_app.send_static_file("inspect.html")


@app.route("/weight_data")
def serve_weight_data():
    return get_weight_data(siren)


@app.route("/activation_map")
def get_activation_map():
    layer = int(request.args.get("layer"))
    feature = int(request.args.get("feature"))
    thumbnail = int(request.args.get("thumbnail"))
    size = thumbnail_size if thumbnail else None
    activation_map = create_activation_map(siren, coordinates, layer, feature, size)
    return serve_pil_image(activation_map)


@app.route("/layer_map")
def get_layer_map():
    layer = int(request.args.get("layer"))
    thumbnail = int(request.args.get("thumbnail"))
    size = thumbnail_size if thumbnail else None
    layer_map = create_layer_map(siren, coordinates, layer, size)
    return serve_pil_image(layer_map)


app.run(host="localhost", port=5214, debug=True)
