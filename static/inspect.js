/* TODO: load the weights data */
/* get num layers */
/* get num hidden features */
/*
make a node list and an edge list
bind these data to d3 selections of some kind

Each node has
    an x, y position, 
    a layer index
    a feature index

Each link has
    src_feature_index
    dest_feature_index
    dest_layer
*/

function get_nodes_and_links_from(data) {
    let nodes = [];
    let links = [];
    let max_feats_per_layer = 0;
    for (let layer_index in data.weights) {
        layer_index = parseInt(layer_index)
        let layer_weights = data.weights[layer_index];
        for (let feature_index in layer_weights) {
            feature_index = parseInt(feature_index)
            let node = {
                'x': layer_index,
                'y': feature_index,
                'layer': layer_index,
                'feature': feature_index,
                'selected': false
            };
            nodes.push(node);
            if (layer_index > 0) {
                let input_weights = layer_weights[feature_index];
                /* Get the maximum absolute value of a weight */
                let max_magnitude = 0;
                for (let src_feature_index in input_weights) {
                    let weight = input_weights[src_feature_index];
                    let magnitude = Math.abs(weight)
                    if (magnitude > max_magnitude) {
                        max_magnitude = magnitude
                    }
                }

                for (let src_feature_index in input_weights) {
                    let weight = input_weights[src_feature_index];
                    let link = {
                        "src_feature": parseInt(src_feature_index),
                        "dest_feature": parseInt(feature_index),
                        "dest_layer": parseInt(layer_index),
                        "weight": parseFloat(weight),
                        "normalized_magnitude": Math.abs(weight) / max_magnitude
                    };
                    links.push(link);
                }
            }
            if (layer_weights.length > max_feats_per_layer) {
                max_feats_per_layer = layer_weights.length
            }
        }
    }
    
    return {
        "nodes": nodes,
        "links": links,
        "num_layers": data.weights.length,
        "max_feats": max_feats_per_layer
    };
}

let node_width = 96;
let node_height = 64;
let node_space_width=192;
let node_space_height=70;


function get_layer_map_thumbnail_url(layer) {
    return "layer_map?layer=" + layer + "&thumbnail=1";
}

function get_layer_map_full_url(layer) {
    return "layer_map?layer=" + layer + "&thumbnail=0";
}

function get_thumnail_url(node) {
    return "activation_map?layer=" + node.layer + "&feature=" + node.feature + "&thumbnail=1";
}

function get_full_map_url(node) {
    return "activation_map?layer=" + node.layer + "&feature=" + node.feature + "&thumbnail=0";
}


WEIGHT_SPACE = 20

function link_x1(target_layer){
    return function(link) {
        let weight_space = link.dest_layer == target_layer ? WEIGHT_SPACE : 0;
        return (link.dest_layer - 1) * node_space_width + node_width + weight_space
    }
}
function link_x2(target_layer){
    return function(link) {
        let weight_space = link.dest_layer == target_layer+1 ? WEIGHT_SPACE : 0;
        return (link.dest_layer) * node_space_width - weight_space
    }
}

function link_y1(nodes) {
    return function(link) {
        let src_node = null;
        for (let i in nodes){
            let node = nodes[i];
            if (node.layer == link.dest_layer-1 && node.feature == link.src_feature){
                src_node = node;
                break;
            }
        }
        /* Use that source node's y value instead of src_feature */
        return (src_node.y + 1) * node_space_height + node_height/2
    }
}
function link_y2(nodes) {
    return function(link) {
        let dest_node = null;
        for (let i in nodes){
            let node = nodes[i];
            if (node.layer == link.dest_layer && node.feature == link.dest_feature){
                dest_node = node;
                break;
            }
        }
        return (dest_node.y+1) * node_space_height + node_height/2
    }
}


function link_color(link) {
    return link.weight > 0 ? 'blue' : 'crimson';
}

function link_width(link) {
    return 4 * link.normalized_magnitude;
}

function link_opacity(link) {
    let x = 2 * link.normalized_magnitude;
    return x > 1 ? 1 : x
}

function get_input_weighted_magnitudes(target_node, links) {
    let weight_magnitudes = [];
    let weight_features = [];
    for (let link_index in links) {
        let link = links[link_index];
        let from_correct_layer = target_node.layer == link.dest_layer;
        let to_correct_feature = link.dest_feature == target_node.feature;
        let connected_to_target_node = from_correct_layer && to_correct_feature;
        if (connected_to_target_node) {
            weight_magnitudes.push(Math.abs(link.weight));
            weight_features.push(link.src_feature);
        }
    }
    return permute(weight_magnitudes, weight_features);
}

function get_output_weighted_magnitudes(target_node, links) {
    let weight_magnitudes = [];
    let weight_features = [];
    for (let link_index in links) {
        let link = links[link_index];
        let from_correct_layer = target_node.layer + 1 == link.dest_layer;
        let to_correct_feature = link.src_feature == target_node.feature;
        let connected_to_target_node = from_correct_layer && to_correct_feature;
        if (connected_to_target_node) {
            weight_magnitudes.push(Math.abs(link.weight));
            weight_features.push(link.dest_feature);
        }
    }
    return permute(weight_magnitudes, weight_features);
}


function get_best_ordering(target_node, magnitudes){
    /* returns a permutation of the magnitudes that minimizes the weighted distance from the target node. */
    /* 
    For a single layer, this should be easy.
     First, just get a permutation that sorts by magnitude
     then, get the permutation that sorts by distance from the target
     apply the distance-sirting permutation to the magnitude-sorting permutation
     return that.
     Permutations are associative, so this works.
    */
    let magnitude_sorting_permutation = argsort(magnitudes)
    let distance_sorting_permutation = order_by_distance_from_index(target_node.y, magnitudes.length)
    return permute(magnitude_sorting_permutation, distance_sorting_permutation)
}
function order_by_distance_from_index(target_position, length){
    /* returns a permutation '0' is closest to the target position, '1' is next closest, etc.
    */
    permutation = []
    for (let i=0; i<length; i++) {
        permutation.push(-1);
    }
    let displacement = 0;
    let next_index = 0;
    while (next_index < length) {
        let position = target_position + displacement
        if (0 <= position && position < length) {
            permutation[position] = next_index;
            next_index += 1;
        }
        if (displacement < 0){
            displacement = -displacement
        } else {
            displacement = -displacement - 1
        }
    }
    return permutation
}

function apply_permutation_to_layer(nodes, layer, permutation) {
    let inverse_permutation = invert_permutation(permutation);
    for (let node_index in nodes) {
        let node = nodes[node_index];
        if (node.layer == layer) {
            node.y = inverse_permutation[node.feature];
        }
    }
}

function argsort(list) {
    /* returns a permutation that sorts the list. */
    return list.map((v,i)=>[v,i]).sort().reverse().map(i=>i[1]);
}

function permute(list, permutation){
    /* Returns a list permuted by the provided permutation. */
    let permuted_list = [];
    for (let i in permutation) {
        let index = permutation[i];
        permuted_list.push(list[index]);
    }
    return permuted_list;
}

function invert_permutation(permutation) {
    /* Returns the inverse permutation */
    inverse = []
    for (let i=0; i<permutation.length; i++) {
        inverse.push(-1);
    }
    for (let index=0; index<permutation.length; index++) {
        let value = permutation[index];
        inverse[value] = index;
    }
    return inverse;
}

function select_node(target_node, nodes, links, num_layers) {
    /* sets the particular node in nodes to selected, and deselects all other nodes */
    for (let i in nodes) {
        let node = nodes[i];
        if (node.layer == target_node.layer && node.feature == target_node.feature) {
            /* If the node is already selected, open a tab with the larger image */
            if (node.selected) {
                let url = get_full_map_url(target_node);
                window.open(url, "_blank");
            }
            node.selected = true;
        } else {
            node.selected = false;
        }
    }

    /* rearrange the features connected to this one so the most influential
    features are the closest. */
    if (target_node.layer != 0) {
        let link_magnitudes = get_input_weighted_magnitudes(target_node, links);
        let permutation = get_best_ordering(target_node, link_magnitudes);
        apply_permutation_to_layer(nodes, target_node.layer-1, permutation);
    }

    if (target_node.layer < num_layers-1) {
        let link_magnitudes = get_output_weighted_magnitudes(target_node, links);
        console.log("link_magnitudes::")
        console.log(link_magnitudes)
        let permutation = get_best_ordering(target_node, link_magnitudes);
        console.log("Permutation::")
        console.log(permutation)
        apply_permutation_to_layer(nodes, target_node.layer+1, permutation);
        console.log("Applied permutation to layer!")
    }
    /* update the y positions of the feature images to match the new order */
    d3.select('svg')
        .selectAll('image.activation_map')
        .transition()
        .attr('y', node => (node.y + 1) * node_space_height);
    console.log("Applied the transition!!")

    d3.selectAll("line").remove();
    d3.selectAll("text").remove();
    let active_links = [];
    for (let i in links) {
        let link = links[i];
        let links_to_dest = link.dest_layer == target_node.layer && link.dest_feature == target_node.feature;
        let links_to_src = link.dest_layer - 1 == target_node.layer && link.src_feature == target_node.feature;
        if (links_to_src || links_to_dest) {
            active_links.push(link)
        }
    }

    let line = d3.select('svg').selectAll("line").data(active_links);
    let lineEnter = line.enter().append("line");

    lineEnter.attr('x1', link_x1(target_node.layer))
    lineEnter.attr('y1', link_y1(nodes))
    lineEnter.attr('x2', link_x2(target_node.layer))
    lineEnter.attr('y2', link_y2(nodes))
    lineEnter.attr('stroke', link_color)
    lineEnter.attr('stroke-width', link_width)
    lineEnter.attr('stroke-opacity', link_opacity)
    /* TODO: display the weight value to two sig figs */
    /* select text */

    function weight_text_x(link) {
        if (link.dest_layer == target_node.layer) {
            return link_x1(nodes)(link);
        } else {
            return link_x2(nodes)(link) - WEIGHT_SPACE;
        }
    }

    function weight_text_y(link) {
        if (link.dest_layer == target_node.layer) {
            return link_y1(nodes)(link);
        } else {
            return link_y2(nodes)(link);
        }
    }

    let text = d3.select('svg')
        .selectAll('text')
        .data(active_links)
        .enter()
        .append('text')
        .attr('x', weight_text_x)
        .attr('y', weight_text_y)
        .text(link => link.weight.toPrecision(2).toString())
        .style('font', '10px sans-serif')
        .attr('fill', 'white');
}

function init_network(data) {
    let parsed_data = get_nodes_and_links_from(data);
    let nodes = parsed_data.nodes;
    let links = parsed_data.links;

    let svg = d3.select('svg');
    svg.attr('width', parsed_data.num_layers * node_space_width)
    svg.attr('height', (parsed_data.max_feats+1) * node_space_height)

    /* TODO: select all image.layer_map and add layer maps along the top */
    layers = []
    for (let layer=0; layer < parsed_data.num_layers; layer += 1) {
        layers.push(layer)
    }
    let layer_image = svg.selectAll("image.layer_map")
        .data(layers)
        .enter()
        .append("image")
        .classed("layer_map", true)
        .attr('href', get_layer_map_thumbnail_url)
        .attr('x', layer => layer * node_space_width)
        .attr('y', 0)
        .attr('width', 96)
        .attr('height', 64)
        .on('click', function(event, layer) {
            let url = get_layer_map_full_url(layer);
            window.open(url, "_blank");
        })

    let image = svg.selectAll("image.activation_map").data(nodes);
    let imageEnter = image.enter().append("image").classed("activation_map", true);

    imageEnter.attr('href', get_thumnail_url)
    imageEnter.attr('x', node => node.x * node_space_width)
    imageEnter.attr('y', node => (node.y+1) * node_space_height)
    imageEnter.attr('width', 96)
    imageEnter.attr('height', 64)

    imageEnter.on('click', function(event, node) {
        select_node(node, nodes, links, parsed_data.num_layers);
    })


    /* TODO */
    /* 
    on click of image, make that image 'active'
    */
    /* only show links which connect to active feature
    
    */
}

fetch("weight_data")
    .then(response => response.json())
    .then(init_network)

