import os
import glob

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

D = 5
BETA = 10
DT = 0.01
ITERATIONS = 5000

INITIAL_NODE = 0.5
INITIAL_WEIGHT = 0.5

PATH_PREFIX = os.path.dirname(os.path.realpath(__file__))


def f(x):
    return (x - 0.25) ** 3


def init_network(initial_node, initial_weight):
    # Set initial values for the whole network
    network = nx.karate_club_graph()
    nx.set_node_attributes(network, initial_node, "state")
    nx.set_edge_attributes(network, initial_weight, "weight")

    # Fix initial values for Mr Hi and Officer
    network.nodes[0]["state"] = 1.0  # Mr Hi
    network.nodes[33]["state"] = 0.0  # John A
    return network


def update_network(network):
    network_copy = network.copy()

    # Diffuse nodes states
    for node_i in network_copy.nodes:
        for node_j in network_copy.adj[node_i]:
            c_diff = (
                network_copy.nodes[node_j]["state"]
                - network_copy.nodes[node_i]["state"]
            )
            w_ij = network_copy[node_i][node_j]["weight"]
            if node_i != 0 and node_i != 33:
                network.nodes[node_i]["state"] += DT * D * c_diff * w_ij
            network[node_i][node_j]["weight"] -= (
                DT * BETA * w_ij * (1 - w_ij) * f(abs(c_diff))
            )
    return network


def show_network(network):
    nodes_colors = [
        "red" if network.nodes[i]["club"] == "Mr. Hi" else "blue"
        for i in network.nodes
    ]
    nx.draw_spring(network, node_color=nodes_colors)
    plt.show(block=False)


def save_network(network, path):
    nodes_colors = [network.nodes[i]["state"] for i in network.nodes]
    fig = plt.figure()
    nx.draw_spring(
        network,
        cmap=plt.cm.spring,
        vmin=0,
        vmax=1,
        with_labels=True,
        node_color=nodes_colors,
    )
    i = 0
    name_index = "000"
    while os.path.exists(os.path.join(path, "frame%s.png" % name_index)):
        i += 1
        name_index = (3 - len(str(i))) * "0" + str(i)
    # leading zeros in frame name
    name_index = (3 - len(str(i))) * "0" + str(i)
    fig_name = os.path.join(path, "frame%s.png" % name_index)
    plt.savefig(fig_name)
    # fig.savefig("temp/graph" + str(int(k / 100)) + ".png")


def main():
    # Create directory for frames
    path = os.path.join(PATH_PREFIX, "temp")
    if not os.path.exists(path):
        os.makedirs(path)

    # Empty the frames directory
    files = glob.glob(os.path.join(PATH_PREFIX, "temp/*"))
    for f in files:
        os.remove(f)

    # Simulate netwrok evolution
    g = init_network(INITIAL_NODE, INITIAL_WEIGHT)
    for i in range(ITERATIONS):
        g = update_network(g)
        if i % 100 == 0:
            save_network(g, path)


if __name__ == "__main__":
    main()
