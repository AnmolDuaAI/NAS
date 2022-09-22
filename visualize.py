import sys
from turtle import fillcolor
import torch
from genotypes import *
from graphviz import Digraph
from model_search import Network

def plot(genotype, filename):
    g = Digraph(
        format = "pdf",
        edge_attr=dict(fontsize = '20', fontname = 'times'),
        node_attr = dict(
            style = "filled",
            shape = "rect",
            align = "center",
            fontsize = "20",
            height = "0.5",
            width = "0.5",
            penwidth = "2",
            fontname = "times"
        ),
        engine = 'dot'
    )


    g.body.extend(['rankdir=LR'])
    g.node("c_{k-2}", fillcolor='darkseagreen2')
    g.node("c_{k-1}", fillcolor='darkseagreen2')
    steps = len(genotype)//2
    for i in range(steps):
        g.node(str(i), fillcolor = "lightblue")

    for i in range(steps):
        for k in [2*i, 2*i + 1]:
            op, j = genotype[k]
            if (j == 0):
                u = "c_{k-2}"
            elif(j == 1):
                u = "c_{k-1}"
            else:
                u = str(j-2)
            v = str(i)
            g.edge(u,v, label = op, fillcolor = "gray")
    
    g.node("c_{k}", fillcolor = "plaegoldenrod")
    for i in range(steps):
        g.edge(str(i), "c_{k}", fillcolor = "gray")

    g.render(filename, view = True)


def main():

    # genotype = Genotype(normal=[('max_pool_3x3', 0), ('sep_conv_5x5', 1), ('max_pool_3x3', 0), ('dil_conv_3x3', 1), ('max_pool_3x3', 0), ('dil_conv_3x3', 3), ('max_pool_3x3', 0), ('sep_conv_5x5', 3)], normal_concat=range(2, 6), reduce=[('sep_conv_7x7', 0), ('sep_conv_5x5', 1), ('max_pool_3x3', 0), ('sep_conv_5x5', 2), ('sep_conv_7x7', 3), ('sep_conv_7x7', 1), ('sep_conv_7x7', 3), ('sep_conv_7x7', 2)], reduce_concat=range(2, 6))

    genotype = Genotype(normal=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('sep_conv_7x7', 3), ('sep_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 1), ('max_pool_3x3', 0), ('dil_conv_5x5', 1), ('max_pool_3x3', 0), ('sep_conv_7x7', 4)], reduce_concat=range(2, 6))

    plot(genotype.normal, "normal")
    plot(genotype.reduce, "reduce")

if __name__ == "__main__":
    main()