import numpy as np
import pandas as pd
import anndata
import scprep
import scanpy as sc
import sklearn
from sklearn.model_selection import train_test_split
import tempfile
import os
from os import path
import sys
import scipy
from scipy import sparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split

import load_raw
import normalize_tools as nm
import metrics


def main():

	# download data
	rna_data, atac_data, rna_cells, atac_cells, rna_genes, atac_genes = load_raw.load_raw_cell_lines()

	

if __name__ == '__main__':
	main()