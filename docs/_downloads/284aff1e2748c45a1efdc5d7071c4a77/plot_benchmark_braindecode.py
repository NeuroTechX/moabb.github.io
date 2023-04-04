"""
=======================================================================
Benchmarking on MOABB with Braindecode (PyTorch) deep net architectures
=======================================================================
This example shows how to use MOABB to benchmark a set of Braindecode pipelines (deep learning
architectures) on all available datasets.
For this example, we will use only 2 datasets to keep the computation time low, but this benchmark is designed
to easily scale to many datasets.
"""
# Authors: Igor Carrara <igor.carrara@inria.fr>
#          Bruno Aristimunha <b.aristimunha@gmail.com>
#          Sylvain Chevallier <sylvain.chevallier@universite-paris-saclay.fr>
#
# License: BSD (3-clause)

import os

import matplotlib.pyplot as plt
import torch
from absl.logging import ERROR, set_verbosity

from moabb import benchmark, set_log_level
from moabb.analysis.plotting import score_plot
from moabb.datasets import BNCI2014001, BNCI2014004
from moabb.utils import setup_seed


set_log_level("info")
# Avoid output Warning
set_verbosity(ERROR)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Print Information PyTorch
print(f"Torch Version: {torch.__version__}")

# Set up GPU if it is there
cuda = torch.cuda.is_available()
device = "cuda" if cuda else "cpu"
print("GPU is", "AVAILABLE" if cuda else "NOT AVAILABLE")

###############################################################################
# In this example, we will use only 2 subjects from the dataset ``BNCI2014001`` and ``BNCI2014004``.
#
# Running the benchmark
# ---------------------
#
# The benchmark is run using the ``benchmark`` function. You need to specify the
# folder containing the pipelines, the kind of evaluation, and the paradigm
# to use. By default, the benchmark will use all available datasets for all
# paradigms listed in the pipelines. You could restrict to specific evaluation and
# paradigm using the ``evaluations`` and ``paradigms`` arguments.
#
# To save computation time, the results are cached. If you want to re-run the
# benchmark, you can set the ``overwrite`` argument to ``True``.
#
# It is possible to indicate the folder to cache the results and the one to save
# the analysis & figures. By default, the results are saved in the ``results``
# folder, and the analysis & figures are saved in the ``benchmark`` folder.
#
# This code is implemented to run on CPU. If you're using a GPU, do not use multithreading
# (i.e. set n_jobs=1)
#
# In order to allow the benchmark function to work with return_epoch=True (Required to use Braindecode(
# we need to call each pipeline as "braindecode_xxx...", with xxx the name of the model to be
# handled correctly by the benchmark function.

# Set up reproducibility of Tensorflow
setup_seed(42)

# Restrict this example only to the first two subjects of BNCI2014001
dataset = BNCI2014001()
dataset2 = BNCI2014004()
dataset.subject_list = dataset.subject_list[:2]
dataset2.subject_list = dataset2.subject_list[:2]
datasets = [dataset, dataset2]

results = benchmark(
    pipelines="./pipelines_braindecode",
    evaluations=["CrossSession"],
    paradigms=["LeftRightImagery"],
    include_datasets=datasets,
    results="./results/",
    overwrite=False,
    plot=False,
    output="./benchmark/",
    n_jobs=-1,
)

###############################################################################
# The deep learning architectures implemented in MOABB using Braindecode are:
#
# - Shallow Convolutional Network [1]_
# - Deep Convolutional Network [1]_
# - EEGNetv4 [2]_
# - EEGInception [3]_
#
# Benchmark prints a summary of the results. Detailed results are saved in a
# pandas dataframe, and can be used to generate figures. The analysis & figures
# are saved in the ``benchmark`` folder.

score_plot(results)
plt.show()

##############################################################################
# References
# ----------
# .. [1] Schirrmeister, R. T., Springenberg, J. T., Fiederer, L. D. J.,
#    Glasstetter, M., Eggensperger, K., Tangermann, M., ... & Ball, T. (2017).
#    `Deep learning with convolutional neural networks for EEG decoding and
#    visualization <https://doi.org/10.1002/hbm.23730>`_.
#    Human brain mapping, 38(11), 5391-5420.
# .. [2] Lawhern, V. J., Solon, A. J., Waytowich, N. R., Gordon, S. M.,
#    Hung, C. P., & Lance, B. J. (2018). `EEGNet: a compact convolutional neural
#    network for EEG-based brain-computer interfaces.
#    <https://doi.org/10.1088/1741-2552/aace8c>`_
#    Journal of neural engineering, 15(5), 056013.
# .. [3] Santamaria-Vazquez, E., Martinez-Cagigal, V., Vaquerizo-Villar,
#    F., & Hornero, R. (2020). `EEG-inception: A novel deep convolutional neural network
#    for assistive ERP-based brain-computer interfaces.
#    <https://doi.org/10.1109/TNSRE.2020.3048106>`_
#    IEEE Transactions on Neural Systems and Rehabilitation Engineering
