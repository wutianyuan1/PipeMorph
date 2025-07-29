# Attack of the Bubbles: Straggler-Resilient Pipeline Parallelism for Large Model Training
*Artifact Evaluation Guidelines for NSDI'26 Paper #218*

## Testbed Environment and Experiments Description

We provide **four VM instances on Alibaba Cloud** with NVIDIA A10 GPUs to reproduce key results from the paper, including:
- All simulation results and scheduler evaluations, including Figures 4, 5, 8, and 19.
- Performance evaluation results (figures in the Evaluation section):
    * Microbenchmarks assessing PipeMorph's behavior
    * Sensitivity analysis to network delays
    * Single-link degradation evaluation
    * Multi-link degradation evaluation
    * Overhead evaluation

*Note 1*: Full reproduction of large-scale experiments (e.g., 140B model training) requires a cluster with hundreds of GPUs. While we cannot provide such infrastructure, reviewers with access to large-scale resources may replicate these experiments. On our provided VMs, the training scale is limited (4-stage PP, no DP or TP), but the key results and trends remain similar.

*Note 2*: The results shown in the paper were evaluated on NVIDIA H800 clusters, while we can only provide NVIDIA A10 GPUs. Thus, the absolute performance will not match, but the overall trends will be similar. Since computation on the A10 is significantly slower than on the H800 (e.g., a forward pass on the H800 may take 10ms, while it takes 30ms on the A10), the injected communication delays are increased in the AE scripts (e.g., from 30/60ms to 60/120ms) to maintain a similar communication-to-computation ratio.

*Note 3*: As running VMs continuously is expensive, we will start them on demand. If you would like to use these VMs, please email me (twubt@connect.ust.hk) or reply to me on the HotCRP site, and I will start them upon your request and give you corresponding node IP and password.


## Environment Setup

### Option1: Using the provided VMs

We strongly recommend using our provided instances, as no additional environment setup is required. The source code is located at `/root/workspace/PipeMorph-AE` on these instances.

### Option2: Setting up locally via Docker

If you wish to evaluate PipeMorph in your local environment using Docker, follow these steps:
- First, pull the pytorch-23.05-py3 image from the NVCR repository:
```shell
docker pull nvcr.io/nvidia/pytorch:23.05-py3
```
- Next, install the required libraries:
```shell
apt-get update
apt-get install redis-server
pip3 install pulp matplotlib regex ninja cmake pybind11 sentencepiece
```
- Next, clone the repo and checkout to the `AE` branch, then compile the failslow-injection library using `PipeMorph/failslow_injection_compile.sh`. You may need to modify relative paths in this script.
- Finally, clone the PipeMorph library and run the scripts. You may need to modify all IPs and paths in the scripts under the `PipeMorph/ae` folder and in `PipeMorph/zerobubble/examples/pipemorph_ae.sh`.

## Reproducing Key Results
- Switch to the AE branch: First, switch to the AE branch:
```shell
git checkout AE
```
- Reproduce simulation and scheduler results: You can find a Jupyter Notebook at `PipeMorph/zerobubble/pipeline_simulator/figures.ipynb`. Running this notebook produces Figures 4, 5, 8, and 19.

- Reproduce evaluation results: In the `PipeMorph/ae` folder, there are subfolders named `fig*`, each corresponding to a evaluation figure in the paper. The entry script is `PipeMorph/ae/run_all.sh`. For example, to reproduce Figure 15, run:
```shell
./ae/run_all.sh fig15
python ./ae/fig15/plot_fig15.py
```
Once the experiment completes, you will see output like All nodes have completed their tasks. You can then check the results under `PipeMorph/ae/fig*/fig*.pdf`.
