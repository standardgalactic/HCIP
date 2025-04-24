# MR_CT_NW_MJ_2024

This repository contains the code and data to generate figures for manuscript "Computational basis of hierarchical and counterfactual information processing".

Link to paper: place_holder

## How to run
Different parts of the project was implemented with different programming languages.

To generate Figure 2, Figure 3, Figure 4-6, run the respective scripts under <code> src/matlab </code>

To generate Figure 1d, Figure 7, Figure 8, run scripts under <code> src/python/submission </code>

## Human Data

The raw human subjects data are also contained in this repository.

For matlab code, the data are contained in <code> src/matlab/<Figure_Number>/Data </code>

For python code, the data are contained in <code> src/python/human </code>


## Overview
The main source files for the RNN training are the following:

<code> src/python/task </code> contains the script generating simulation trials.

<code> src/python/nets </code> contains the code for RNNs.

<code> src/python/config </code> contains the hyperparameters for the task variables, RNN size, and training schedule. <code> hp0.py-hp7.py </code> are the hyperparameters for different RNN variants.

<code> src/python/analysis </code> contains the post-hoc analysis code for trained RNNs. 




