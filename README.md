# ectoderm-myeloid-tracking-analysis

This repo hosts the code for the python-based quantitative analysis of tracking data in the paper entitled "[Tissue flow acts as a guidance cue for immune cell polarisation and directional migration](https://doi.org/10.1038/s41556-026-02058-9)" by Hoang Anh Le and colleagues, as described in the Materials and Methods section under the heading `Trajectory deviation, speed correlation, and cosine similarity analyses`. All code was written by Jonas Hartmann.


### Analyses Included

- Local myeloid-ectoderm cross-correlation analysis *ex vivo* (figs. `4 I-J` and `S3 I-L`)
- Adapted pipeline and analysis for *in vivo* data (figs. `S3 M-N`)
- Radial velocity analysis and comparison to active wetting model (fig. `4 L`) [model by Ricard Alert]


### Repo structure

- The `tracking_analysis` module contains a variety of refactored functions that are used in the notebooks
- `RUN` notebooks are sequential pipelines that ingest (raw) data, process it, and output derived metrics
    1. `RUN - 1 - Preprocessing`: Parse, clean, and convert the raw tracks into pandas dataframes
    2. `RUN - 2 - Movement vectors`: Construct motion vectors and interpolate local neighborhood consensus
    3. `RUN - 3 - Correlation and similarity measurements`: Measure local correlations/similarities
- `ANA` notebooks ingest the derived metrics and produce figures and statistics
- The `in vivo` notebooks are a simplified equivalent of the initially developed *ex vivo* analysis
- For more information see the notes at the top of each notebook


### Data Availability

The `Data/` dir contains the raw tracking files to be ingested by the `RUN - 1 - Preprocessing` notebooks. All derived data can be reconstructed by running the notebooks as described above. The raw time-lapse microscopy data from which tracks were derived are not included; please contact the corresponding authors to obtain these data. The tracks used here were generated automatically using TrackMate with StarDist for the ectoderm, and manually using the MTrackJ plugin for the embryonic myeloid cells.


### Dependencies

- The code was written in `python 3.12.9` (on Windows 10) using the following common scientific libraries:
    - `numpy 2.3.1`, `scipy 1.15.2`, `matplotlib 3.10.3`, `networkx 3.5`, `scikit-learn 1.7.0`, `scikit-image 0.25.2`, `statsmodels 0.14.4`, `ipykernel 6.29.5`, `ipywidgets 8.1.7`
- [Miniforge](https://conda-forge.org/download/) was used for environment/package management using `conda`
- With conda, the environment used here can be recreated as follows:
    - For `conda >= 26.5`: `conda create --name em_tracking_analysis --file conda-lock.yaml`
    - Otherwise: `conda create --name em_tracking_analysis --file conda-spec-file.txt`


### Contact and Support

- The study's corresponding authors are Hoang Anh Le (anh.le@domain) and Roberto Mayor (r.mayor@domain)
- For questions about this repo, contact Jonas Hartmann (jonas.hartmann@domain) or open a GitHub issue
- Note that we cannot promise support for uses other than direct reproduction of the study's results
- Domain: ucl.ac.uk
