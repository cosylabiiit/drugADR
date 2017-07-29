# drugADR

The project involves prediction of side effects prediction using existing data ([SIDER](http://sideeffects.embl.de/)) by leveraging machine learning with statistical data analysis. During the course of this project the following tasks were performed:
- Dealing with biological, chemical data.
- Computational manipulation of data from drugs and phenotypic side effects.
- Implementation of machine learning algorithms for prediction of side effects.

## Pre-requisites

The following are a couple of instructions that must be gone through in order to execute different (or all) sections of this project.

1. Clone the project, replacing ``drugADR`` with the name of the directory you are creating:

        $ git clone https://github.com/sominwadhwa/drugADR.git drugADR
        $ cd drugADR

2. Make sure you have ``python 3.4.x`` running on your local system. If you do, skip this step. In case you don't, head
head [here](https://www.python.org/downloads/).

3. ``virtualenv`` is a tool used for creating isolated 'virtual' python environments. It is advisable to create one here as well (to avoid installing the pre-requisites into the system-root). Do the following within the project directory:

        $ [sudo] pip install virtualenv
        $ virtualenv --system-site-packages drugADR
        $ source drugADR/bin/activate

To deactivate later, once you're done with the project, just type ``deactivate``.

4. Install the pre-requisites from ``requirements.txt`` & run ``test/init.py`` to check if all the required packages were correctly installed:

        $ pip install -r requirements.txt
        $ python test/init.py

You should see an output - ``Imports successful. Good to go!``

## Directory Structure

#### Top-Level directory structure:

    .
    ├── src                     # Source files
    ├── data                    # Data used and/or generated
    ├── test                    # testing modules (including those for random-control experiments)                  
    ├── LICENSE
    └── README.md


#### Files' Description:

- ``/data/meddra_all_se.tsv``: File obtained from [SIDER](http://sideeffects.embl.de/download/) containing drug-ADR associations.
- ``/src/preprocess_sider.py``: Loads the original SIDER data & fetches all the relevant identification tags (InChi, SMILES etc) required for extraction of the drug chemical properties through ``pubchempy``. Output of the script is a dataframe (table) dump in ``/data/id_df.sav`` containing various identification tags of all 1430 drugs present in SIDER4.
- ``/data/2d_prop.xlsx`` & ``/data/3d_prop.xlsx``: Chemical Properties for 1430 drugs generated using DiscoveryStudio4. They form the basis of our feature set.
- ``/src/prop_pca.py``: Code for Principal Component Analysis on 2D & 3D molecular properties of drugs. Outputs cumulative preserved variance of first one hundred principal components (>99%).
- ``/src/base_se.py``: Code for predicting ADR at the SE level using OneVsRest multi-class multi-label classification. Results generated are saved in a (table/dataframe) ``pickle`` dump ``/data/all_se_clf_data.sav``.
-  ``/data/o_v2.xlsx``: Base data file used for organ level classification.
-  ``/data/sub_sys.xlsx``: Base data file used for sub-systems level classification.
-  ``/data/os_v2.xlsx``: Base data file used for organ-systems level classification.
- ``/src/base_o.py``: Prediction of ADR with first level of classification based on anatomical schema -- organ level, 61 classes against 1430 drugs. Generated an output (workbook) in ``/data/o_v2_results.xlsx`` containing the results.
- ``/src/base_osub.py``: Prediction of ADR with second level of classification based on anatomical schema -- sub-systems level, 30 classes against 1430 drugs. Generated an output (workbook) in ``/data/osub_results.xlsx`` containing the results.
- ``/src/base_osys.py``: Prediction of ADR with final level of classification based on anatomical schema -- sub-systems level, 11 classes against 1430 drugs. Generated an output (workbook) in ``/data/osys_results.xlsx`` containing the results.
- ``/test/rand_o.py``: Script to run random-control experiments on organ-level. Generates an output with a compilation of results. ``/data/list_res_organ.sav``.
- ``/test/rand_osub.py``: Script to run random-control experiments on sub-systems level. Generates an output with a compilation of results.``/data/list_res_Sub_Sys.sav``.
- ``/test/rand_osys.py``: Script to run random-control experiments on organ-systems level. Generates an output with a compilation of results.``/data/list_res_S.sav``.
