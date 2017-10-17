# drugADR

The project involves implementation of a hierarchical anatomical schema for aggregation of side effects towards prediction of side effects using existing data ([SIDER4](http://sideeffects.embl.de/)) by leveraging machine learning and statistical data analysis. During the course of this project the following tasks were performed:
- Extraction of relevant data of drug side effects, chemical properties etc.
- Hierarchical classification of side effects based on organ/system involved.
- Data preprocessing.
- Implementation of machine learning algorithms for prediction of side effects.

Authors: Somin Wadhwa†, [Aishwarya Gupta](https://github.com/agupta04)†, [Shubham Dokania](https://github.com/shubham1810)†, [Rakesh Kanji](http://cosylab.iiitd.edu.in/people/RKanji.html), [Ganesh Bagler](http://cosylab.iiitd.edu.in/)*

† Equal contribution

&ast; Corresponding Author (bagler@iiitd.ac.in)

This work was done in the Complex Systems Laboratory, Center for Computational Biology, IIIT-Delhi.

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

#### Top-Level Structure:

    .
    .
    ├── data                     # Data used and/or generated
    │   ├── 2d_prop.xlsx
    │   ├── 3d_prop.xlsx
    │   ├── all_se_clf_data.sav
    │   ├── AssociatedDrugsVsSideEffects.png
    │   ├── id_df.sav
    │   ├── list_res_organ.sav
    │   ├── list_res_S.sav
    │   ├── list_res_Sub_Sys.sav
    │   ├── meddra_all_se.tsv
    │   ├── misc.xlsx
    │   ├── o_v2.xlsx
    │   ├── os_v2.xlsx
    │   ├── SideEffectsVsAssociatedDrugs.png
    │   ├── sub_sys.xlsx
    │   ├── unique_SE.csv
    ├── src                    # Source Files
    │   ├── base_o.py
    │   ├── base_osub.py
    │   ├── base_osys.py
    │   ├── match.py
    │   ├── preprocess_sider.py
    │   ├── prop_pca.py
    │   ├── voting_o.py
    │   ├── voting_osub.py
    │   ├── voting_osys.py
    ├── test                    # Testing modules (including those for random-control experiments)
    │   ├── init.py
    │   ├── rand_o.py
    │   ├── random_osub.py
    │   ├── random_osys.py                  
    ├── LICENSE
    └── README.md
    .
    .


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
- ``/src/voting_o.py``: Voting ensemble model at organ level. Generates & stores output ``/data/o_votingModel_results.xlsx``.
- ``/src/voting_osub.py``: Voting ensemble model at sub-systems level. Generates & stores output ``/data/osub_votingModel_results.xlsx``.
- ``/src/voting_osys.py``: Voting ensemble model at organ-system level. Generates & stores output ``/data/osys_votingModel_results.xlsx``.
- ``/test/rand_o.py``: Script to run random-control experiments on organ-level. Generates an output with a compilation of results. ``/data/list_res_organ.sav``.
- ``/test/rand_osub.py``: Script to run random-control experiments on sub-systems level. Generates an output with a compilation of results.``/data/list_res_Sub_Sys.sav``.
- ``/test/rand_osys.py``: Script to run random-control experiments on organ-systems level. Generates an output with a compilation of results.``/data/list_res_S.sav``.

Description/Information about files other than those mentioned up can be directly inferred from the article/paper.

## Running the tests

To run something simple, simply execute the standalone ``.py`` script via command line:

        $ python3 test/rand_o.py
        $ python3 src/base_o.py
        $ python3 src/prop_pca.py

Advisory: All these experiments were carried out on IIIT-Delhi's HPC server-node with [these specifications](http://it.iiitd.edu.in/HPC_final_doc.pdf) due to the volume & time of compute required. It is advised to run any tests in a similar environment.

## Acknowledgements

G.B. thanks the Indraprastha Institute of Information Technology (IIIT-Delhi) for providing computational facilities and support. S.W., A.G. and S.D. were Summer Research Interns in Dr. Bagler's lab at the Center for Computational Biology, and are thankful to IIIT-Delhi for the support and fellowship. R.K. thanks the Ministry of Human Resource Development, Government of India and Indian Institute of Technology Jodhpur for the senior research fellowship.  
