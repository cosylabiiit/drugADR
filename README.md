# drugADR

The project involves prediction of side effects prediction using existing data ([SIDER](http://http://sideeffects.embl.de/)) by leveraging machine learning with statistical data analysis. During the course of this work the following tasks were performed:
- Dealing with biological, chemical data.
- Computational manipulation of data from drugs and phenotypic side effects.
- Implementation of machine learning algorithms for prediction of side effects.

## Pre-requisites

The following are a couple of instructions to execute different (or all) sections of the project.

1. Clone the project, replacing ``drugADR`` with the name of the project you are creating:

        $ git clone https://github.com/sominwadhwa/drugADR.git drugADR
        $ cd drugADR

2. Make sure you have ``python 3.4.x`` running on your local system. If you do, skip this step. In case you don't, head
head [here](https'://www.python.org/downloads/).

3. ``virtualenv`` is a tool used for creating isolated 'virtual' python environments. It is advisable to create one here as well (to avoid installing the pre-requisites into the system-root). Do the following within the project directory:

        $ [sudo] pip install virtualenv
        $ virtualenv drugADR
        $ source drugADR/bin/activate

To deactivate later, just type ``deactivate``.

4.
