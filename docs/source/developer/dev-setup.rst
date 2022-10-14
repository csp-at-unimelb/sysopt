Developer Environment Setup
===========================

Development Toolchain
---------------------
Source repository for `sysopt` is hosted by github.
Make sure you have the appropriate level of access to the repo, or make a fork.

You will need `git`.

Install From Source
-------------------
It is recommended to use a virtual environment (https://virtualenv.pypa.io/en/stable/) for development.

To install sysopt from source fork or clone the git repository at (https://github.com/csp-at-unimelb/sysopt), navigate to the root directory (`sysopt/`) and install an editable copy using pip:

.. code-block:: console

    $ pip install -e .

Install Development Dependencies
--------------------------------
Additional dependencies for development, testing and documentation are listed in 'test-requirements.txt'.
Install these using pip:

.. code-block:: console

    $ pip install -r test-requirements.txt

Workflow
--------
Sysopt is currently using a topic branch workflow (https://git-scm.com/book/en/v2/Git-Branching-Branching-Workflows).
Generally speaking, this means the development process follows:
1. A new feature has been identified, or a bug needs to be fixed.
2. The developer creates a new branch from the most up-to-date version of `main`.
3. The developer locally checks out the branch, proceeds to implement the feature, appropriate tests and documentation.
4. 

