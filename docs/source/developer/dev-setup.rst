Development Overview
====================

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

Development Workflow
--------------------
Sysopt is currently using a topic branch workflow (https://git-scm.com/book/en/v2/Git-Branching-Branching-Workflows).
Generally speaking, this means the development process follows:
1. Identify a new feature, or a bug to be fixed.
2. Create a new branch from the most up-to-date version of `main`.
3. Check out the branch on a local machine, implement the feature along with appropriate tests and documentation.
4. Commit changes and push the local branch to the remote repo
5. Open a pull request to merge the new feature into `main`.
6. Resolve any merge conflicts, test failures and linting issues.
7. Respond to any review comments.
8. Delete the branch once it's been accepted.
