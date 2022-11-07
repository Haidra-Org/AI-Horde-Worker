# Contributing

Thanks for helping!

We welcome all kinds of contributions:

- Bug fixes
- Documentation improvements
- New features
- Refactoring & tidying


## Getting started

If you have a specific contribution in mind, be sure to check the
[issues](https://github.com/Sygil-Dev/nataili/issues)
and [pull requests](https://github.com/Sygil-Dev/nataili/pulls)
in progress - someone could already be working on something similar
and you can help out.


## Coding guidelines

Several tools are used to ensure a coherent coding style.
You need to make sure that your code satisfy those requirements
or the automated tests will fail.

- [black code formatter](https://github.com/psf/black)
- [flake8 style enforcement](https://flake8.pycqa.org/en/latest/index.html)
- [isort to sort imports alphabetically](https://isort.readthedocs.io/en/stable/)

On Linux or MacOS, you can fix and check your code style by running
the command `style.sh --fix`

On Windows, you can fix and check your code by running
the command `style.cmd --fix`

To only verify the code without making changes, run the above script without --fix

## Development dependencies setup

To install the correct development dependencies of the linting
tools described above, run:

```sh
python -m pip install -r requirements.dev.txt
```

To setup a pre-commit hook to verify your code at each commit:

On linux, run:
```sh
ln -s ../../style.sh ./.git/hooks/pre-commit
```

On windows, put this in ./.git/hooks/pre-commit :
```sh
#!/bin/sh
sh style.sh
```

## How to create a good Pull Request

1. Make a fork of the main branch on github
2. Clone your forked repo on your computer
3. Create a feature branch `git checkout -b feature_my_awesome_feature`
4. Modify the code
5. Verify that the [Coding guidelines](#coding-guidelines) are respected
6. Make a commit and push it to your fork
7. From github, create the pull request. Automated tests from GitHub actions
will then automatically check the style
8. If other modifications are needed, you are free to create more commits and
push them on your branch. They'll get added to the PR automatically.

Once the Pull Request is accepted and merged, you can safely
delete the branch (and the forked repo if no more development is needed).
