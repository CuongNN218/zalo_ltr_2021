<!---
Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Checks on a Pull Request

When you open a pull request on 🤗 Transformers, a fair number of checks will be run to make sure the patch you are adding is not breaking anything existing. Those checks are of four types:
- regular tests
- documentation build
- code and documentation style
- general repository consistency

In this document, we will take a stab at explaining what those various checks are and the reason behind them, as well as how to debug them locally if one of them fails on your PR.

Note that they all require you to have a dev install:

```bash
pip install transformers[dev]
```

or for an editable install:

```bash
pip install -e .[dev]
```

inside the Transformers repo.

## Tests

All the jobs that begin with `ci/circleci: run_tests_` run parts of the Transformers testing suite. Each of those jobs focuses on a part of the library in a certain environment: for instance `ci/circleci: run_tests_pipelines_tf` runs the pipelines test in an environment where TensorFlow only is installed.

Note that to avoid running tests when there is no real change in the modules they are testing, only part of the test suite is run each time: a utility is run to determine the differences in the library between before and after the PR (what GitHub shows you in the "Files changes" tab) and picks the tests impacted by that diff. That utility can be run locally with:

```bash
python utils/test_fetcher.py
```

from the root of the Transformers repo. It will:

1. Check for each file in the diff if the changes are in the code or only in comments or docstrings. Only the files with real code changes are kept.
2. Build an internal map that gives for each file of the source code of the library all the files it recursively impacts. Module A is said to impact module B if module B imports module A. For the recursive impact, we need a chain of modules going from module A to module B in which each module imports the previous one.
3. Apply this map on the files gathered in step 1, which  gives us the list of model files impacted by the PR.
4. Map each of those files to their corresponding test file(s) and get the list of tests to run.

When executing the script locally, you should get the results of step 1, 3 and 4 printed and thus know which tests are run. The script will also create a file named `test_list.txt` which contains the list of tests to run, and you can run them locally with the following command:

```bash
python -m pytest -n 8 --dist=loadfile -rA -s $(cat test_list.txt)
```

Just in case anything slipped through the cracks, the full test suite is also run daily.

## Documentation build

The job `ci/circleci: build_doc` runs a build of the documentation just to make sure everything will be okay once your PR is merged. If that steps fails, you can inspect it locally by going into the `docs` folder of the Transformers repo and then typing

```bash
make html
```

Sphinx is not known for its helpful error messages, so you might have to try a few things to really find the source of the error.

## Code and documentation style

Code formatting is applied to all the source files, the examples and the tests using `black` and `isort`. We also have a custom tool taking care of the formatting of docstrings and `rst` files (`utils/style_doc.py`), as well as the order of the lazy imports performed in the Transformers `__init__.py` files (`utils/custom_init_isort.py`). All of this can be launched by executing

```bash
make style
```

The CI checks those have been applied inside the `ci/circleci: check_code_quality` check. It also runs `flake8`, that will have a basic look at your code and will complain if it finds an undefined variable, or one that is not used. To run that check locally, use

```bash
make quality
```

This can take a lot of time, so to run the same thing on only the files you modified in the current branch, run

```bash
make fixup
```

This last command will also run all the additional checks for the repository consistency. Let's have a look at them.

## Repository consistency

This regroups all the tests to make sure your PR leaves the repository in a good state, and is performed by the `ci/circleci: check_repository_consistency` check. You can locally run that check by executing the following:

```bash
make repo-consistency
```

This checks that:

- All objects added to the init are documented (performed by `utils/check_repo.py`)
- All `__init__.py` files have the same content in their two sections (performed by `utils/check_inits.py`)
- All code identified as a copy from another module is consistent with the original (performed by `utils/check_copies.py`)
- The translations of the READMEs and the index of the doc have the same model list as the main README (performed by `utils/check_copies.py`)
- The auto-generated tables in the documentation are up to date (performed by `utils/check_table.py`)
- The library has all objects available even if not all optional dependencies are installed (performed by `utils/check_dummies.py`)

Should this check fail, the first two items require manual fixing, the last four can be fixed automatically for you by running the command

```bash
make fix-copies
```

Additional checks concern PRs that add new models, mainly that:

- All models added are in an Auto-mapping (performed by `utils/check_repo.py`)
<!-- TODO Sylvain, add a check that makes sure the common tests are implemented.-->
- All models are properly tested (performed by `utils/check_repo.py`)

<!-- TODO Sylvain, add the following
- All models are added to the main README, inside the master doc
- All checkpoints used actually exist on the Hub

-->