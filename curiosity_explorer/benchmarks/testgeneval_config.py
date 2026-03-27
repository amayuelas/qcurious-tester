"""TestGenEval Lite — per-repo Docker configuration.

Each repo in the TestGenEval Lite dataset uses a different SWE-bench testbed
Docker image with its own Python environment, working directory, and setup.

Two environment types:
  - conda: Python at /home/swe-bench/miniconda3/envs/{testbed}/bin/python
  - pyenv: Python at /opt/pyenv/versions/{pyver}/bin/python

For coverage measurement, we need to:
  1. Use the correct Python binary
  2. Set the correct working directory (where source is)
  3. Run any setup commands (e.g., django.setup())
"""

# Image pattern: aorwall/swe-bench-{owner}_{repo}-testbed:{version}

REPO_CONFIGS = {
    "django/django": {
        "image_template": "aorwall/swe-bench-django_django-testbed:{version}",
        "env_type": "pyenv",
        "working_dir": "/opt/django__django",
        "setup_code": "import django; django.setup()",
        "env": {"DJANGO_SETTINGS_MODULE": "tests.test_sqlite"},
        "prompt_note": "Django is already configured. Do NOT call settings.configure() or django.setup().",
        "available_versions": {"3.0", "3.1", "3.2", "4.0", "4.1", "4.2", "5.0"},
    },
    "sympy/sympy": {
        "image_template": "aorwall/swe-bench-sympy_sympy-testbed:{version}",
        "env_type": "conda",
        "testbed_template": "sympy__sympy__{version}",
        "working_dir": "/home/swe-bench/sympy__sympy",
        "setup_code": "",
        "env": {},
        "prompt_note": "",
        "available_versions": {"1.0", "1.1", "1.2", "1.4", "1.5", "1.6",
                               "1.7", "1.8", "1.9", "1.10", "1.11", "1.12", "1.13"},
    },
    "scikit-learn/scikit-learn": {
        "image_template": "aorwall/swe-bench-scikit-learn_scikit-learn-testbed:{version}",
        "env_type": "pyenv",
        "pyenv_version": "3.9.19",
        "working_dir": "/opt/scikit-learn__scikit-learn",
        "setup_code": "",
        "env": {},
        "prompt_note": "",
        "pre_install": "pip install --no-use-pep517 --no-build-isolation -e .",
        "available_versions": {"0.20", "0.21", "0.22", "1.3"},
    },
    "pytest-dev/pytest": {
        "image_template": "aorwall/swe-bench-pytest-dev_pytest-testbed:{version}",
        "env_type": "conda",
        "testbed_template": "pytest-dev__pytest__{version}",
        "working_dir": "/home/swe-bench/pytest-dev__pytest",
        "setup_code": "",
        "env": {},
        "prompt_note": "",
        "available_versions": {"4.4", "5.0", "5.2", "5.4", "6.0", "6.3", "7.0"},
    },
    "matplotlib/matplotlib": {
        "image_template": "aorwall/swe-bench-matplotlib_matplotlib-testbed:{version}",
        "env_type": "conda",
        "testbed_template": "matplotlib__matplotlib__{version}",
        "working_dir": "/home/swe-bench/matplotlib__matplotlib",
        "setup_code": "",
        "env": {},
        "prompt_note": "",
        "available_versions": {"3.5", "3.6", "3.7"},
    },
    "astropy/astropy": {
        "image_template": "aorwall/swe-bench-astropy_astropy-testbed:{version}",
        "env_type": "conda",
        "testbed_template": "astropy__astropy__{version}",
        "working_dir": "/home/swe-bench/astropy__astropy",
        "setup_code": "",
        "env": {},
        "prompt_note": "",
        "available_versions": {"1.3", "4.3", "5.1"},
    },
    "pydata/xarray": {
        "image_template": "aorwall/swe-bench-pydata_xarray-testbed:{version}",
        "env_type": "conda",
        "testbed_template": "pydata__xarray__{version}",
        "working_dir": "/home/swe-bench/pydata__xarray",
        "setup_code": "",
        "env": {},
        "prompt_note": "",
        "available_versions": {"0.12"},
    },
    "mwaskom/seaborn": {
        "image_template": "aorwall/swe-bench-mwaskom_seaborn-testbed:{version}",
        "env_type": "conda",
        "testbed_template": "mwaskom__seaborn__{version}",
        "working_dir": "/home/swe-bench/mwaskom__seaborn",
        "setup_code": "",
        "env": {},
        "prompt_note": "",
        "available_versions": {"0.12"},
    },
    "sphinx-doc/sphinx": {
        "image_template": "aorwall/swe-bench-sphinx-doc_sphinx-testbed:{version}",
        "env_type": "conda",
        "testbed_template": "sphinx-doc__sphinx__{version}",
        "working_dir": "/home/swe-bench/sphinx-doc__sphinx",
        "setup_code": "",
        "env": {},
        "prompt_note": "",
        "available_versions": {"3.2", "3.5"},
    },
    "pylint-dev/pylint": {
        "image_template": "aorwall/swe-bench-pylint-dev_pylint-testbed:{version}",
        "env_type": "conda",
        "testbed_template": "pylint-dev__pylint__{version}",
        "working_dir": "/home/swe-bench/pylint-dev__pylint",
        "setup_code": "",
        "env": {},
        "prompt_note": "",
        "pre_install": "pip install -e .",
        "available_versions": {"2.13", "2.15"},
    },
    "pallets/flask": {
        "image_template": "aorwall/swe-bench-pallets_flask-testbed:{version}",
        "env_type": "conda",
        "testbed_template": "pallets__flask__{version}",
        "working_dir": "/home/swe-bench/pallets__flask",
        "setup_code": "",
        "env": {},
        "prompt_note": "",
        "available_versions": {"2.3"},
    },
}


def get_repo_config(repo, version):
    """Get Docker config for a TestGenEval repo/version pair.

    Returns dict with: image, working_dir, setup_code, env, prompt_note,
                       env_type, python_bin (for conda repos)
    """
    if repo not in REPO_CONFIGS:
        raise ValueError(f"Unknown repo: {repo}. Available: {list(REPO_CONFIGS.keys())}")

    cfg = REPO_CONFIGS[repo]
    if version not in cfg["available_versions"]:
        raise ValueError(f"Version {version} not available for {repo}. "
                         f"Available: {cfg['available_versions']}")

    image = cfg["image_template"].format(version=version)

    result = {
        "image": image,
        "working_dir": cfg["working_dir"],
        "setup_code": cfg["setup_code"],
        "env": dict(cfg["env"]),
        "prompt_note": cfg["prompt_note"],
        "env_type": cfg["env_type"],
    }

    # For conda repos, compute the conda env name and Python binary path
    if cfg["env_type"] == "conda":
        testbed = cfg.get("testbed_template", "").format(version=version)
        result["python_bin"] = f"/home/swe-bench/miniconda3/envs/{testbed}/bin/python"
        result["testbed_name"] = testbed

    # For pyenv repos (scikit-learn), set python_bin from pyenv_version
    if cfg["env_type"] == "pyenv" and "pyenv_version" in cfg:
        result["python_bin"] = f"/opt/pyenv/versions/{cfg['pyenv_version']}/bin/python"

    # Pass through pre_install if present
    if "pre_install" in cfg:
        result["pre_install"] = cfg["pre_install"]

    return result


def load_testgeneval_examples(repos=None, max_examples=None):
    """Load all TestGenEval Lite examples with Docker configs.

    Args:
        repos: Filter by repo (e.g., ["django/django", "sympy/sympy"])
        max_examples: Limit total examples

    Returns list of dicts with: module, repo, version, code_file, code_src,
    docker_image, setup_code, working_dir, env, prompt_note, env_type
    """
    from datasets import load_dataset
    ds = load_dataset("kjain14/testgenevallite")
    test = ds["test"]

    examples = []
    for ex in test:
        repo = ex["repo"]
        version = ex["version"]

        if repo not in REPO_CONFIGS:
            continue
        if version not in REPO_CONFIGS[repo]["available_versions"]:
            continue
        if repos and repo not in repos:
            continue

        cfg = get_repo_config(repo, version)
        module = ex["code_file"].replace("/", ".").replace(".py", "")
        # Strip source layout prefixes (lib/, src/) that aren't part of the
        # Python package name. e.g., lib.matplotlib.figure → matplotlib.figure
        for prefix in ("lib.", "src."):
            if module.startswith(prefix):
                module = module[len(prefix):]
                break

        examples.append({
            "module": module,
            "repo": repo,
            "version": version,
            "code_file": ex["code_file"],
            "code_src": ex["code_src"],
            "baseline_cov": ex["baseline_covs"]["first"],
            "instance_id": ex["instance_id"],
            **cfg,
        })

    examples.sort(key=lambda x: x["baseline_cov"])

    if max_examples:
        examples = examples[:max_examples]

    return examples
