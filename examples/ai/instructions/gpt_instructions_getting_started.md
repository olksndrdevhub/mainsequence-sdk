# Getting Started with the Main Sequence Platform

## 1. Create a New Project

To get started with the Main Sequence Platform, first create a project via the GUI.

1. Go to https://main-sequence.app/projects/ and click the **Create New Project** button in the top-right corner.
2. Select **Create in Cloud Platform**.
3. Enter the project name. For now, keep **Data Source** as **Default**.
4. Leave **Environment Variables** empty; you can change them later.
5. Click **Create**.

Your new project is being linked and set up on the platform. You’re ready to start developing.

## 2. Set Up Your Project Locally

Now let’s set up your local environment to communicate with the Main Sequence Platform. You can do this using the CLI; see the [CLI instructions](./gpt_instructions_cli.md) for details.

You should now have your project set up locally.

## 3. Developing on the Main Sequence Platform: Core Concepts

Your project automatically creates a GitHub repository with a branch. Everything you push to this branch becomes available to the platform’s compute engine immediately.

A newly created project’s structure looks like this:

```
REPO_NAME/
├─ dashboards/
├─ src/
│  └─ data_nodes/
├─ scripts/
├─ requirements.txt
├─ pyproject.toml
├─ Readme.md

```

### Repository conventions

- **Can I change the structure?** Yes—but we recommend keeping the default layout so projects stay consistent across teams and tooling.

- **Project vs. library.** This repository is meant to be run as a project, not published as a library. You don’t need packaging or installation steps like `setup.py` or `pip install .`.  
  *Note:* `pyproject.toml` is primarily for tooling/config in this template, not for packaging to PyPI.

- **Dashboards.** Keep the `dashboards/` directory so dashboards are discovered and integrated automatically. For details, see the [dashboard instructions](./gpt_dashboards_instructions.md).

- **Dependencies (UV recommended).** We use **uv** for dependency and environment management.
  - Declare dependencies in `pyproject.toml`.
  - Commit the `uv.lock` file for reproducible builds.
  - If your CI or the platform requires a `requirements.txt`, generate it **from the lockfile** (do not hand-edit). This keeps one source of truth while still producing a pinned `requirements.txt` when needed.

- **Runtime parity.** Use the provided `Dockerfile` to run the project locally; it matches the image used by the platform to run your jobs, minimizing “works on my machine” drift.



