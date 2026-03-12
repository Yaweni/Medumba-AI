# Medumba AI

Repository for Medumba translation and training tools.

Contents:
- training scripts and notebooks
- server proxy and site files
- tooling for preparing corpora and dictionaries

Note: Large model checkpoints and heavy artifacts are excluded via `.gitignore` (check `outputs/` exclusions).

To publish this repo to GitHub (local machine):

1. Ensure `git` and `gh` (GitHub CLI) are installed and authenticated.
2. From the repository root run:

```bash
git init
git add -A
git commit -m "Initial commit: Medumba AI"
# create public repo and push (automatic if you have gh auth)
gh repo create Medumba-AI --public --source=. --remote=origin --push
```

If `gh` is not available, create a repo on GitHub and follow its instructions to add `origin` and push.
