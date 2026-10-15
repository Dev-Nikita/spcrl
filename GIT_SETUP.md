# Creating the GitHub repository and pushing this folder

Everything below is run from inside this folder on your Mac:

```bash
cd ~/"Science Article 2025-/Structural Priors Enable Generalizable Causal Diagnosis and Counterfactual Recovery in Dynamic Systems/spcrl-public"
```

## 0. One-time: who the commits belong to

```bash
git config --global user.name  "Nikita Tarasov"
git config --global user.email "dev.nikita@outlook.com"
```

(Use `git config` without `--global` if you want this only for this repository.)

## 1. Create the repository on GitHub

**Make it private for now.** The manuscript is under review; a public
repository exposes the full experimental design and the results before the paper
is out. Flip it to public on the day of acceptance — Zenodo only needs it public
at the moment you create the release.

Open <https://github.com/new> and create:

- Owner: `Dev-Nikita`
- Repository name: `spcrl`
- Visibility: **Private**
- Do **not** add a README, .gitignore or licence — this folder already has them.

(If you have the GitHub CLI installed, `gh repo create Dev-Nikita/spcrl --private --source=. --remote=origin` does the same thing and sets the remote for you — then skip step 3.)

## 2. Initialise and make the first commit

```bash
git init
git branch -M main
git add .
git status                 # read this before committing: nothing from the
                           # manuscript, the cover letter or docs/sections
                           # should appear in the list
git commit -m "SP-CRL: structural priors for causal learning under distribution shift"
```

## 3. Connect the remote and push

```bash
git remote add origin https://github.com/Dev-Nikita/spcrl.git
git push -u origin main
```

If GitHub asks for a password, it wants a personal access token, not your
account password: <https://github.com/settings/tokens> → "Generate new token
(classic)" → scope `repo`. Paste the token as the password.

## 4. Later changes

```bash
git add -A
git commit -m "what changed"
git push
```

## 5. At acceptance: the release and the DOI

```bash
# make the repository public in Settings → General → Danger Zone
git tag -a v1.0 -m "Version accompanying the published paper"
git push origin v1.0
```

Then on Zenodo: log in with GitHub, Settings → GitHub, switch `Dev-Nikita/spcrl`
on, and create a GitHub **Release** from the `v1.0` tag. Zenodo archives the
repository at that tag and issues a DOI. Put that DOI in the manuscript's Data
availability statement.

## What is deliberately not in this repository

The `.gitignore` keeps out the manuscript and everything carrying its text:
`manuscript/`, `docs/sections/`, `docs/build/`, `docs/submission/`,
`docs/reference/`, any `.tex`, `.bib`, `.cls`, `.bst`, any `cover_letter*`, any
PDF, and the figures. Those stay in `submission/` next to this folder, outside
git. If you ever copy the paper into this folder by accident, git will ignore
it — but check `git status` before each commit anyway.

What *is* published: the library, the experiments, the analysis, the testbed
harnesses, the pre-registrations, the claim ledger, the external campaign
telemetry and the result records — everything a reviewer or a reader needs to
rerun the work and recompute every number, and nothing that lets someone rewrite
the paper before you publish it.
