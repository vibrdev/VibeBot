# Working in this repo

## Git

- `main` is the trunk. All work merges into `main` — branch off it, open a PR
  against it, merge it. Do not leave finished work parked on a branch.

## Conventions

- Defaults must stay free and local. Anything that costs money or calls out to a
  third party is opt-in via `config.yaml`, never the default.
- Claims about model behaviour (latency, accuracy, what a field means) belong in
  the code only after they have been measured against the real model. If a
  vendor's README says one thing and a run says another, the run wins and the
  discrepancy goes in a comment.
