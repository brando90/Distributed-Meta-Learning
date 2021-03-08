# Distributed-Meta-Learning

The goal is to write distributed pytorch code for episodic meta-learning.
In particular the tasks in a meta-batch are required to be processed in parallel due to the meta-adaptation using the support set and it's not easy to batch.
Instead this repo plans to show case three cases:

- distributed MAML with torchmeta & higher (custom DDP with cherry since DDP doesn't work with those libraries as of now)
- distributed MAML with learn2learn using DDP
- distributed MAML completely custom without cherry

## TODO
- distributed MAML with learn2learn
