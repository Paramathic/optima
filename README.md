## Instruction on Setting Up This Repository as Subtree

In your main repo, add the current `pruning` repository and pull as a subtree.

```bash
git remote add pruning https://github.com/Mohammad-Mozaffari/pruning.git
git subtree pull --prefix=pruning/ pruning main
```

## Instruction on Making Changes on Main Repository and This Repository

Commit your changes first, then push to current `pruning` repository.
Next, pull the changes to your main repository to reflect the changes.
Finally, push the reflected changes in `pruning` to your main repository.

```bash
git subtree push --prefix=pruning/ pruning main
git subtree pull --prefix=pruning/ pruning main
git push
```