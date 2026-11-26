# Colibri Studio file ownership and permission policy

- Colibri Studio must be launched as the ordinary desktop user. Do not run `app.py` with `sudo`.
- Reads, writes, creates, backups, diffs, renames, moves and deletes execute in the ordinary Studio process. The application never elevates these operations and never retries a failed file operation as root.
- New files and folders inherit the ordinary user's normal `umask`. Existing file mode is preserved when atomically replacing a file where the platform permits it.
- Studio does not automatically run `chmod`, `chown`, recursive metadata changes, or permission normalization. In particular, it never applies `777` or silently changes ownership.
- `inspect_workspace_permissions` reports owner, group, mode and effective access without modifying anything.
- `repair_workspace_metadata` accepts only a single existing, non-symlink path inside the active workspace and only `chmod` or `chown`. It requires `approved=true` and uses `pkexec`/PolicyKit for the narrow metadata command. It never accepts arbitrary shell text and never performs recursive changes.
- If PolicyKit/`pkexec` is unavailable or authorization is declined, no change is made. The user must resolve the issue through Ubuntu's normal administration UI or configure an appropriately restricted authorization policy.
- After a successful metadata repair, retry the original file operation as the ordinary Studio user.
