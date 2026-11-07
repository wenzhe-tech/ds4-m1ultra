#!/bin/bash
# Idempotent fast-forward updater for this checkout.
# Read README.md first. Safe to run repeatedly.
set -euo pipefail

REPO=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
BRANCH=main
# Backups live outside the tree so a fresh clone stays clean. Override with
# DS4_BACKUP_DIR if you want them next to the repo.
BACKUP=${DS4_BACKUP_DIR:-$HOME/.cache/ds4-local/backups}
STAMP=$(date +%Y%m%d-%H%M%S)
PROXY=${DS4_UPDATE_PROXY:-http://127.0.0.1:7897}
FORCE_BUILD=${DS4_FORCE_BUILD:-0}
SYNC_ALL=0
for arg in "$@"; do
    case "$arg" in
        --all-branches) SYNC_ALL=1 ;;
        *) echo "ERROR: unknown option: $arg" >&2; exit 2 ;;
    esac
done

cd "$REPO"
mkdir -p "$BACKUP"

cur=$(git symbolic-ref --short HEAD)
if [ "$cur" != "$BRANCH" ]; then
    echo "ERROR: on branch '$cur', expected '$BRANCH'. Switch manually first." >&2
    exit 1
fi

# GitHub needs the local Clash proxy in the agent shell; fall back to direct.
if nc -z 127.0.0.1 7897 >/dev/null 2>&1; then
    git -c http.proxy="$PROXY" -c https.proxy="$PROXY" fetch --all --prune --tags
else
    echo "[warn] 127.0.0.1:7897 not listening, fetching directly"
    git fetch --all --prune --tags
fi

# 1. Never lose local work. Do NOT auto-replay: upstream rewrites these files.
backed_up=0
if ! git diff --quiet || ! git diff --cached --quiet; then
    patch="$BACKUP/local-changes-$STAMP.patch"
    git diff --binary HEAD > "$patch"
    git stash push -m "ds4-update auto backup $STAMP"
    backed_up=1
    echo "[backup] local changes -> $patch"
    echo "[backup] also in stash; NOT auto-replayed, review manually"
fi

# 2. Fast-forward only. Refuse diverged history.
before=$(git rev-parse HEAD)
after=$(git rev-parse "origin/$BRANCH")
if [ "$before" = "$after" ]; then
    echo "[up to date] $before"
    updated=0
else
    if ! git merge-base --is-ancestor HEAD "origin/$BRANCH"; then
        echo "ERROR: HEAD diverged from origin/$BRANCH; refusing non-fast-forward update." >&2
        if [ "$backed_up" = "1" ]; then
            echo "Local changes are still in the stash." >&2
        fi
        exit 1
    fi
    echo "[update] $(git rev-list --count "HEAD..origin/$BRANCH") commits: $before -> $after"
    git merge --ff-only "origin/$BRANCH"
    updated=1
fi

# 2b. Optional: align every local branch with its upstream counterpart.
#     Safe by construction: create missing branches, fast-forward existing
#     ones, and SKIP (never drop) any branch with local-only commits.
#     NOTE: refs/remotes/origin/HEAD is a symref that shows up as bare
#     "origin" with %(refname:short) -- the '^origin/' filter excludes it,
#     otherwise a bogus local branch named "origin" gets created.
if [ "$SYNC_ALL" = "1" ]; then
    echo "[branches] aligning local branches with origin/*"
    for ref in $(git for-each-ref --format='%(refname:short)' refs/remotes/origin/ | grep '^origin/' | grep -v '/HEAD$'); do
        name=${ref#origin/}
        if [ "$name" = "$BRANCH" ]; then
            continue
        fi
        if git show-ref --verify --quiet "refs/heads/$name"; then
            if git merge-base --is-ancestor "$name" "$ref"; then
                git branch -f "$name" "$ref" >/dev/null
                echo "  aligned  $name -> $(git rev-parse --short "$ref")"
            else
                echo "  SKIP     $name (diverged: local-only commits, refusing to drop)" >&2
            fi
        else
            git branch --track "$name" "$ref" >/dev/null 2>&1
            echo "  created  $name -> $(git rev-parse --short "$ref")"
        fi
    done
fi

# 3. Full rebuild. Makefile has no header dependency tracking: a plain
#    `make` keeps stale .o files built against the old ds4_gpu.h.
need_build=$updated
if [ "$FORCE_BUILD" = "1" ]; then
    need_build=1
fi
for b in ds4 ds4-server ds4-bench ds4-eval ds4-agent; do
    if [ ! -x "$REPO/$b" ]; then
        need_build=1
    fi
done
if [ "$need_build" = "1" ]; then
    log="$BACKUP/make-$STAMP.log"
    echo "[build] make clean && make -j4 -> $log"
    make clean >/dev/null 2>&1
    if ! make -j4 > "$log" 2>&1; then
        echo "ERROR: build failed, see $log" >&2
        tail -30 "$log" >&2
        exit 1
    fi
    echo "[build] ok"
else
    echo "[build] skipped (up to date, all binaries present; DS4_FORCE_BUILD=1 to force)"
fi

# 4. Smoke test only. Never load a model here (see AGENT.md safety rules).
if ! ./ds4 --help >/dev/null 2>&1; then
    echo "ERROR: ./ds4 --help failed" >&2
    exit 1
fi
echo "[smoke] ./ds4 --help ok"

echo
echo "--- result ---"
git status --short --branch
git log -1 --format='%h %ci %s'
echo "behind: $(git rev-list --count "HEAD..origin/$BRANCH")"
echo "stash:"
git stash list
if [ "$SYNC_ALL" = "1" ]; then
    echo "branches:"
    git branch -vv
fi
if [ "$backed_up" = "1" ]; then
    echo
    echo "!! Local changes were backed up and NOT reapplied."
    echo "!! See README.md before deciding whether to port them."
fi
