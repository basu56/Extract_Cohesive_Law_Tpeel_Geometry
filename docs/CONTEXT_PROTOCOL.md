# Context protocol — keeping Claude Code and Claude Web in sync

GitHub (`basu56/Extract_Cohesive_Law_Tpeel_Geometry`) is the only source of truth.
Neither Claude environment remembers the other's conversations; whatever is not in
this repository does not exist for the next session.

## Files and their roles

| File | Role | Who edits | Style |
|---|---|---|---|
| `CLAUDE.md` | Rules, current phase, cluster facts | both, rarely | short; loaded automatically by Claude Code |
| `docs/RESEARCH_PLAN.md` | Formulation, hypotheses, phases | both, on agreement | stable; changes logged in DECISIONS |
| `docs/DECISIONS.md` | Append-only decision log (D1, D2, …) | both | never rewrite old entries; supersede with a new one |
| `docs/STATUS.md` | **Handoff page**: where we are, next tasks, open questions | both, every session | overwrite freely; keep under ~1 page |
| `docs/sessions/YYYY-MM-DD_<code\|web>_<topic>.md` | One log per session | the session that ran | never edited afterwards |
| `docs/EXISTING_MODEL.md` | Review of the inherited ABAQUS model | reference | |

## Session start (both environments)

1. Get the latest repo: Claude Code runs `git pull`; Claude Web reads the synced repo in
   the Claude Project (re-sync the GitHub source if it is stale) or the user pastes
   the files.
2. Read `CLAUDE.md` → `docs/STATUS.md` → newest 1–3 `docs/sessions/` logs →
   `docs/DECISIONS.md` (and `RESEARCH_PLAN.md` before any methods discussion).
3. If STATUS and the newest session log disagree, the session log is newer; ask the user.

## Session end

1. Write `docs/sessions/<date>_<env>_<topic>.md` using the template below.
2. Update `docs/STATUS.md` (current state, next actions, open questions).
3. Append any new or changed decision to `docs/DECISIONS.md` with status
   `proposed` / `agreed` / `superseded by Dn`.
4. Commit and push:
   - **Claude Code**: commits and pushes directly (SSH remote), reports the commit hash.
   - **Claude Web**: cannot push. It outputs one bash script with heredocs that writes the
     files and commits them (the pattern of `add_project_context.sh`). Use the **SSH**
     URL `git@github.com:basu56/Extract_Cohesive_Law_Tpeel_Geometry.git`; HTTPS has no
     credentials on the cluster. The user runs it on Bell or asks Claude Code to apply it.
     Scripts should *append* to DECISIONS.md (`cat >>`), not overwrite it.
5. Conflicting edits: pull first; never force-push; if two sessions changed STATUS.md,
   merge by hand and keep both sets of information.

## Session-log template

```markdown
# <date> — <Claude Code | Claude Web> — <topic>

## Context / goal
## What was discussed or done
## Results (label [D]/[H]/[I]; include run IDs, commit hashes, file paths)
## Decisions (reference Dn in DECISIONS.md)
## Failed / aborted / not done
## Next steps and open questions
```

## What does not go here

Raw ABAQUS output (scratch only, see `.gitignore`), credentials, and unpublished
experimental data unless the user approves.
