# Vocalis production preparation

This document is the deployment plan. It does not deploy Vocalis.

VM 105 (`192.168.0.110`) was not contacted. Nothing under `/opt/voxcpm2` or `/opt/vocalis` was created or changed. `vocalis.service` is not installed. The VoxCPM2 weights were not loaded. Phase 6 added an adapter and tested it with a stand-in of the 2.0.3 API. A successful mock workflow does not show that real inference works. Loading the model and generating speech on the VM is Phase 9 and needs a separate approval.

## Target layout

| Path | Role |
|---|---|
| `/opt/vocalis` | Application source. Not a virtual environment. |
| `/opt/vocalis/vocalis.env` | Production settings, from `deploy/vocalis.env.example`. |
| `/opt/vocalis/data/history.db` | SQLite history. Starts empty. |
| `/opt/vocalis/uploads/references` | Persistent copies of validated reference audio. |
| `/opt/vocalis/logs/vocalis.log` | Rotating application log, 1 MiB, five backups. |
| `/opt/voxcpm2/venv` | Existing Python 3.11 environment. Reuse it. |
| `/opt/voxcpm2/models` | Existing VoxCPM2 weights. Read them. Do not download or replace them. |
| `/opt/voxcpm2/output` | Existing output directory. Vocalis adds `design/` and `clone/` underneath. |
| `/opt/voxcpm2/output/test-tagalog.wav` | Existing test file. Leave it in place. It is not a Vocalis history row. |

The service account in `deploy/vocalis.service` is `ryan`, which is the account named for this host. Phase 9 has to confirm that account exists and can read the model directory and write the Vocalis data, upload, log, and output directories. Do not change ownership of `/opt/voxcpm2/models` or the existing output file.

`APP_HOST` stays `127.0.0.1`. The studio is not bound to the LAN, and Cloudflare Tunnel is not part of this plan.

## Dependency strategy

Local development installs `requirements.txt`. That file is a freeze taken on macOS arm64. It contains Gradio and its dependencies. It does not contain VoxCPM2 or PyTorch, and it must not be installed into `/opt/voxcpm2/venv`. Doing so could replace `torch==2.5.1+cpu` or `voxcpm==2.0.3`.

`requirements-prod.txt` lists only the three application packages:

- `gradio==6.32.0`
- `python-dotenv==1.2.4`
- `psutil==7.1.3`

Those packages are not installed on the VM yet. Their Linux wheels were not resolved here. During an approved deployment, dry-run the install inside the existing virtual environment and stop if pip proposes a change to `torch` or `voxcpm`:

```bash
/opt/voxcpm2/venv/bin/python -m pip install --dry-run -r /opt/vocalis/requirements-prod.txt
```

Install only after that review. Do not reinstall PyTorch and do not download the model.

## Files that stay on the development machine

The local `data/`, `output/`, `uploads/`, and `logs/` directories are development artifacts. `deploy/rsync-excludes.txt` lists them so a later copy of the source does not carry them to the VM.

The local database `data/history.db` has one row, job `f6cbad990f77441a9bf5a88af7eee723`, text `Phase seven check.`, status succeeded, `simulated` set. Its audio is:

```text
output/design/mock-design-62094d0f41da4e129e7158e14120be2e.wav
```

A second tone remains from an earlier manual check, and its history row was already deleted:

```text
output/design/mock-design-6d2a33138d4b492a851309ba4d44de6c.wav
```

Both names start with `mock-`. The remaining history row is marked simulated. These files are not VoxCPM2 speech and are not production history. Production starts with an empty `history.db`. Do not copy either WAV or the local database onto the VM.

## Proposed deployment sequence

Run this only after Phase 9 is approved. Inspect the VM first and stop if the Python, `voxcpm`, or `torch` versions differ from the ones above, or if `test-tagalog.wav` is missing.

1. Confirm the `ryan` account, free disk, and free memory.
2. Create `/opt/vocalis` without modifying `/opt/voxcpm2/venv`, `/opt/voxcpm2/models`, or existing files in `/opt/voxcpm2/output`.
3. Copy the source with the exclude file. From a clean checkout:

```bash
rsync -a --exclude-from=deploy/rsync-excludes.txt ./ ryan@192.168.0.110:/opt/vocalis/
```

4. Install `deploy/vocalis.env.example` as `/opt/vocalis/vocalis.env`. Do not copy the development `.env`.
5. Dry-run, then install `requirements-prod.txt` into `/opt/voxcpm2/venv`.
6. Run the health check below before enabling the service.
7. Install `deploy/vocalis.service` to `/etc/systemd/system/vocalis.service`, then `systemctl daemon-reload` and `systemctl enable --now vocalis`.
8. Generate one short voice-design sample and one cloning sample. Confirm the history row, the WAV under `/opt/voxcpm2/output/design` or `clone`, and that `test-tagalog.wav` is unchanged.
9. Restart the service and confirm the new rows are still present and no row is left `queued` or `running`.

## Process control

`deploy/vocalis.service` starts `/opt/voxcpm2/venv/bin/python /opt/vocalis/app.py` as `ryan`, after the network is online, with `Restart=on-failure`. It is not installed.

A service stop calls the application's shutdown and waits up to five minutes for the generation that is already running. Jobs that are only waiting are cancelled. `TimeoutStopSec=330` gives systemd a margin past that wait. If a generation is still running when systemd kills the process, the next start marks that row failed with `Generation interrupted because Vocalis restarted before completion.` The partial result is not reported as success.

The in-memory queue keeps the newest 100 jobs (`DEFAULT_CACHED_JOBS`). Queued and running jobs stay in memory even when that number is already reached. Older finished jobs leave memory and remain in SQLite. The history page reads SQLite.

Voice cloning copies a validated upload into `uploads/references` under a new `reference-` name. The original file is only read. History stores the copy.

## Health checks

On the VM, before enabling the service:

```bash
cd /opt/vocalis
/opt/voxcpm2/venv/bin/python app.py --check
```

With `MODEL_PROVIDER=voxcpm`, `--check` loads the model, confirms startup, and shuts the engine down. It does not generate speech. It is the configuration and load check, not the inference test from step 8 above.

After the service is running:

```bash
systemctl is-active vocalis
curl -fsS -o /dev/null -w '%{http_code}\n' http://127.0.0.1:7860/
```

The HTTP check expects `200`. It does not prove that a generation succeeded.

## Backup

Stop Vocalis, or use SQLite's backup API, before copying the database. The application uses WAL mode, so copying `history.db` alone while the service is running can miss committed rows.

```bash
sqlite3 /opt/vocalis/data/history.db ".backup '/opt/vocalis/data/history-backup.db'"
```

Back up these directories together with that database file:

- `/opt/vocalis/uploads/references`
- `/opt/voxcpm2/output/design`
- `/opt/voxcpm2/output/clone`

Keep `/opt/voxcpm2/models` and `/opt/voxcpm2/output/test-tagalog.wav` outside the Vocalis backup. They are existing host data. Do not delete them as part of a restore, and do not restore a development `history.db` over the production file.

History deletion in the studio removes a row and leaves the audio files on disk. Restoring a row does not require recreating a file that is still present.

## Maintenance

Logs rotate inside the process. `journalctl -u vocalis` shows the same stream the process writes to the console. Prompts, voice descriptions, and transcripts are not written to the log.

`MAX_QUEUE_SIZE` is the number of jobs allowed to wait. One generation runs at a time. Raising the pending limit does not run jobs in parallel.

A reference upload has to be WAV, FLAC, or MP3, and it has to be within `MAX_UPLOAD_BYTES` (20 MiB unless the env file changes it). The stored copy is the file history keeps. Removing the original upload does not remove that copy.

An upgrade replaces the source under `/opt/vocalis` and restarts the service. It does not reinstall the model environment unless a reviewed change to `requirements-prod.txt` says so.
