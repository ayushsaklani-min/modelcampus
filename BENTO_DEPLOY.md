## BentoCloud Deployment Checklist

Follow these steps any time you build and push a new Bento from Windows.

1. **Install toolchain**
   ```powershell
   pip install bentoml
   ```
   (Optional) add Bento CLI completion with `bentoml --help`.

2. **Wire up Git hooks (one-time)**
   ```powershell
   git config core.hooksPath .githooks
   ```
   The pre-push hook automatically normalizes Bento install scripts so they do not ship with CRLF line endings.

3. **Build and normalize**
   ```powershell
   bentoml build
   python scripts/normalize_bento_install.py
   ```
   The helper script scans `~/bentoml/bentos/**/env/python/install.sh` and removes `\r` characters that break `set -euo pipefail` on Linux builders.

4. **Push Bento to BentoCloud**
   ```powershell
   bentoml push failure_predictor:latest
   ```

5. **Deploy**
   - Open the BentoCloud console → Deployments → New Deployment.
   - Select `failure_predictor:latest`.
   - Choose a CPU cluster with at least 1 GB RAM (model artifact is ~430 MB).
   - Enable auto-scaling if desired.

6. **Smoke test**
   ```bash
   curl -X POST "https://<deployment>.bentoml.app/predict" \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer <API_TOKEN>" \
     -d @sample_payload.json
   ```

If you ever skip step 3, BentoCloud will rebuild an image that fails with
`set: pipefail: invalid option name` because `install.sh` was produced with CRLF
line endings on Windows. Always run the normalization step (the git hook will
try to do it automatically on every `git push`).


