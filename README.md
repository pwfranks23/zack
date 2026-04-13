## Zack

Why would you need Isaac when you've got Zack?
This is a low-effort side project, adjust expectations accordingly. 

## Configuring Local Environment
I was using python 3.12.3.
Setting up local environment:
1. Create venv with `python3 -m venv ~/venv/zack`
2. Source venv with `source ~/venv/zack/bin/activate`
3. Install requirements with `pip install -r requirements.txt`

## Tracking Experiments Locally
Training logs metrics to MLflow in the local `mlruns/` directory.

- Start training with `python3 -m training`
- Open the local MLflow UI with `mlflow ui --backend-store-uri ./mlruns`
- Then browse to http://127.0.0.1:5000

Each training run logs:
- parameters from `TrainingConfig`
- per-episode metrics such as reward, loss, and running average reward
- periodic evaluation metrics
- checkpoints as artifacts

The default experiment name is `cartpole-local`.

## Web Demo (GitHub Pages)

The `docs/` folder contains a browser-based CartPole demo that uses:
- **Pyodide** — real CPython + NumPy in WebAssembly, running the actual Gymnasium physics
- **ONNX Runtime Web** — runs the exported trained policy in the browser, no server needed

**One-time setup after training:**
1. Export the trained model to ONNX: `python3 export_model.py`
2. Commit `docs/model.onnx` to the repo
3. In GitHub repo Settings → Pages, set source to **Deploy from branch**, branch `main`, folder `/docs`
4. Visit `https://<username>.github.io/<repo>/`

The demo runs entirely client-side. First load downloads ~15 MB (Pyodide runtime + gymnasium).
Click left or right on the canvas to disturb the pole and watch the agent recover.
