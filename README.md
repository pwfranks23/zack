## Zack

Why would you need Isaac when you've got Zack?
This is a low-effort side project to spend some time with RL and tools that are common in industry that I don't use in my google corp stack. 

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
