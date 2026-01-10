import os
import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.utils import set_random_seed

# Custom libraries
from utils.envs import make_pybullet_env

torch.set_float32_matmul_precision('high')


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    # -------------------------
    # Reproducibility
    # -------------------------
    set_random_seed(cfg.envs.seed)
    torch.set_float32_matmul_precision("high")

    # -------------------------
    # Environments
    # -------------------------
    train_env = make_pybullet_env(cfg.envs, eval=False)
    eval_env = make_pybullet_env(cfg.envs, eval=True)

    # -------------------------
    # Logging dirs
    # -------------------------
    log_dir = os.path.join("logs", cfg.name)
    os.makedirs(log_dir, exist_ok=True)

    # -------------------------
    # Model
    # -------------------------
    model = PPO(
        policy=cfg.algorithms.policy,
        env=train_env,
        learning_rate=cfg.algorithms.learning_rate,
        n_steps=cfg.algorithms.n_steps,
        batch_size=cfg.algorithms.batch_size,
        n_epochs=cfg.algorithms.n_epochs,
        gamma=cfg.algorithms.gamma,
        gae_lambda=cfg.algorithms.gae_lambda,
        clip_range=cfg.algorithms.clip_range,
        ent_coef=cfg.algorithms.ent_coef,
        vf_coef=cfg.algorithms.vf_coef,
        max_grad_norm=cfg.algorithms.max_grad_norm,
        verbose=1,
        tensorboard_log=log_dir,
    )

    # -------------------------
    # Evaluation callback
    # -------------------------
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=log_dir,
        log_path=log_dir,
        eval_freq=10_000,
        deterministic=True,
    )

    # -------------------------
    # Train
    # -------------------------
    model.learn(
        total_timesteps=cfg.algorithms.total_timesteps,
        callback=eval_callback,
    )

    # -------------------------
    # Save final model
    # -------------------------
    model.save(os.path.join(log_dir, "final_model.zip"))

    print("Training complete")


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    main()