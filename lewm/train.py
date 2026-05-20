import torch
import gymnasium as gym
import wandb
from lewm.encoder import Encoder
from lewm.predictor import Predictor
from lewm.reward_predictor import RewardPredictor
from lewm.sigreg import SIGReg
from lewm.utils import collect_data
from lewm.planning import cem, evaluate, demo
from lewm.train_world_model import train_epoch as wm_train_epoch
from lewm.train_reward_predictor import train_epoch as rp_train_epoch


OBS_DIM = 8
ACTION_DIM = 4
LATENT_DIM = 16
N_COLLECT = 1_000
BATCH_SIZE = 256
LR = 3e-4
LAMBDA = 1.0
N_EPOCHS_WM = 200
N_EPOCHS_RP = 400
N_ITERATIONS = 50
EVAL_EVERY = 5
N_EVAL_EPISODES = 10
SUCCESS_THRESHOLD = 200.0


def make_models(device):
    encoder = Encoder(OBS_DIM, LATENT_DIM).to(device)
    predictor = Predictor(LATENT_DIM, ACTION_DIM).to(device)
    reward_predictor = RewardPredictor(LATENT_DIM, ACTION_DIM).to(device)
    sigreg = SIGReg().to(device)
    wm_optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(predictor.parameters()), lr=LR
    )
    rp_optimizer = torch.optim.Adam(reward_predictor.parameters(), lr=LR)
    return encoder, predictor, reward_predictor, sigreg, wm_optimizer, rp_optimizer


def make_cem_fn(encoder, predictor, reward_predictor, device):
    def cem_fn(obs):
        obs_t = torch.tensor(obs, dtype=torch.float32).to(device)
        with torch.no_grad():
            return cem(obs_t, encoder, predictor, reward_predictor)
    return cem_fn


def run_collect(env, iteration, encoder, predictor, reward_predictor, device):
    fn = None if iteration == 0 else make_cem_fn(encoder, predictor, reward_predictor, device)
    label = "random" if iteration == 0 else "CEM"
    print(f"  collecting ({label})...")
    return collect_data(env, N_COLLECT, sample_action_fn=fn)


def run_wm_training(encoder, predictor, sigreg, optimizer, data, device):
    encoder.train()
    predictor.train()
    for epoch in range(N_EPOCHS_WM):
        stats = wm_train_epoch(encoder, predictor, sigreg, optimizer, data, device, BATCH_SIZE, LAMBDA)
        print(f"\r  wm e{epoch + 1:3d}/{N_EPOCHS_WM} | loss {stats['loss']:.4f}"
              f" | pred {stats['pred_loss']:.4f} | sig {stats['reg_loss']:.4f}"
              f" | var {stats['mean_var']:.3f} | dead {stats['dead_dims']:.0f}", end="")
        wandb.log({f"wm/{k}": v for k, v in stats.items()})
    print()


def run_rp_training(encoder, reward_predictor, optimizer, data, device):
    encoder.eval()
    reward_predictor.train()
    for epoch in range(N_EPOCHS_RP):
        loss = rp_train_epoch(encoder, reward_predictor, optimizer, data, device, BATCH_SIZE)
        print(f"\r  rp e{epoch + 1:3d}/{N_EPOCHS_RP} | rew {loss:.4f}", end="")
        wandb.log({"rp/loss": loss})
    print()


def run_eval(eval_env, encoder, predictor, reward_predictor, device):
    encoder.eval()
    predictor.eval()
    reward_predictor.eval()
    rewards = evaluate(eval_env, encoder, predictor, reward_predictor, N_EVAL_EPISODES, device)
    mean_r = sum(rewards) / len(rewards)
    n_success = sum(r >= SUCCESS_THRESHOLD for r in rewards)
    print(f"  eval | mean {mean_r:.1f} | {n_success}/{N_EVAL_EPISODES} success "
          f"| {[f'{r:.0f}' for r in rewards]}")
    wandb.log({"eval/mean_reward": mean_r, "eval/n_success": n_success})
    return n_success == N_EVAL_EPISODES


def save(encoder, predictor, reward_predictor):
    torch.save({"encoder": encoder.state_dict(), "predictor": predictor.state_dict()},
               "data/checkpoint_final.pt")
    torch.save(reward_predictor.state_dict(), "data/reward_predictor_final.pt")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    wandb.init(project="lewm-lunarlander", config={
        "latent_dim": LATENT_DIM, "lr": LR, "lambda": LAMBDA,
        "n_collect": N_COLLECT, "n_epochs_wm": N_EPOCHS_WM, "n_epochs_rp": N_EPOCHS_RP,
    })
    encoder, predictor, reward_predictor, sigreg, wm_opt, rp_opt = make_models(device)
    train_env = gym.make("LunarLander-v3")
    eval_env = gym.make("LunarLander-v3")
    for iteration in range(N_ITERATIONS):
        print(f"\n=== iteration {iteration + 1}/{N_ITERATIONS} ===")
        data = run_collect(train_env, iteration, encoder, predictor, reward_predictor, device)
        run_wm_training(encoder, predictor, sigreg, wm_opt, data, device)
        run_rp_training(encoder, reward_predictor, rp_opt, data, device)
        if (iteration + 1) % EVAL_EVERY == 0:
            save(encoder, predictor, reward_predictor)
            demo(encoder, predictor, reward_predictor, device)
            if run_eval(eval_env, encoder, predictor, reward_predictor, device):
                print("solved!")
                break
    wandb.finish()


if __name__ == "__main__":
    main()
