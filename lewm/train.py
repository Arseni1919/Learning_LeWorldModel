import torch
import gymnasium as gym
import wandb
from lewm.encoder import Encoder
from lewm.decoder import Decoder
from lewm.predictor import Predictor
from lewm.sigreg import SIGReg
from lewm.utils import signed_log
from tqdm import tqdm
from lewm.train_world_model import train_epoch as wm_train_epoch
from lewm.train_decoder import train_epoch as dec_train_epoch
from lewm.planning import cem, evaluate as cem_evaluate
from lewm.params import OBS_DIM, ACTION_DIM, LATENT_DIM


def make_models(device):
    encoder = Encoder(OBS_DIM * 2 + 1, LATENT_DIM).to(device)
    predictor = Predictor(LATENT_DIM, ACTION_DIM).to(device)
    decoder = Decoder(LATENT_DIM, OBS_DIM).to(device)
    sigreg = SIGReg().to(device)
    return encoder, predictor, decoder, sigreg


def collect(env, encoder, predictor, decoder, device, n_steps, use_cem=False):
    data = []
    obs, _ = env.reset()
    prev_obs = obs.copy()
    prev_reward = 0.0
    encoder.eval()
    predictor.eval()
    decoder.eval()
    for _ in tqdm(range(n_steps), desc="collecting", unit="step"):
        if use_cem:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
            prev_obs_t = torch.tensor(prev_obs, dtype=torch.float32, device=device)
            with torch.no_grad():
                action = cem(obs_t, encoder, predictor, decoder,
                             prev_obs=prev_obs_t,
                             prev_reward_log=float(signed_log(prev_reward)))
        else:
            action = env.action_space.sample()
        next_obs, reward, terminated, truncated, _ = env.step(action)
        data.append((prev_obs.copy(), obs.copy(), prev_reward, action,
                     next_obs.copy(), reward, terminated))
        prev_obs = obs.copy()
        prev_reward = reward
        obs = next_obs
        if terminated or truncated:
            obs, _ = env.reset()
            prev_obs = obs.copy()
            prev_reward = 0.0
    return data


def learn_wm(encoder, predictor, decoder, sigreg, data, device, *,
             n_epochs_wm=200, n_epochs_dec=400, batch_size=256, lr=3e-4, lam=1.0):
    wm_optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(predictor.parameters()), lr=lr
    )
    dec_optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
    encoder.train()
    predictor.train()
    for epoch in range(n_epochs_wm):
        stats = wm_train_epoch(encoder, predictor, sigreg, wm_optimizer, data, device, batch_size, lam)
        print(f"\r  wm e{epoch + 1:3d}/{n_epochs_wm} | loss {stats['loss']:.4f}"
              f" | pred {stats['pred_loss']:.4f} | sig {stats['reg_loss']:.4f}"
              f" | var {stats['mean_var']:.3f} | dead {stats['dead_dims']:.0f}", end="")
        wandb.log({f"wm/{k}": v for k, v in stats.items()})
    print()
    encoder.eval()
    predictor.eval()
    decoder.train()
    for epoch in range(n_epochs_dec):
        loss = dec_train_epoch(encoder, predictor, decoder, dec_optimizer, data, device, batch_size)
        print(f"\r  dec e{epoch + 1:3d}/{n_epochs_dec} | dec {loss:.4f}", end="")
        wandb.log({"dec/loss": loss})
    print()


def evaluate(env, encoder, predictor, decoder, device, n_episodes, *, success_threshold=200.0):
    encoder.eval()
    predictor.eval()
    decoder.eval()
    rewards = cem_evaluate(env, encoder, predictor, decoder, n_episodes, device)
    mean_r = sum(rewards) / len(rewards)
    n_success = sum(r >= success_threshold for r in rewards)
    print(f"  eval | mean {mean_r:.1f} | {n_success}/{n_episodes} success "
          f"| {[f'{r:.0f}' for r in rewards]}")
    wandb.log({"eval/mean_reward": mean_r, "eval/n_success": n_success})
    return rewards


def save(encoder, predictor, decoder):
    torch.save({"encoder": encoder.state_dict(), "predictor": predictor.state_dict()},
               "data/checkpoint_final.pt")
    torch.save(decoder.state_dict(), "data/decoder_final.pt")


def main():
    n_collect = 10_000
    n_epochs_wm = 20
    n_epochs_dec = 200
    batch_size = 256
    lr = 3e-4
    lam = 1.0
    n_eval_episodes = 10
    success_threshold = 200.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    wandb.init(project="lewm-lunarlander", config={
        "latent_dim": LATENT_DIM, "lr": lr, "lam": lam,
        "n_collect": n_collect, "n_epochs_wm": n_epochs_wm,
        "n_epochs_dec": n_epochs_dec,
    })
    encoder, predictor, decoder, sigreg = make_models(device)
    train_env = gym.make("LunarLander-v3")
    eval_env = gym.make("LunarLander-v3")
    iteration = 0
    while True:
        iteration += 1
        use_cem = iteration > 1
        print(f"\n=== iteration {iteration} ===")
        print(f"  collecting ({'CEM' if use_cem else 'random'})...")
        data = collect(train_env, encoder, predictor, decoder, device, n_collect, use_cem=use_cem)
        print(f"  collected {len(data)} transitions")
        learn_wm(encoder, predictor, decoder, sigreg, data, device,
                 n_epochs_wm=n_epochs_wm, n_epochs_dec=n_epochs_dec,
                 batch_size=batch_size, lr=lr, lam=lam)
        save(encoder, predictor, decoder)
        rewards = evaluate(eval_env, encoder, predictor, decoder, device, n_eval_episodes,
                           success_threshold=success_threshold)
        if sum(r >= success_threshold for r in rewards) == n_eval_episodes:
            print("solved!")
            break
    wandb.finish()


if __name__ == "__main__":
    main()
