import torch
import gymnasium as gym
import wandb
from stable_baselines3 import PPO
from lewm.encoder import Encoder
from lewm.decoder import Decoder
from lewm.predictor import Predictor
from lewm.sigreg import SIGReg
from lewm.utils import signed_log
from lewm.train_world_model import train_epoch as wm_train_epoch
from lewm.train_decoder import train_epoch as dec_train_epoch
from lewm.train_latent_ppo import LatentEnv, evaluate_in_real_env, make_enc_in


OBS_DIM = 8
ACTION_DIM = 4
LATENT_DIM = 16


def make_models(device):
    encoder = Encoder(OBS_DIM * 2 + 1, LATENT_DIM).to(device)
    predictor = Predictor(LATENT_DIM, ACTION_DIM).to(device)
    decoder = Decoder(LATENT_DIM, OBS_DIM).to(device)
    sigreg = SIGReg().to(device)
    return encoder, predictor, decoder, sigreg


def collect(env, encoder, policy, device, n_steps):
    data = []
    obs, _ = env.reset()
    prev_obs = obs.copy()
    prev_reward = 0.0
    encoder.eval()
    for _ in range(n_steps):
        if policy is None:
            action = env.action_space.sample()
        else:
            enc_in = make_enc_in(prev_obs, obs, signed_log(prev_reward)).to(device)
            with torch.no_grad():
                z = encoder(enc_in).squeeze(0)
            action, _ = policy.predict(z.cpu().numpy(), deterministic=False)
            action = int(action)
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
    decoder.train()
    for epoch in range(n_epochs_dec):
        loss = dec_train_epoch(encoder, decoder, dec_optimizer, data, device, batch_size)
        print(f"\r  dec e{epoch + 1:3d}/{n_epochs_dec} | dec {loss:.4f}", end="")
        wandb.log({"dec/loss": loss})
    print()


def learn_policy(encoder, predictor, decoder, real_env, policy, device, *,
                 n_timesteps=50_000, max_steps=500):
    latent_env = LatentEnv(encoder, predictor, decoder, real_env, max_steps=max_steps)
    if policy is None:
        policy = PPO("MlpPolicy", latent_env, verbose=1, device=str(device))
    else:
        policy.set_env(latent_env)
    policy.learn(total_timesteps=n_timesteps, reset_num_timesteps=False)
    return policy


def evaluate(encoder, policy, device, n_episodes, *, success_threshold=200.0):
    rewards = evaluate_in_real_env(policy, encoder, device, n_episodes, render=False)
    mean_r = sum(rewards) / len(rewards)
    n_success = sum(r >= success_threshold for r in rewards)
    print(f"  eval | mean {mean_r:.1f} | {n_success}/{n_episodes} success "
          f"| {[f'{r:.0f}' for r in rewards]}")
    wandb.log({"eval/mean_reward": mean_r, "eval/n_success": n_success})
    return rewards


def save(encoder, predictor, decoder, policy):
    torch.save({"encoder": encoder.state_dict(), "predictor": predictor.state_dict()},
               "data/checkpoint_final.pt")
    torch.save(decoder.state_dict(), "data/decoder_final.pt")
    policy.save("data/ppo_latent")


def main():
    n_collect = 10_000
    n_epochs_wm = 200
    n_epochs_dec = 400
    batch_size = 256
    lr = 3e-4
    lam = 1.0
    n_timesteps_ppo = 50_000
    max_steps = 500
    n_eval_episodes = 10
    success_threshold = 200.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    wandb.init(project="lewm-lunarlander", config={
        "latent_dim": LATENT_DIM, "lr": lr, "lam": lam,
        "n_collect": n_collect, "n_epochs_wm": n_epochs_wm,
        "n_epochs_dec": n_epochs_dec, "n_timesteps_ppo": n_timesteps_ppo,
    })
    encoder, predictor, decoder, sigreg = make_models(device)
    real_env = gym.make("LunarLander-v3")
    policy = None
    iteration = 0
    while True:
        iteration += 1
        print(f"\n=== iteration {iteration} ===")
        print(f"  collecting ({'random' if policy is None else 'PPO'})...")
        data = collect(real_env, encoder, policy, device, n_collect)
        print(f"  collected {len(data)} transitions")
        learn_wm(encoder, predictor, decoder, sigreg, data, device,
                 n_epochs_wm=n_epochs_wm, n_epochs_dec=n_epochs_dec,
                 batch_size=batch_size, lr=lr, lam=lam)
        policy = learn_policy(encoder, predictor, decoder, real_env, policy, device,
                              n_timesteps=n_timesteps_ppo, max_steps=max_steps)
        save(encoder, predictor, decoder, policy)
        rewards = evaluate(encoder, policy, device, n_eval_episodes,
                           success_threshold=success_threshold)
        if sum(r >= success_threshold for r in rewards) == n_eval_episodes:
            print("solved!")
            break
    wandb.finish()


if __name__ == "__main__":
    main()
