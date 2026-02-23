import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


class ProbCQL(nn.Module):

    def __init__(self, cnn, action_dim: int = 6,
                 alpha: float = 0.2, beta_kl: float = 1e-3):
        super().__init__()

        self.action_dim = action_dim
        self.alpha      = alpha      # CQL regularisation weight
        self.beta_kl    = beta_kl   # KL / Information-Bottleneck weight

        # Frozen CNN backbone (outputs 3136-dim feature)
        self.cnn = cnn
        for p in self.cnn.parameters():
            p.requires_grad = False
        print("✓ ProbCQL CNN: all conv layers frozen")

        # Probabilistic latent head: two parallel projections from 3136 → 512
        self.fc_mu     = nn.Linear(3136, 512)
        self.fc_logvar = nn.Linear(3136, 512)

        # Q-head: 512-dim latent z → Q(s, ·)
        self.q_head = nn.Linear(512, action_dim)

        # Target networks: fc_mu_target + q_head_target form a stable bootstrap
        # target.  fc_mu changes every gradient step, so it MUST have its own
        # frozen copy — otherwise the "stable" target collapses to tracking the
        # live encoder, defeating the purpose of a target network entirely.
        self.fc_mu_target     = copy.deepcopy(self.fc_mu)
        self.q_head_target    = copy.deepcopy(self.q_head)
        for p in self.fc_mu_target.parameters():
            p.requires_grad = False
        for p in self.q_head_target.parameters():
            p.requires_grad = False

        self.training_step = 0

    # ------------------------------------------------------------------
    # Building blocks
    # ------------------------------------------------------------------

    def encode(self, state: torch.Tensor):
        feat   = self.cnn(state)                # (B, 3136)
        mu     = self.fc_mu(feat)               # (B, 512)
        logvar = self.fc_logvar(feat).clamp(-10.0, 2.0)  # (B, 512)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        if self.training:
            std = (0.5 * logvar).exp()
            return mu + std * torch.randn_like(std)
        return mu

    def forward(self, state: torch.Tensor):
        mu, logvar = self.encode(state)
        z          = self.reparameterize(mu, logvar)
        q_values   = self.q_head(z)
        return q_values, mu, logvar

    def get_action(self, state: torch.Tensor, deterministic: bool = True):
        if state.dim() == 3:
            state = state.unsqueeze(0)
        with torch.no_grad():
            q_values, _, _ = self.forward(state)
        action = q_values.argmax(dim=-1)
        return action.item() if action.numel() == 1 else action

    def get_uncertainty(self, state: torch.Tensor):
        """
        Extract latent mean, variance, and scalar uncertainty for fMRI regression.

        Parameters
        ----------
        state : (B, 4, 84, 84) or (4, 84, 84) uint8/float

        Returns
        -------
        mu       : (B, 512)  – latent mean vector  [use as fMRI regressor]
        sigma_sq : (B, 512)  – latent variance = exp(logvar)
        mean_var : (B,)      – scalar uncertainty per sample (mean over 512)
        """
        if state.dim() == 3:
            state = state.unsqueeze(0)
        if state.dtype == torch.uint8:
            state = state.float() / 255.0
        with torch.no_grad():
            mu, logvar = self.encode(state)
        sigma_sq = logvar.exp()
        mean_var = sigma_sq.mean(dim=-1)
        return mu, sigma_sq, mean_var

    def update_target(self):
        self.fc_mu_target.load_state_dict(self.fc_mu.state_dict())
        self.q_head_target.load_state_dict(self.q_head.state_dict())


# ---------------------------------------------------------------------------
# KL helper
# ---------------------------------------------------------------------------

def _kl_standard_normal(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    kl = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())  # (B, 512)
    return kl.sum(dim=-1).mean()                             # scalar


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def train_prob_cql(model: ProbCQL, dataloader, optimizer, device,
                   gamma: float = 0.99, target_update_freq: int = 100,
                   reward_scale: float = 0.1):
    model.train()
    tot_td = tot_cql = tot_kl = tot_total = tot_q = tot_var = 0.0
    n = 0

    for batch in dataloader:
        model.training_step += 1

        s  = batch['state'].to(device).float() / 255.0
        a  = batch['action'].to(device)
        r  = batch['reward'].to(device).float() * reward_scale
        ns = batch['next_state'].to(device).float() / 255.0
        d  = batch['done'].to(device).float()

        if r.dim() == 2: r = r.squeeze(1)
        if d.dim() == 2: d = d.squeeze(1)

        a_idx = a.argmax(dim=-1) if a.dim() == 2 else a

        # ---- Online forward (stochastic z during training) ----
        q_values, mu, logvar = model(s)                          # (B,A), (B,512)×2
        q_sa = q_values.gather(1, a_idx.unsqueeze(1)).squeeze(1) # (B,)

        # ---- Target: fc_mu_target + q_head_target (both frozen snapshots) ----
        # Using the live fc_mu here would cause the target to shift every step,
        # nullifying the stability guarantee of the target network.
        with torch.no_grad():
            next_feat  = model.cnn(ns)                           # (B, 3136)
            next_mu    = model.fc_mu_target(next_feat)           # (B, 512)
            next_q     = model.q_head_target(next_mu)            # (B, A)
            target_q   = r + gamma * next_q.max(dim=1)[0] * (1.0 - d)

        # ---- Losses ----
        td_loss  = F.smooth_l1_loss(q_sa, target_q)
        cql_loss = (torch.logsumexp(q_values, dim=1) - q_sa).mean()
        kl_loss  = _kl_standard_normal(mu, logvar)
        loss     = td_loss + model.alpha * cql_loss + model.beta_kl * kl_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()

        if model.training_step % target_update_freq == 0:
            model.update_target()

        bs          = s.size(0)
        tot_td     += td_loss.item()           * bs
        tot_cql    += cql_loss.item()          * bs
        tot_kl     += kl_loss.item()           * bs
        tot_total  += loss.item()              * bs
        tot_q      += q_values.mean().item()   * bs
        tot_var    += logvar.exp().mean().item() * bs
        n          += bs

    return (tot_td/n, tot_cql/n, tot_kl/n, tot_total/n,
            tot_q/n,  tot_var/n)


# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------

def val_prob_cql(model: ProbCQL, dataloader, device,
                 gamma: float = 0.99, reward_scale: float = 0.1,
                 action_weights=None):
    model.eval()
    tot_td = tot_cql = tot_kl = tot_total = tot_q = tot_var = 0.0
    tot_ce = tot_wce = tot_correct = 0.0
    n = 0

    with torch.no_grad():
        for batch in dataloader:
            s  = batch['state'].to(device).float() / 255.0
            a  = batch['action'].to(device)
            r  = batch['reward'].to(device).float() * reward_scale
            ns = batch['next_state'].to(device).float() / 255.0
            d  = batch['done'].to(device).float()

            if r.dim() == 2: r = r.squeeze(1)
            if d.dim() == 2: d = d.squeeze(1)

            a_idx = a.argmax(dim=-1) if a.dim() == 2 else a

            # model.eval() → reparameterize returns μ (no noise)
            q_values, mu, logvar = model(s)
            q_sa = q_values.gather(1, a_idx.unsqueeze(1)).squeeze(1)

            next_feat  = model.cnn(ns)
            next_mu    = model.fc_mu_target(next_feat)
            next_q     = model.q_head_target(next_mu)
            target_q   = r + gamma * next_q.max(dim=1)[0] * (1.0 - d)

            td_loss  = F.smooth_l1_loss(q_sa, target_q)
            cql_loss = (torch.logsumexp(q_values, dim=1) - q_sa).mean()
            kl_loss  = _kl_standard_normal(mu, logvar)
            loss     = td_loss + model.alpha * cql_loss + model.beta_kl * kl_loss

            # Human-action accuracy: raw Q-values as logits.
            # F.cross_entropy applies log_softmax internally, so it expects
            # unnormalised logits.  Z-score normalisation would compress the
            # inter-action margin and destroy the temperature information that
            # reflects how confidently the model prefers one action over others.
            ce_loss  = F.cross_entropy(q_values, a_idx)
            wce_loss = F.cross_entropy(q_values, a_idx, weight=action_weights)
            pred     = q_values.argmax(dim=-1)

            bs           = s.size(0)
            tot_td      += td_loss.item()            * bs
            tot_cql     += cql_loss.item()           * bs
            tot_kl      += kl_loss.item()            * bs
            tot_total   += loss.item()               * bs
            tot_q       += q_values.mean().item()    * bs
            tot_var     += logvar.exp().mean().item() * bs
            tot_ce      += ce_loss.item()            * bs
            tot_wce     += wce_loss.item()           * bs
            tot_correct += (pred == a_idx).sum().item()
            n           += bs

    return (
        tot_td/n,      tot_cql/n,     tot_kl/n,      tot_total/n,
        tot_q/n,       tot_var/n,
        tot_ce/n,      tot_wce/n,     tot_correct/n,
    )
