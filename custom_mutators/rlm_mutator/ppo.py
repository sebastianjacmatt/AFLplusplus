"""PPO trainer for rlm_mutator.

Subclass of BaseTrainer that overrides HF Trainer.compute_loss with the
clipped PPO objective (Schulman et al., 2017), plus an optional KL penalty
against the reference snapshot.

compute_loss and _advantage are left unimplemented — see their docstrings
for the pieces that need to be wired in.
"""

from base_trainer import BaseTrainer


class PPOTrainer(BaseTrainer):
    """PPO finetuner for the mutator policy.

    Implements J_PPO(theta) = E[min(ratio · A, clip(ratio, 1-eps, 1+eps) · A)]
    with an optional -beta · KL(pi || pi_ref) penalty.
    """

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Clipped PPO objective + critic value loss + entropy bonus + optional KL.

        Needed pieces (not yet implemented):
          * Forward pass through model to get per-token log pi_theta(y_t | x_t)
            under the CURRENT actor, using inputs['input_ids'] and inputs['labels'].
          * ratio = exp(current_logprob - inputs['old_log_prob']).
          * advantage from self._advantage(inputs) (critic returns - value baseline).
          * Clipped objective: min(ratio · A, clip(ratio, 1-eps, 1+eps) · A)
            with eps = self.training_cfg.clip_epsilon.
          * Value loss: (V_psi(x_t) - return)^2 weighted by training_cfg.ppo.value_coef.
          * Entropy bonus weighted by training_cfg.ppo.entropy_coef.
          * Optional beta · KL(pi || pi_ref) via self.kl_divergence() when
            training_cfg.kl_coef > 0.
        """
        raise NotImplementedError("PPOTrainer.compute_loss not yet implemented")

    def _advantage(self, inputs):
        """GAE-lambda advantage — needs a critic (value head on the encoder).

        Not yet implemented.  Open design question: add a separate linear value
        head on top of the encoder's pooled output, vs. a lightweight MLP critic
        that shares no parameters with the actor.  training_cfg.ppo.gae_lambda
        controls the smoothing factor.
        """
        raise NotImplementedError("PPOTrainer._advantage not yet implemented")
