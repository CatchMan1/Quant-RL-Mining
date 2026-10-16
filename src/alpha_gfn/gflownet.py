import torch
from typing import Tuple
from gfn.containers import Trajectories
from gfn.env import Env
from gfn.gflownet.trajectory_balance import TBGFlowNet
from gfn.modules import DiscretePolicyEstimator
from torchtyping import TensorType as TT


class EntropyTBGFlowNet(TBGFlowNet):
    def __init__(
        self,
        pf: DiscretePolicyEstimator, # 前向策略
        pb: DiscretePolicyEstimator, # 后向策略
        entropy_coef: float = 0.0, # 熵系数
        # 熵系数越大，策略越随机
        entropy_temperature: float = 1.0, # 熵温度
        **kwargs,
    ):
        super().__init__(pf, pb, **kwargs)
        self.entropy_coef = entropy_coef
        self.entropy_temperature = entropy_temperature

    def get_trajectories_scores(
        self, trajectories: Trajectories, recalculate_all_logprobs: bool = False
    ) -> Tuple[
        TT["n_trajectories", torch.float],
        TT["n_trajectories", torch.float],
        TT["n_trajectories", torch.float],
        TT["n_trajectories", torch.float]
    ]:
        # trajectories是一个包含多个轨迹的容器，存储了 GFlowNet 采样到的完整生成过程。
        if self.entropy_coef <= 0:
            # 不启用熵正则化，直接返回原始分数
            return (*super().get_trajectories_scores(trajectories, recalculate_all_logprobs), torch.zeros(trajectories.n_trajectories, device=trajectories.states.device))
        # 计算熵项
        log_pf, log_pb, scores = super().get_trajectories_scores(
            trajectories, recalculate_all_logprobs
        )

        # 只在真实动作步上计算，避免补齐位置的全 -inf logits 产生 NaN。
        step_count = trajectories.actions.tensor.shape[0]
        steps = torch.arange(step_count, device=trajectories.states.device).unsqueeze(1)
        valid_steps = steps < trajectories.when_is_done.unsqueeze(0)
        action_states = trajectories.states[:-1][valid_steps]
        if trajectories.estimator_outputs is not None and not recalculate_all_logprobs:
            logits = trajectories.estimator_outputs[valid_steps]
        else:
            logits = self.pf(action_states)
        logits = logits.masked_fill(~action_states.forward_masks, -torch.inf)
        probs = torch.softmax(logits / self.entropy_temperature, dim=-1)
        entropy = -(probs * probs.clamp(min=1e-9).log()).sum(dim=-1)
        entropy_per_step = torch.zeros(valid_steps.shape, device=logits.device, dtype=logits.dtype)
        entropy_per_step[valid_steps] = entropy
        entropy_term = entropy_per_step.sum(dim=0) * self.entropy_coef
        #scores = log_pf - log_pb - log_reward；reward是该轨迹下的总奖励
        return log_pf, log_pb, scores, entropy_term
    
    def loss(
        self,
        env: Env,
        trajectories: Trajectories,
        recalculate_all_logprobs: bool = False,
    ) -> TT[0, float]:
        """Trajectory balance loss.

        The trajectory balance loss is described in 2.3 of
        [Trajectory balance: Improved credit assignment in GFlowNets](https://arxiv.org/abs/2201.13259))

        Raises:
            ValueError: if the loss is NaN.
        """
        del env  # unused
        _, _, scores, entropy_term = self.get_trajectories_scores(
            trajectories, recalculate_all_logprobs=recalculate_all_logprobs
        )# total_log_pf_trajectories - total_log_pb_trajectories - log_rewards,
        loss = (scores + self.logZ).pow(2).mean() - entropy_term.mean()
        if torch.isnan(loss):
            # set inf
            loss = torch.tensor(float('inf'))

        return loss

