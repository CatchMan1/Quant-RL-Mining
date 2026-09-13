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

        # Calculate entropy term

        entropy_term = torch.zeros(trajectories.n_trajectories, device=trajectories.states.device)
        if trajectories.estimator_outputs is not None:  
            logits = trajectories.estimator_outputs  #从策略网络输出logits计算概率 (max_len, batch_size, n_actions) 输出的是每个动作的概率
            temp_logits = logits / self.entropy_temperature
            probs = torch.softmax(temp_logits, dim=-1) # (max_len, batch_size, n_actions)
            # 计算熵：H(p) = -Σ p(x) * log(p(x))
            entropy_per_step = -torch.sum(
                probs * torch.log(probs.clamp(min=1e-9)), dim=-1
            )  # (max_len, batch_size)

            # Mask out invalid steps for each trajectory
            # 不同轨迹的长度不同，需要对每个轨迹的熵进行求和
            # 对没有到达max_len的步骤mask为False,熵为0
            max_len = logits.shape[0]
            step_indices = torch.arange(max_len, device=logits.device).unsqueeze(1)
            valid_steps_mask = step_indices < trajectories.when_is_done.unsqueeze(0)

            masked_entropy = entropy_per_step * valid_steps_mask

            # Sum entropy over trajectory steps for each trajectory in the batch
            entropy_sum_per_trajectory = masked_entropy.sum(dim=0)  # (batch_size,)

            entropy_term = entropy_sum_per_trajectory * self.entropy_coef

        # 如果没有imator_outputs，使用旧的计算熵的方法
        elif self.entropy_coef > 0:
            # Fallback for older gfn versions or if outputs are not saved
            non_terminal_mask = ~trajectories.is_done # 找到还在进行中的轨迹
            if torch.any(non_terminal_mask):
                # This part is not batched correctly for n_trajectories > 1
                # but kept for compatibility with the old logic.
                non_terminal_states = trajectories.states[non_terminal_mask]
                # 手动预处理+策略网络前向传播
                preprocessed_states = self.pf.preprocessor(non_terminal_states)
                logits = self.pf.module(preprocessed_states)# logits: (n_non_terminal, n_actions)
                # 计算熵
                temp_logits = logits / self.entropy_temperature

                probs = torch.softmax(temp_logits, dim=-1)
                entropy = -torch.sum(
                    probs * torch.log(probs.clamp(min=1e-9)), dim=-1
                )

                entropy_term = entropy.sum() * self.entropy_coef
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

