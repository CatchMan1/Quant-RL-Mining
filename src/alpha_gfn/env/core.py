import random
from typing import List, Tuple, Optional
import torch
import numpy as np
import math
from torch import nn
from gfn.env import DiscreteEnv
from gfn.states import DiscreteStates
from gfn.actions import Actions

from alphagen.data.tokens import *
from alphagen.data.tree import ExpressionBuilder, OutOfDataRangeError
from ..config import *
from ..alpha_pool import AlphaPoolGFN
from ..preprocessors import IntegerPreprocessor

class GFNEnvCore(DiscreteEnv):
    def __init__(self, pool: AlphaPoolGFN,
                 encoder: nn.Module = None,
                 device: torch.device = torch.device('cuda:0'),
                 mask_dropout_prob: float = 0.1,
                 ssl_weight: float = 0.1,
                 nov_weight: float = 0.1):
        self.pool = pool # 因子池
        self.encoder = encoder # RGCN
        self.mask_dropout_prob = mask_dropout_prob # 掩码dropout概率
        self.ssl_weight = ssl_weight # 自监督学习权重
        self.nov_weight = nov_weight # 新颖性权重
        self.builder = ExpressionBuilder()# 表达式构建器

        # Token定义
        self.beg_token = [BEG_TOKEN] # 起始符
        self.operators = [OperatorToken(op) for op in OPERATORS]
        self.features = [FeatureToken(feat) for feat in FEATURES]
        self.delta_times = [DeltaTimeToken(dt) for dt in DELTA_TIMES]
        self.sep_token = [SEP_TOKEN]
        # 动作列表action_list
        # 动作顺序：
        # [0]      BEG_TOKEN
        # [1..n]   操作符 (OperatorToken)
        # [n+1..m] 特征 (FeatureToken: $close, $open, etc.)
        # [m+1..p] 时间偏移 (DeltaTimeToken: -1, -5, -10, etc.)
        # [p+1]    SEP_TOKEN
        self.action_list: List[Token] = self.beg_token + self.operators + self.features + self.delta_times + self.sep_token
        self.id_to_token_map = {i: token for i, token in enumerate(self.action_list)}
        n_actions = len(self.action_list)
        print(self.token_to_id_map)
        # 初始状态 s0: [BEG_TOKEN, -1, -1, ..., -1]
        s0 = torch.tensor([self.token_to_id_map[BEG_TOKEN]] + [-1] * (MAX_EXPR_LENGTH - 1), dtype=torch.long, device=device)
        # Sink state: a special state that represents completed trajectories
        # We use -2 to distinguish it from the padding value -1
        # 终止状态 sf: [SEP_TOKEN, SEP_TOKEN, ..., SEP_TOKEN]
        sf = torch.full((MAX_EXPR_LENGTH,), self.token_to_id_map[SEP_TOKEN], dtype=torch.long, device=device)
        preprocessor = IntegerPreprocessor(output_dim=MAX_EXPR_LENGTH)

        super().__init__(
            n_actions=n_actions,
            s0=s0,
            sf=sf,
            state_shape=(MAX_EXPR_LENGTH,),
            dummy_action=torch.tensor([-1], dtype=torch.long, device=device),
            exit_action=torch.tensor([self.token_to_id_map[SEP_TOKEN]], dtype=torch.long, device=device),
            device_str=str(device),
            preprocessor=preprocessor
        )

    @property
    def token_to_id_map(self):
        # The last action is the exit action
        mapping = {token: i for i, token in enumerate(self.action_list)}
        return mapping

    def set_encoder(self, encoder: nn.Module):
        self.encoder = encoder

    def tensor_to_tokens(self, tensor: torch.Tensor) -> List[Optional[Token]]:
        return [self.id_to_token_map.get(i.item()) for i in tensor]
    '''
    当前状态: [BEG, $low, Abs, -1, -1, ...]
    动作: Ref (ID=8)
    下一状态: [BEG, $low, Abs, Ref, -1, ...]

    当前状态: [BEG, $low, Abs, -10, Ref, ...]
    动作: SEP (结束)
    下一状态: [SEP, SEP, SEP, SEP, SEP, ...] (终止状态)
    '''
    def step(self, states: DiscreteStates, actions: Actions) -> torch.Tensor:
        next_states_tensor = states.tensor.clone()
        for i, (state_tensor, action_id_tensor) in enumerate(zip(states.tensor, actions.tensor.squeeze(-1))):
            action_id = action_id_tensor.item()
            # 如果是结束动作，直接跳转到终止状态
            if self.id_to_token_map[action_id] == SEP_TOKEN: # Exit action - transition to sink state
                next_states_tensor[i] = self.sf
            else: # Not an exit action
                # 否则，将动作添加到当前状态
                non_padded_len = (state_tensor != -1).sum()
                if non_padded_len >= MAX_EXPR_LENGTH:
                    raise ValueError("Full states only allow the exit action")
                next_states_tensor[i, non_padded_len] = action_id
        return next_states_tensor

    def backward_step(self, states: DiscreteStates, actions: Actions) -> torch.Tensor:
        next_states_tensor = states.tensor.clone()
        for i, (state, action) in enumerate(zip(states.tensor, actions.tensor.squeeze(-1))):
            length = int((state != -1).sum())
            if torch.equal(state, self.sf) or length <= 1 or state[length - 1] != action:
                raise ValueError("Backward action must remove the last token of a non-initial state")
            next_states_tensor[i, length - 1] = -1
        return next_states_tensor

    def update_masks(self, states: DiscreteStates):
        batch_masks = []
        flat_states = states.tensor.reshape(-1, MAX_EXPR_LENGTH)
        backward_masks = torch.zeros((len(flat_states), self.n_actions - 1),
                                     dtype=torch.bool, device=self.device)
        for row, state_tensor in enumerate(flat_states):
            # 终止状态的所有动作都无效
            # 因为终止状态表示一个完整的表达式，不能添加新的操作
            if torch.all(state_tensor == self.sf):  # This is a sink state
                batch_masks.append([False] * self.n_actions)
                continue
            # 使用 ExpressionBuilder 检查当前状态的有效性
            builder = ExpressionBuilder()
            token_ids = [tid.item() for tid in state_tensor if tid >= 0]
            if len(token_ids) > 1:
                backward_masks[row, token_ids[-1]] = True
            for token_id in token_ids[1:]:
                builder.add_token(self.id_to_token_map[token_id])

            valid_actions = [False] * self.n_actions

            # Account for BEG_TOKEN at index 0
            beg_offset = len(self.beg_token)  # = 1
            n_ops = len(self.operators)
            n_features = len(self.features)
            n_dts = len(self.delta_times)

            for i, op_token in enumerate(self.operators):
                valid_actions[beg_offset + i] = builder.validate(op_token)
            for i, feature_token in enumerate(self.features):
                valid_actions[beg_offset + n_ops + i] = builder.validate(feature_token)
            for i, dt_token in enumerate(self.delta_times):
                valid_actions[beg_offset + n_ops + n_features + i] = builder.validate(dt_token)

            if len(token_ids) < MAX_EXPR_LENGTH:
                if builder.is_valid():
                    valid_actions[-1] = True
            else:
                valid_actions = [False] * self.n_actions
                valid_actions[-1] = True

            # Apply random mask dropout based on expression length
            # Longer expressions get higher dropout probability
            # 随机掩码，表达式越长，掩码概率越高
            expr_length = len(token_ids)
            length_based_dropout_prob = self.mask_dropout_prob * (expr_length / MAX_EXPR_LENGTH)

            # 如果最后一个动作是结束动作，不掩码其他动作
            # 因为结束动作表示表达式结束，不能添加新的操作
            # set True actions to False (except the last action which is SEP)
            true_indices = [i for i in range(len(valid_actions) - 1) if valid_actions[i]]  # Exclude last action
            if valid_actions[-1] == True:
                if np.random.rand() < length_based_dropout_prob:
                    for idx in true_indices:
                        valid_actions[idx] = False
                    #print(f"[Mask Debug] Expr length: {expr_length}, Masked all {len(true_indices)} actions")

            # 生成合法动作的掩码
            batch_masks.append(valid_actions)

        states.forward_masks = torch.tensor(batch_masks, dtype=torch.bool, device=self.device).reshape(
            *states.batch_shape, self.n_actions)
        states.backward_masks = backward_masks.reshape(*states.batch_shape, self.n_actions - 1)

    def reward(self, final_states: DiscreteStates) -> torch.Tensor:
        rewards = []
        for state_tensor in final_states.tensor:
            builder = ExpressionBuilder()
            token_ids = [tid.item() for tid in state_tensor if tid >= 0]

            # Reconstruct the expression for reward calculation
            for token_id in token_ids[1:]:
                builder.add_token(self.id_to_token_map[token_id])

            reward = 0.0
            if builder.is_valid():
                try:
                    expr = builder.get_tree()

                    # Compute embedding only for this valid expression
                    embedding = None
                    if self.encoder is not None:
                        with torch.no_grad():
                            # Add batch dimension for single state
                            single_state = state_tensor.unsqueeze(0)
                            embedding = self.encoder(single_state).squeeze(0)
                    ic_reward, nov_reward, ssl_reward = self.pool.try_new_expr_with_ssl(expr, embedding)
                    reward = ic_reward + self.ssl_weight * ssl_reward + self.nov_weight * nov_reward
                except OutOfDataRangeError:
                    reward = 0.0
            rewards.append(np.maximum(reward, np.exp(-10)))

        return torch.tensor(rewards, dtype=torch.float, device=self.device)
