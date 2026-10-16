import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / 'src'), str(Path(__file__).resolve().parents[1])]
from alpha_gfn.config import MAX_EXPR_LENGTH
from alpha_gfn.env.core import GFNEnvCore
from alpha_gfn.modules import _build_graph_from_rpn
from alpha_gfn.gflownet import EntropyTBGFlowNet
from gfn.modules import DiscretePolicyEstimator
from gfn.samplers import Sampler
from train_gfn import WeightScheduler, GFNLogger


@pytest.fixture
def env():
    pool = SimpleNamespace(try_new_expr_with_ssl=lambda *args: (1., 0., 0.))
    return GFNEnvCore(pool, device=torch.device('cpu'), mask_dropout_prob=0)


def ids(env):
    m = env.token_to_id_map
    return m[env.features[0]], m[env.operators[0]]


def test_graph_operator_mapping(env):
    op, _ = ids(env)
    sub = next(i for i, t in env.id_to_token_map.items() if str(t) == "Sub")
    graph = _build_graph_from_rpn([0, op, op, sub], nn.Embedding(env.n_actions, 4), 0)
    assert graph.edge_index.tolist() == [[2, 1], [3, 3]]
    assert graph.edge_type.tolist() == [3, 2]


def test_unique_parent_and_reverse_step(env):
    op, _ = ids(env)
    initial = env.reset(batch_shape=(1,))
    assert not initial.backward_masks.any()
    action = env.actions_from_tensor(torch.tensor([[op]]))
    child = env._step(initial, action)
    assert child.backward_masks.sum().item() == 1
    assert child.backward_masks[0, op]
    assert torch.equal(env.backward_step(child, action), initial.tensor)


def test_length_limit(env):
    op, unary = ids(env)
    state = env.states_from_tensor(torch.tensor([[0, op] + [unary] * (MAX_EXPR_LENGTH - 2)]))
    env.update_masks(state)
    assert state.forward_masks.sum().item() == 1
    assert state.forward_masks[0, -1]
    with pytest.raises(ValueError):
        env.step(state, env.actions_from_tensor(torch.tensor([[unary]])))


@pytest.mark.parametrize('kind', ['linear', 'polynomial', 'exponential'])
@pytest.mark.parametrize('ratio', [0., .2])
def test_weight_decay(kind, ratio):
    if kind == 'exponential' and ratio == 0:
        with pytest.raises(ValueError):
            WeightScheduler(.5, .3, ratio, 10, kind)
        return
    scheduler = WeightScheduler(.5, .3, ratio, 10, kind)
    values = [scheduler.get_current_weights()[0]]
    for _ in range(12):
        scheduler.step()
        values.append(scheduler.get_current_weights()[0])
    assert values[0] == .5
    assert values[10] == pytest.approx(.5 * ratio)
    assert values[12] == pytest.approx(.5 * ratio)
    assert all(a >= b for a, b in zip(values, values[1:]))


def test_entropy_saved_and_fallback_agree(env):
    torch.manual_seed(7)
    pf = DiscretePolicyEstimator(nn.Linear(MAX_EXPR_LENGTH, env.n_actions), env.n_actions, preprocessor=lambda s: s.tensor.float())
    pb = DiscretePolicyEstimator(nn.Linear(MAX_EXPR_LENGTH, env.n_actions - 1), env.n_actions, preprocessor=lambda s: s.tensor.float(), is_backward=True)
    with torch.no_grad():
        pf.module.weight.zero_()
        pf.module.bias.zero_()
        pf.module.bias[-1] = 2.
    loss = EntropyTBGFlowNet(pf, pb, entropy_coef=.1)
    trajectories = Sampler(pf).sample_trajectories(env, n_trajectories=8, save_estimator_outputs=True)
    assert trajectories.when_is_done.max() <= MAX_EXPR_LENGTH
    saved = loss.get_trajectories_scores(trajectories)[-1]
    assert saved.shape == (8,)
    recalculated = loss.get_trajectories_scores(trajectories, recalculate_all_logprobs=True)[-1]
    assert torch.allclose(saved, recalculated)
    trajectories.estimator_outputs = None
    fallback = loss.get_trajectories_scores(trajectories)[-1]
    assert torch.isfinite(saved).all()
    assert torch.allclose(saved, fallback)
    loss.loss(env, trajectories).backward()
    assert torch.isfinite(pf.module.weight.grad).all()


def test_checkpoint_saves_logz(tmp_path):
    logger = GFNLogger.__new__(GFNLogger)
    logger.log_dir = str(tmp_path)
    logger.model = nn.Linear(2, 2)
    logger.pb_head = None
    logger.logZ = nn.Parameter(torch.tensor(3.25))
    logger.args = None
    logger.pool = SimpleNamespace(embeddings=[], size=0, to_dict=lambda: {})
    logger.save_final(12)
    checkpoint = torch.load(tmp_path / 'model.pt', weights_only=False)
    assert checkpoint['logZ'].item() == 3.25


def test_masks_support_empty_and_multidimensional_batches(env):
    empty = env.states_from_tensor(torch.empty((0, MAX_EXPR_LENGTH), dtype=torch.long))
    assert empty.backward_masks.shape == (0, env.n_actions - 1)
    states = env.states_from_tensor(torch.stack([env.s0, env.sf]).reshape(1, 2, -1))
    assert states.forward_masks.shape == (1, 2, env.n_actions)
    assert not states.backward_masks.any()
    assert not states.forward_masks[0, 1].any()


def test_gnn_sampling_loss_has_finite_gradients(env):
    from alpha_gfn.modules import SequenceEncoder
    from alpha_gfn.config import HIDDEN_DIM
    torch.manual_seed(11)
    backbone = SequenceEncoder(env.n_actions, encoder_type='gnn')
    pf = DiscretePolicyEstimator(nn.Sequential(backbone, nn.Linear(HIDDEN_DIM, env.n_actions)),
                                 env.n_actions, env.preprocessor)
    pb = DiscretePolicyEstimator(nn.Sequential(backbone, nn.Linear(HIDDEN_DIM, env.n_actions - 1)),
                                 env.n_actions, env.preprocessor, is_backward=True)
    loss_fn = EntropyTBGFlowNet(pf, pb, entropy_coef=.01)
    trajectories = Sampler(pf).sample_trajectories(env, n_trajectories=4, save_estimator_outputs=True)
    assert trajectories.when_is_done.max() <= MAX_EXPR_LENGTH
    _, log_pb, _, _ = loss_fn.get_trajectories_scores(trajectories)
    assert torch.equal(log_pb, torch.zeros_like(log_pb))
    loss = loss_fn.loss(env, trajectories)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(loss_fn.logZ.grad)
    assert all(torch.isfinite(p.grad).all() for p in backbone.parameters() if p.grad is not None)
