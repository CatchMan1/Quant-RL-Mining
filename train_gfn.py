import sys
import os
# 环境变量配置
_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, 'src'))

import torch
import random
import numpy as np
import argparse
import json
from torch.optim import Adam
from torch.distributions import Categorical
from torch import nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from src.alpha_gfn.config import *
from src.alpha_gfn.env.core import GFNEnvCore
from src.alpha_gfn.modules import SequenceEncoder
from src.alpha_gfn.alpha_pool import AlphaPoolGFN
from src.alphagen.data.expression import *
from src.alphagen_qlib.stock_data import StockData
from src.alphagen.utils.correlation import batch_pearsonr
from src.alphagen.models.alpha_pool import _timeseries_pearsonr
from src.alpha_gfn.gflownet import EntropyTBGFlowNet

from gfn.samplers import Sampler
from gfn.gflownet.trajectory_balance import TBGFlowNet
from gfn.modules import DiscretePolicyEstimator
from gfn.utils.modules import NeuralNet

class GFNLogger:
    def __init__(self, model: nn.Module, pool: AlphaPoolGFN, log_dir: str, test_data: StockData, target: Expression,
                 val_data: StockData = None, pb_head: nn.Module = None, logZ: nn.Parameter = None, args=None):
        self.model = model
        self.pool = pool
        self.log_dir = log_dir
        self.test_data = test_data
        self.val_data = val_data
        self.target = target
        self.pb_head = pb_head
        self.logZ = logZ
        self.args = args
        self.writer = SummaryWriter(log_dir)
        self.target_test = self.pool._normalize_by_day(self.target.evaluate(self.test_data))
        self.target_val = None
        self.target_val = self.pool._normalize_by_day(self.target.evaluate(self.val_data))

    def log_metrics(self, episode: int, losses=False, logZ=False):
        self.writer.add_scalar('pool/size', self.pool.size, episode)
        if self.pool.size > 0:
            self.writer.add_scalar('pool/best_single_ic', np.max(self.pool.single_ics[:self.pool.size]), episode)
            ic_test, rank_ic_test = self.pool.test_ensemble(self.test_data, self.target)
            self.writer.add_scalar('test/ic', ic_test, episode)
            self.writer.add_scalar('test/rank_ic', rank_ic_test, episode)
            ic_val, rank_ic_val = self.pool.test_ensemble(self.val_data, self.target)
            self.writer.add_scalar('val/ic', ic_val, episode)
            self.writer.add_scalar('val/rank_ic', rank_ic_val, episode)
        self.writer.add_scalar('pool/eval_cnt', self.pool.eval_cnt, episode)
        if losses:
            self.writer.add_scalar('train/loss', sum(losses) / len(losses), episode)
            self.writer.add_scalar('train/logZ', logZ, episode)

    def save_final(self, episode: int, optimizer: Adam = None):
        """训练结束后保存一次：模型权重 model.pt + 因子池 pool.json"""
        model_path = os.path.join(self.log_dir, 'model.pt')
        pool_path = os.path.join(self.log_dir, 'pool.json')
        ckpt = {
            'pf_state_dict': self.model.state_dict(),
            'episode': episode
        }
        if self.pb_head is not None:
            ckpt['pb_head_state_dict'] = self.pb_head.state_dict()
        if optimizer is not None:
            ckpt['optimizer_state_dict'] = optimizer.state_dict()
        # logZ 是独立的可学习标量，不在 model.state_dict() 里，单独保存
        if self.logZ is not None:
            ckpt['logZ'] = self.logZ.detach().cpu()
        # 记录本次运行的超参，供增量训练自动继承
        if self.args is not None:
            ckpt['args'] = vars(self.args)
        # 因子池的结构嵌入（SSL / 新颖性奖励依赖），与因子池一起保存
        ckpt['pool_embeddings'] = [
            e.detach().cpu() if e is not None else None
            for e in self.pool.embeddings[:self.pool.size]
        ]
        torch.save(ckpt, model_path)
        with open(pool_path, 'w') as f:
            json.dump(self.pool.to_dict(), f, indent=4)

    def show_pool_state(self):
        state = self.pool.state
        exprs = state.get('exprs', [])
        n = len(exprs)
        print('---------------------------------------------')
        for i in range(n):
            expr = state['exprs'][i]
            expr_str = str(expr)
            ic_ret = state['ics_ret'][i]

            # Calculate test IC (time-series)
            value_test = self.pool._normalize_by_day(expr.evaluate(self.test_data))
            ic_test = _timeseries_pearsonr(value_test, self.target_test)

            # Calculate validation IC (time-series)
            val_ic_str = ''
            value_val = self.pool._normalize_by_day(expr.evaluate(self.val_data))
            val_ic = _timeseries_pearsonr(value_val, self.target_val)
            val_ic_str = f', val_ic={val_ic:.4f}'

            print(f'> Alpha #{i}: ic={ic_ret:.4f}{val_ic_str}, test_ic={ic_test:.4f}, expr={expr_str}')
        if self.pool.size > 0:
            print(f'>> Best single ic: {np.max(self.pool.single_ics[:self.pool.size]):.4f}')
        print('---------------------------------------------')

    def close(self):
        self.writer.close()


class WeightScheduler:
    """按训练步数直接计算奖励权重，达到终值后保持不变。"""

    def __init__(self, initial_ssl_weight, initial_nov_weight, final_ratio, total_steps, scheduler_type='linear'):
        if total_steps <= 0 or not 0 <= final_ratio <= 1:
            raise ValueError("total_steps must be positive and final_ratio must be in [0, 1]")
        if scheduler_type not in ('linear', 'polynomial', 'exponential'):
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")
        if scheduler_type == 'exponential' and final_ratio == 0:
            raise ValueError("Exponential decay requires final_weight_ratio > 0; use linear or polynomial to reach zero")
        self.initial_ssl_weight = initial_ssl_weight
        self.initial_nov_weight = initial_nov_weight
        self.final_ratio = final_ratio
        self.total_steps = total_steps
        self.scheduler_type = scheduler_type
        self.current_step = 0

    def step(self):
        self.current_step = min(self.current_step + 1, self.total_steps)

    def get_current_weights(self):
        progress = self.current_step / self.total_steps
        if self.scheduler_type == 'exponential':
            ratio = self.final_ratio ** progress
        else:
            power = 2 if self.scheduler_type == 'polynomial' else 1
            ratio = self.final_ratio + (1 - self.final_ratio) * (1 - progress) ** power
        return self.initial_ssl_weight * ratio, self.initial_nov_weight * ratio

# ===== Resume helpers =====
# 这些超参与 checkpoint 强相关（模型结构或训练信号），
# 增量训练时若命令行未显式给出，则自动继承 checkpoint 中记录的值。
RESUME_INHERIT_KEYS = (
    'encoder_type', 'pool_capacity', 'entropy_coef', 'entropy_temperature',
    'mask_dropout_prob', 'ssl_weight', 'nov_weight',
    'weight_decay_type', 'final_weight_ratio',
)


def _resolve_resume_paths(resume_path: str):
    """返回 (model_path, resume_dir)。目录输入时优先 model.pt，其次最新的 model_{episode}.pt。"""
    if os.path.isfile(resume_path):
        return resume_path, (os.path.dirname(resume_path) or '.')
    resume_dir = resume_path
    candidates = [f for f in os.listdir(resume_dir)
                  if f == 'model.pt' or (f.startswith('model_') and f.endswith('.pt'))]
    if not candidates:
        raise FileNotFoundError(f"No model checkpoint found in {resume_dir}")
    if 'model.pt' in candidates:
        return os.path.join(resume_dir, 'model.pt'), resume_dir
    candidates.sort(key=lambda f: int(f.replace('model_', '').replace('.pt', '')))
    return os.path.join(resume_dir, candidates[-1]), resume_dir


def _resolve_pool_path(resume_dir: str, model_path: str, ckpt: dict) -> str:
    """新版为 pool.json；旧版为 pool_{episode}.json"""
    if os.path.exists(os.path.join(resume_dir, 'pool.json')):
        return os.path.join(resume_dir, 'pool.json')
    model_file = os.path.basename(model_path)
    ep = model_file[len('model_'):-len('.pt')] if model_file.startswith('model_') else ckpt.get('episode', 0)
    return os.path.join(resume_dir, f'pool_{ep}.json')


def _inherit_saved_args(args, ckpt: dict) -> None:
    """用 checkpoint 中记录的超参，补齐命令行未显式传入的项。"""
    saved = ckpt.get('args')
    if not saved:
        return
    cli_set = {a[2:].split('=')[0] for a in sys.argv[1:] if a.startswith('--')}
    inherited = []
    for k in RESUME_INHERIT_KEYS:
        if k in saved and k not in cli_set and getattr(args, k, None) != saved[k]:
            setattr(args, k, saved[k])
            inherited.append(f'{k}={saved[k]}')
    if inherited:
        print('[Resume] 继承 checkpoint 中的超参: ' + ', '.join(inherited))


def train(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # ===== Resume（第一步）：读取 checkpoint 并继承超参 =====
    # 必须在建模型 / 环境 / 调度器之前，否则 encoder_type、pool_capacity 等改了也不生效
    resume_ckpt = None
    resume_model_path = resume_dir = None
    if args.resume is not None:
        resume_model_path, resume_dir = _resolve_resume_paths(args.resume)
        resume_ckpt = torch.load(resume_model_path, map_location=device)
        _inherit_saved_args(args, resume_ckpt)

    # Reproducibility
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    # Initialize StockData and target expression (15-min bars, single asset)
    data = StockData(csv_path='data/kline_im.csv', start_time='2022-10-10', end_time='2024-12-31',
                     max_backtrack_days=200, max_future_days=1, device=device) # 计算50tick的奖励，200bar的回看
    data_val = StockData(csv_path='data/kline_im.csv', start_time='2025-01-01', end_time='2025-07-31',
                          max_backtrack_days=200, max_future_days=1, device=device)
    data_test = StockData(csv_path='data/kline_im.csv', start_time='2025-08-01', end_time='2026-03-09',
                          max_backtrack_days=200, max_future_days=1, device=device)
    close = Feature(FeatureType.CLOSE)
    target = Ref(close, -1) / close - 1  # 1 bars = 15-min bars, single asset
    
    # Initialize AlphaPoolGFN
    pool = AlphaPoolGFN(capacity=args.pool_capacity, stock_data=data, val_data=data_val, target=target)

    # Initialize model 初始化动作空间
    n_tokens = len(FEATURES) + len(OPERATORS) + len(DELTA_TIMES)
    
    # 编码器 embedding
    backbone = SequenceEncoder(
        n_tokens,
        args.encoder_type
    )
    
    # Initialize environment with encoder and mask dropout
    env = GFNEnvCore(pool=pool,
                     encoder=backbone, 
                     device=device, 
                     mask_dropout_prob=args.mask_dropout_prob,
                     ssl_weight=args.ssl_weight,
                     nov_weight=args.nov_weight)
    
    # 初始化策略网络
    pf_head = NeuralNet(input_dim=HIDDEN_DIM, output_dim=env.n_actions, n_hidden_layers=0)
    pb_head = NeuralNet(input_dim=HIDDEN_DIM, output_dim=env.n_actions - 1, n_hidden_layers=0) # pb does not predict exit action
    
    pf_module = nn.Sequential(backbone, pf_head)
    pb_module = nn.Sequential(backbone, pb_head)
    
    pf = DiscretePolicyEstimator(pf_module, n_actions=env.n_actions, preprocessor=env.preprocessor)
    pb = DiscretePolicyEstimator(pb_module, n_actions=env.n_actions, preprocessor=env.preprocessor, is_backward=True)

    loss_fn = EntropyTBGFlowNet(
        pf=pf,
        pb=pb,
        entropy_coef=args.entropy_coef,
        entropy_temperature=args.entropy_temperature
    )
    loss_fn.to(device)
    # 初始化采样器
    sampler = Sampler(estimator=pf)
    
    params = list(backbone.parameters()) + list(pf_head.parameters()) + list(pb_head.parameters()) + [loss_fn.logZ]
    optimizer = Adam(params, lr=LEARNING_RATE)

    # ===== Setup logging =====
    # 目录只用 run 名，不再拼接时间戳和超参数
    log_dir = os.path.join('data/gfn_logs', f'pool_{args.pool_capacity}', args.name)
    os.makedirs(log_dir, exist_ok=True)

    # ===== Training loop parameters =====
    losses = []
    minibatch_loss = 0
    episode_losses = []
    update_freq = args.update_freq
    n_episodes = args.n_episodes
    n_trajectories = args.n_trajectories

    # ===== Resume（第二步）：载入模型 / 优化器 / logZ / 因子池 =====
    start_episode = 0
    if resume_ckpt is not None:
        pool_path = _resolve_pool_path(resume_dir, resume_model_path, resume_ckpt)
        if not os.path.exists(pool_path):
            raise FileNotFoundError(f"Pool not found: {pool_path}")
        print(f"Resuming from: model={resume_model_path}, pool={pool_path}")

        pf.load_state_dict(resume_ckpt['pf_state_dict'])
        if 'pb_head_state_dict' in resume_ckpt:
            pb_head.load_state_dict(resume_ckpt['pb_head_state_dict'])
        if 'optimizer_state_dict' in resume_ckpt:
            optimizer.load_state_dict(resume_ckpt['optimizer_state_dict'])
        # logZ 不属于 pf 的 state_dict，单独恢复，否则会从初值重新学
        if 'logZ' in resume_ckpt:
            with torch.no_grad():
                loss_fn.logZ.copy_(resume_ckpt['logZ'].to(loss_fn.logZ.device))

        start_episode = int(resume_ckpt.get('episode', 0))
        print(f"Resuming from episode: {start_episode}")

        import alphagen.data.expression as _expr
        eval_ns = {"__builtins__": {}}
        for name in dir(_expr):
            if not name.startswith('_') and name[0].isupper():
                eval_ns[name] = getattr(_expr, name)
        for ft in FeatureType:
            eval_ns[ft.name.lower()] = _expr.Feature(ft)

        with open(pool_path, 'r') as f:
            pool_dict = json.load(f)
        exprs_raw = pool_dict.get('exprs', [])
        weights_saved = pool_dict.get('weights', [])
        embeddings_saved = resume_ckpt.get('pool_embeddings')  # 旧 checkpoint 无此字段
        for i, expr_str in enumerate(exprs_raw):
            clean_str = expr_str.replace('$', '')
            expr = eval(clean_str, eval_ns)
            value = pool._normalize_by_day(expr.evaluate(data))
            ic_ret, ic_mut = pool._calc_ics(value, ic_mut_threshold=0.99)
            if ic_ret is None:
                continue
            pool.exprs[pool.size] = expr
            pool.values[pool.size] = value
            pool.single_ics[pool.size] = abs(float(ic_ret))
            if len(ic_mut) > 0:
                pool.mutual_ics[pool.size, :pool.size] = ic_mut
                pool.mutual_ics[:pool.size, pool.size] = ic_mut
            pool.weights[pool.size] = weights_saved[i] if i < len(weights_saved) else abs(float(ic_ret))
            # 恢复结构嵌入，否则 SSL / 新颖性奖励对恢复出来的旧因子失效
            if embeddings_saved is not None and i < len(embeddings_saved) and embeddings_saved[i] is not None:
                pool.embeddings[pool.size] = embeddings_saved[i].to(device)
            pool.size += 1
            if pool.size >= pool.capacity:
                pool._pop()
        pool.weights[:pool.size] = np.abs(pool.weights[:pool.size])
        pool.weights[:pool.size] /= pool.weights[:pool.size].sum()
        pool.best_ic_ret = max(pool.single_ics[:pool.size]) if pool.size > 0 else 0.0
        pool.record_reward = pool_dict.get('record_reward', [])
        print(f"Restored pool with {pool.size} factors")

        n_episodes = start_episode + args.n_episodes

    # ===== Initialize weight scheduler =====
    weight_scheduler = WeightScheduler(
        initial_ssl_weight=args.ssl_weight,
        initial_nov_weight=args.nov_weight,
        final_ratio=args.final_weight_ratio,
        total_steps=n_episodes,
        scheduler_type=args.weight_decay_type
    )

    for _ in range(start_episode):
        weight_scheduler.step()

    logger = GFNLogger(pf, pool, log_dir, data_test, target, data_val, pb_head=pb_head,
                       logZ=loss_fn.logZ, args=args)

    print(f"Weight decay strategy: {args.weight_decay_type}")
    print(f"SSL weight: {args.ssl_weight:.4f} -> {args.ssl_weight * args.final_weight_ratio:.4f}")
    print(f"Novelty weight: {args.nov_weight:.4f} -> {args.nov_weight * args.final_weight_ratio:.4f}")
    print(f"Training episodes: {n_episodes} (starting from {start_episode})")
    print("=" * 50)

    for episode in tqdm(range(start_episode, n_episodes), initial=start_episode, total=n_episodes):
        # Get current weights from scheduler
        current_ssl_weight, current_nov_weight = weight_scheduler.get_current_weights()
        
        # Update environment weights
        env.ssl_weight = current_ssl_weight
        env.nov_weight = current_nov_weight
        # 采样轨迹，生成一个表达式
        save_estimator_outputs = args.entropy_coef > 0
        trajectories = sampler.sample_trajectories(
            env=env, n_trajectories=n_trajectories, save_estimator_outputs=save_estimator_outputs
        )
        loss = loss_fn.loss(env=env, trajectories=trajectories)

        if loss is not None and torch.isfinite(loss):
            minibatch_loss += loss
            episode_losses.append(loss.item())
            
        if episode > 0 and (episode + 1) % update_freq == 0 and minibatch_loss != 0:
            avg_ep_loss = sum(episode_losses) / len(episode_losses)
            losses.append(avg_ep_loss)
            minibatch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            tqdm.write(f'[Update {len(losses)}] avg_loss={avg_ep_loss:.6f} | logZ={loss_fn.logZ.item():.4f} | pool_size={pool.size}')
            minibatch_loss = 0
            episode_losses = []

        if episode > 0 and (episode + 1) % args.log_freq == 0:
            logger.log_metrics(episode, losses=losses, logZ=loss_fn.logZ.item())
            logger.show_pool_state()
            print(f"----Episode {episode}/{n_episodes} done----")
            print(f"Current weights: SSL={current_ssl_weight:.4f}, Novelty={current_nov_weight:.4f}")
            if losses:
                print(f"Recent avg loss: {sum(losses[-5:])/len(losses[-5:]):.6f} | logZ: {loss_fn.logZ.item():.4f}")
            
            # Log weight decay to tensorboard
            logger.writer.add_scalar('weights/ssl_weight', current_ssl_weight, episode)
            logger.writer.add_scalar('weights/nov_weight', current_nov_weight, episode)
        
        # Step the weight scheduler at the end of each episode
        weight_scheduler.step()

    # ===== 模型权重 + 因子池，供后续增量训练使用 =====
    logger.save_final(n_episodes, optimizer=optimizer)
    print(f"Saved final checkpoint to: {log_dir}  (model.pt, pool.json)")

    logger.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--instrument', type=str, default='im')
    parser.add_argument('--pool_capacity', type=int, default=10)
    parser.add_argument('--name', type=str, default='run',
                        help='本次运行的文件夹名: data/gfn_logs/pool_{pool_capacity}/{name}/')
    parser.add_argument('--log_freq', type=int, default=100)
    parser.add_argument('--update_freq', type=int, default=1)
    parser.add_argument('--n_trajectories', type=int, default=32)
    parser.add_argument('--n_episodes', type=int, default=256)
    parser.add_argument('--encoder_type', type=str, default='gnn', choices=['transformer', 'lstm', 'gnn'])
    parser.add_argument('--entropy_coef', type=float, default=0.01, help='Coefficient for entropy regularization')
    parser.add_argument('--entropy_temperature', type=float, default=1.0, help='Temperature for entropy calculation')
    parser.add_argument('--mask_dropout_prob', type=float, default=0.5, help='Probability of masking out valid actions based on expression length')
    parser.add_argument('--ssl_weight', type=float, default=0.5, help='Initial weight for SSL reward (will decay during training)')
    parser.add_argument('--nov_weight', type=float, default=0.5, help='Initial weight for novelty reward (will decay during training)')
    parser.add_argument('--weight_decay_type', type=str, default='linear', choices=['linear', 'exponential', 'polynomial'], help='Type of weight decay to apply')
    parser.add_argument('--final_weight_ratio', type=float, default=0.0, help='Final weight ratio in [0, 1]; exponential requires > 0 (e.g., 0.1)')
    # 用于增量训练，指定恢复训练的检查点目录路径
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint directory to resume training from (e.g. data/gfn_logs/pool_10/xxx/)')
    args = parser.parse_args()
    print(args)
    train(args)
