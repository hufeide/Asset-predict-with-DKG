# -*- coding: utf-8 -*-
"""
真实数据驱动的高级事件DKG（安全版本 - 无外部数据依赖）
包含真实经济数据、大类资产数据和突发事件建模
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, TransformerConv
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import datetime
from typing import List, Dict, Tuple, Optional
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class RealFinancialEventTypes:
    """真实金融事件类型定义"""
    MONETARY_POLICY = 0      # 货币政策（利率决议）
    INFLATION_SHOCK = 1      # 通胀冲击（CPI/PPI异常）
    ECONOMIC_DATA = 2        # 经济数据发布（PMI、就业等）
    GEOPOLITICAL = 3         # 地缘政治事件
    COMMODITY_SHOCK = 4      # 大宗商品冲击
    MARKET_CRASH = 5         # 市场崩盘
    LIQUIDITY_CRISIS = 6     # 流动性危机
    SAFE_HAVEN_FLOW = 7      # 避险资金流动
    CENTRAL_BANK_ACTION = 8  # 央行行动
    PANDEMIC_IMPACT = 9      # 疫情影响

class AssetTypes:
    """资产类型定义"""
    EQUITY = 0      # 股票 (S&P500)
    BOND = 1        # 债券 (10Y Treasury)
    COMMODITY = 2   # 大宗商品 (原油)
    PRECIOUS_METAL = 3  # 贵金属 (黄金)
    CURRENCY = 4    # 货币 (美元指数)
    VOLATILITY = 5  # 波动率 (VIX)

class AdvancedEventDKG(nn.Module):
    """
    高级事件驱动动态知识图谱
    """
    def __init__(self, num_nodes: int, node_dim: int, hidden_dim: int, 
                 event_dim: int, num_event_types: int = 10):
        super(AdvancedEventDKG, self).__init__()
        
        self.num_nodes = num_nodes
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.event_dim = event_dim
        self.num_event_types = num_event_types
        
        # 节点类型嵌入
        self.node_type_embedding = nn.Embedding(num_nodes, node_dim)
        
        # 多层事件编码器
        self.event_encoder = nn.Sequential(
            nn.Linear(event_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # 事件类型嵌入
        self.event_type_embedding = nn.Embedding(num_event_types, hidden_dim)
        
        # 多头注意力机制
        self.transformer = TransformerConv(
            in_channels=node_dim + hidden_dim,
            out_channels=hidden_dim,
            heads=4,
            concat=False,
            dropout=0.1
        )
        
        # 图注意力网络层
        self.gat_layers = nn.ModuleList([
            GATConv(hidden_dim, hidden_dim, heads=4, concat=False, dropout=0.1)
            for _ in range(2)
        ])
        
        # 时序记忆模块（LSTM风格）
        self.memory_cell = nn.LSTMCell(hidden_dim, hidden_dim)
        
        # 关系预测器（预测节点间关系强度）
        self.relation_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),  # 正相关、负相关、无关系
            nn.Softmax(dim=-1)
        )
        
        # 风险评估器
        self.risk_assessor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # 多任务预测头
        self.price_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.volatility_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # 初始化记忆状态
        self.hidden_state = None
        self.cell_state = None
        
    def reset_memory(self):
        """重置记忆状态"""
        self.hidden_state = None
        self.cell_state = None
        
    def detect_complex_events(self, features: torch.Tensor, 
                            historical_features: List[torch.Tensor] = None) -> Dict:
        """
        检测复杂金融事件（基于真实数据特征）
        """
        events = {}
        
        # 确保features有足够的维度
        if len(features) < 9:
            return events
            
        # 提取各类资产和经济指标变化
        sp500_change = abs(features[0])      # S&P500变化
        bond_change = abs(features[1])       # 债券收益率变化  
        oil_change = abs(features[2])        # 原油变化
        gold_change = abs(features[3])       # 黄金变化
        usd_change = abs(features[4])        # 美元指数变化
        vix_change = abs(features[5])        # VIX变化
        pmi_change = abs(features[6])        # PMI变化
        cpi_change = abs(features[7])        # 通胀变化
        unemployment_change = abs(features[8])  # 失业率变化
        
        # ===== 原有事件类型 =====
        
        # 1. 货币政策事件（基于利率和债券收益率变化）
        rate_change = bond_change  # 使用债券收益率作为利率代理
        if rate_change > 0.005:  # 50个基点
            events[RealFinancialEventTypes.MONETARY_POLICY] = min(rate_change * 10, 1.0)
        
        # 2. 市场冲击事件
        market_change = sp500_change
        if market_change > 0.03:  # 3%以上变动
            events[RealFinancialEventTypes.MARKET_CRASH] = min(market_change * 5, 1.0)
        
        # 3. 波动率飙升
        if historical_features and len(historical_features) >= 5:
            recent_volatility = torch.std(torch.stack(historical_features[-5:]))
            if recent_volatility > 0.02:
                # 使用新的事件类型，但保持相同逻辑
                events[RealFinancialEventTypes.MARKET_CRASH] = max(
                    events.get(RealFinancialEventTypes.MARKET_CRASH, 0),
                    min(float(recent_volatility) * 10, 1.0)
                )
        
        # 4. 经济指标异常
        if bond_change > 0.01:
            events[RealFinancialEventTypes.ECONOMIC_DATA] = min(bond_change * 20, 1.0)
        
        # 5. 流动性危机（多个指标同时异常）
        if rate_change > 0.01 and market_change > 0.05 and bond_change > 0.02:
            events[RealFinancialEventTypes.LIQUIDITY_CRISIS] = 0.8
        
        # ===== 新增真实数据事件类型 =====
        
        # 6. 通胀冲击检测
        if cpi_change > 0.002:  # 通胀年率变化超过0.2%
            events[RealFinancialEventTypes.INFLATION_SHOCK] = min(cpi_change * 50, 1.0)
        
        # 7. 经济数据异常（PMI和失业率）
        if pmi_change > 2.0 or unemployment_change > 0.5:
            # 与上面的经济指标异常合并，取最大值
            econ_intensity = min((pmi_change/10 + unemployment_change*2), 1.0)
            events[RealFinancialEventTypes.ECONOMIC_DATA] = max(
                events.get(RealFinancialEventTypes.ECONOMIC_DATA, 0),
                econ_intensity
            )
        
        # 8. 大宗商品冲击
        if oil_change > 0.05:  # 原油变化超过5%
            events[RealFinancialEventTypes.COMMODITY_SHOCK] = min(oil_change * 10, 1.0)
        
        # 9. 避险资金流动
        if gold_change > 0.02 and bond_change > 0.1:  # 黄金和债券同时上涨
            events[RealFinancialEventTypes.SAFE_HAVEN_FLOW] = min((gold_change + bond_change) * 10, 1.0)
        
        # 10. 地缘政治事件（原油和黄金同时大涨）
        if oil_change > 0.03 and gold_change > 0.02 and sp500_change > 0.02:
            events[RealFinancialEventTypes.GEOPOLITICAL] = min((oil_change + gold_change) * 15, 1.0)
        
        # 11. 央行行动（美元指数异常波动）
        if usd_change > 0.015:
            events[RealFinancialEventTypes.CENTRAL_BANK_ACTION] = min(usd_change * 30, 1.0)
        
        # 12. 综合流动性危机检测（多资产同时异常波动）
        crisis_score = (sp500_change > 0.04) + (bond_change > 0.15) + (vix_change > 8) + (usd_change > 0.02)
        if crisis_score >= 3:
            # 与上面的流动性危机合并，取最大值
            events[RealFinancialEventTypes.LIQUIDITY_CRISIS] = max(
                events.get(RealFinancialEventTypes.LIQUIDITY_CRISIS, 0),
                0.9
            )
        
        return events
        
    def build_dynamic_graph(self, node_states: torch.Tensor, 
                          event_info: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        构建基于真实资产关系的动态图结构
        节点顺序: [股票, 债券, 原油, 黄金, 美元, VIX]
        """
        num_nodes = node_states.size(0)
        
        # 基于真实金融市场关系的基础连接矩阵
        base_connections = torch.tensor([
            #  股票  债券  原油  黄金  美元  VIX
            [0.0, 0.6, 0.4, 0.3, 0.5, 0.8],  # 股票
            [0.6, 0.0, 0.2, 0.4, 0.7, 0.5],  # 债券  
            [0.4, 0.2, 0.0, 0.3, 0.6, 0.4],  # 原油
            [0.3, 0.4, 0.3, 0.0, 0.5, 0.3],  # 黄金
            [0.5, 0.7, 0.6, 0.5, 0.0, 0.4],  # 美元
            [0.8, 0.5, 0.4, 0.3, 0.4, 0.0]   # VIX
        ], dtype=torch.float)
        
        # 事件驱动的连接调整
        for event_type, intensity in event_info.items():
            # ===== 原有经典事件类型 =====
            if event_type == RealFinancialEventTypes.MONETARY_POLICY:
                # 货币政策主要影响Fed->其他的连接（这里用债券节点作为Fed代理）
                base_connections[1, :] *= (1 + intensity)  # 债券->所有其他资产
                
            elif event_type == RealFinancialEventTypes.MARKET_CRASH:
                # 市场冲击增强所有连接
                base_connections *= (1 + intensity * 0.5)
                
            # 波动率飙升增强市场内部连接（股票与VIX的关系）
            elif event_type == RealFinancialEventTypes.MARKET_CRASH and intensity > 0.5:
                # 当市场崩盘强度较高时，视为波动率飙升
                base_connections[0, 5] *= (1 + intensity)  # 股票-VIX
                base_connections[5, 0] *= (1 + intensity)  # VIX-股票
                
            # ===== 新增真实数据事件类型 =====
            elif event_type == RealFinancialEventTypes.INFLATION_SHOCK:
                # 通胀冲击增强商品与股债的负相关
                base_connections[0, 2] *= (1 + intensity * 0.3)  # 股票-原油
                base_connections[0, 3] *= (1 + intensity * 0.3)  # 股票-黄金
                
            elif event_type == RealFinancialEventTypes.ECONOMIC_DATA:
                # 经济指标异常影响债券市场
                base_connections[1, :] *= (1 + intensity * 0.3)  # 债券与其他资产
                
            elif event_type == RealFinancialEventTypes.GEOPOLITICAL:
                # 地缘政治增强避险资产间连接
                base_connections[1, 3] *= (1 + intensity * 0.4)  # 债券-黄金
                base_connections[3, 1] *= (1 + intensity * 0.4)
                
            elif event_type == RealFinancialEventTypes.SAFE_HAVEN_FLOW:
                # 避险流动增强安全资产间连接
                base_connections[1, 3] *= (1 + intensity * 0.5)  # 债券-黄金
                base_connections[3, 1] *= (1 + intensity * 0.5)
                
            elif event_type == RealFinancialEventTypes.COMMODITY_SHOCK:
                # 商品冲击增强原油与其他资产连接
                base_connections[2, :] *= (1 + intensity * 0.4)
                base_connections[:, 2] *= (1 + intensity * 0.4)
                
            elif event_type == RealFinancialEventTypes.LIQUIDITY_CRISIS:
                # 流动性危机（多个指标同时异常）- 增强所有风险资产相关性
                # 影响股票、原油、美元的相互关系
                base_connections[0, 2] *= (1 + intensity * 0.6)  # 股票-原油
                base_connections[0, 4] *= (1 + intensity * 0.6)  # 股票-美元
                base_connections[2, 4] *= (1 + intensity * 0.6)  # 原油-美元
                
            elif event_type == RealFinancialEventTypes.CENTRAL_BANK_ACTION:
                # 央行行动主要影响货币和债券市场
                base_connections[1, 4] *= (1 + intensity * 0.7)  # 债券-美元
                base_connections[4, 1] *= (1 + intensity * 0.7)  # 美元-债券
                
            elif event_type == RealFinancialEventTypes.PANDEMIC_IMPACT:
                # 疫情影响增强所有资产的相关性（系统性风险）
                base_connections *= (1 + intensity * 0.4)
        
        # 生成边（使用阈值过滤）
        edge_index = []
        edge_attr = []
        threshold = 0.25
        
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j and base_connections[i, j] > threshold:
                    edge_index.append([i, j])
                    edge_attr.append(base_connections[i, j].item())
        
        # 确保图连通性
        if len(edge_index) == 0:
            # 创建最小连接（股票为中心的星型图）
            for i in range(1, num_nodes):
                edge_index.extend([[0, i], [i, 0]])
                edge_attr.extend([0.5, 0.5])
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        
        return edge_index, edge_attr
        
    def forward(self, node_ids: torch.Tensor, event_features: torch.Tensor,
                event_types: List[int], historical_features: List[torch.Tensor] = None):
        """
        前向传播
        """
        # 节点嵌入
        node_emb = self.node_type_embedding(node_ids)
        
        # 检测复杂事件
        event_info = self.detect_complex_events(event_features, historical_features)
        
        # 编码事件特征
        event_encoded = self.event_encoder(event_features.unsqueeze(0))
        
        # 事件类型嵌入
        if event_types:
            event_type_embs = []
            for et in event_types:
                if et in event_info:
                    emb = self.event_type_embedding(torch.tensor(et))
                    event_type_embs.append(emb * event_info[et])
            
            if event_type_embs:
                event_type_combined = torch.stack(event_type_embs).mean(dim=0)
                event_encoded = event_encoded + event_type_combined.unsqueeze(0)
        
        # 广播事件特征到所有节点
        event_broadcast = event_encoded.expand(self.num_nodes, -1)
        
        # 组合节点和事件特征
        combined_features = torch.cat([node_emb, event_broadcast], dim=1)
        
        # 构建动态图
        edge_index, edge_attr = self.build_dynamic_graph(node_emb, event_info)
        
        # Transformer层
        x = self.transformer(combined_features, edge_index)
        
        # GAT层
        for gat_layer in self.gat_layers:
            x = F.relu(gat_layer(x, edge_index))
            
        # 时序记忆更新
        if self.hidden_state is None:
            self.hidden_state = torch.zeros(self.num_nodes, self.hidden_dim)
            self.cell_state = torch.zeros(self.num_nodes, self.hidden_dim)
            
        # 对每个节点更新LSTM状态
        new_hidden = []
        new_cell = []
        
        for i in range(self.num_nodes):
            h_new, c_new = self.memory_cell(x[i].unsqueeze(0), 
                                          (self.hidden_state[i].unsqueeze(0), 
                                           self.cell_state[i].unsqueeze(0)))
            new_hidden.append(h_new.squeeze(0))
            new_cell.append(c_new.squeeze(0))
            
        new_hidden = torch.stack(new_hidden)
        new_cell = torch.stack(new_cell)
        
        return new_hidden, new_cell, edge_index, event_info

import pandas as pd
import numpy as np
import time

class RealDataCollector:
    """
    真实数据收集器 - 从实际API获取金融数据
    """
    def __init__(self):
        self.major_events = self._define_major_events()
        # 检查数据源可用性
        try:
            import yfinance as yf
            from pandas_datareader import data as pdr
            self.yf = yf
            self.pdr = pdr
            self.data_available = True
            print("✓ 真实数据源可用 (yfinance + pandas_datareader)")
        except ImportError as e:
            print(f"⚠ 真实数据源不可用: {e}")
            print("  将使用基于历史模式的高质量模拟数据")
            self.data_available = False
        
    def _define_major_events(self) -> Dict:
        """定义重大历史事件"""
        return {
            '2020-03-15': {'type': RealFinancialEventTypes.PANDEMIC_IMPACT, 'intensity': 1.0, 'description': 'COVID-19 Pandemic Declaration'},
            '2020-03-23': {'type': RealFinancialEventTypes.MARKET_CRASH, 'intensity': 0.9, 'description': 'Market Bottom'},
            '2020-04-20': {'type': RealFinancialEventTypes.COMMODITY_SHOCK, 'intensity': 1.0, 'description': 'Oil Price Negative'},
            '2021-03-10': {'type': RealFinancialEventTypes.INFLATION_SHOCK, 'intensity': 0.6, 'description': 'Inflation Concerns Rise'},
            '2022-02-24': {'type': RealFinancialEventTypes.GEOPOLITICAL, 'intensity': 0.8, 'description': 'Russia-Ukraine Conflict'},
            '2022-03-16': {'type': RealFinancialEventTypes.MONETARY_POLICY, 'intensity': 0.7, 'description': 'Fed Rate Hike Cycle Begins'},
            '2023-03-10': {'type': RealFinancialEventTypes.LIQUIDITY_CRISIS, 'intensity': 0.8, 'description': 'Silicon Valley Bank Collapse'},
        }

    def collect_real_data(self, start_date='2000-01-01', end_date='2024-12-31', block_years=5) -> pd.DataFrame:
        """
        分块下载真实金融和经济数据（月度）
        """
        if not self.data_available:
            raise RuntimeError("真实数据源不可用")

        tickers = {
            'SP500': '^GSPC',
            'Treasury_10Y': 'DGS10',  # 添加10年期美债收益率
            'Oil_WTI': 'CL=F',
            'Gold': 'GC=F',
            'USD_Index': 'DX-Y.NYB',
            'VIX': '^VIX'
        }

        econ_indicators = {
            'PMI': 'MANEMP',
            'CPI_YoY': 'CPIAUCSL',
            'Unemployment': 'UNRATE'
        }

        # 生成时间段块
        start_year = pd.Timestamp(start_date).year
        end_year = pd.Timestamp(end_date).year
        blocks = [(str(y), str(min(y + block_years - 1, end_year))) for y in range(start_year, end_year + 1, block_years)]

        data_dict = {name: [] for name in tickers.keys()}
        econ_dict = {name: [] for name in econ_indicators.keys()}

        for start_y, end_y in blocks:
            block_start = f"{start_y}-01-01"
            block_end = f"{end_y}-12-31"
            print(f"⏳ 下载区间: {block_start} 到 {block_end}")

            # 下载资产数据
            for name, ticker in tickers.items():
                for attempt in range(3):
                    try:
                        df = self.yf.download(ticker, start=block_start, end=block_end, progress=False, threads=False)
                        monthly = df['Adj Close'].resample('M').last()
                        data_dict[name].append(monthly)
                        print(f"  ✓ {name} 区块下载成功: {len(monthly)} 个数据点")
                        time.sleep(2)  # 避免短时间连续请求
                        break
                    except Exception as e:
                        print(f"  ⚠ {name} 下载失败 (尝试 {attempt+1}): {e}")
                        if attempt < 2:
                            time.sleep(5)

            # 下载经济指标
            for name, code in econ_indicators.items():
                try:
                    df = self.pdr.DataReader(code, 'fred', block_start, block_end)
                    if name == 'CPI_YoY':
                        df = df.pct_change(12)
                    monthly = df.squeeze().resample('M').last()
                    econ_dict[name].append(monthly)
                except Exception as e:
                    print(f"  ⚠ {name} 下载失败: {e}")

        # 合并所有块
        for name in tickers.keys():
            data_dict[name] = pd.concat(data_dict[name])
        for name in econ_indicators.keys():
            econ_dict[name] = pd.concat(econ_dict[name])

        all_data = pd.DataFrame({**data_dict, **econ_dict}).fillna(method='ffill').dropna()
        print(f"✅ 数据处理完成: {len(all_data)} 个月度数据点")
        print(f"📊 包含资产/指标: {list(all_data.columns)}")
        return all_data


    def _generate_fallback_data(self, start_date, end_date) -> pd.DataFrame:
        """
        生成基于真实历史模式的高质量备用数据
        """
        print("  📊 生成基于2020-2023真实历史模式的数据...")
        
        dates = pd.date_range(start=start_date, end=end_date, freq='W')
        n_periods = len(dates)
        
        np.random.seed(42)
        
        # 1. S&P 500 - 基于COVID-19和后续恢复的真实模式
        sp500_base = 3200
        covid_impact = np.concatenate([
            np.linspace(0, -0.35, 10),      # 2020年3月崩盘
            np.linspace(-0.35, 0.5, 40),    # 强劲恢复
            np.linspace(0.5, 0.2, 50),      # 继续上涨但放缓
            np.linspace(0.2, -0.1, n_periods-100) if n_periods > 100 else np.linspace(0.2, 0.1, n_periods-100)
        ])
        sp500_noise = np.random.normal(0, 0.02, n_periods)
        sp500_returns = covid_impact + sp500_noise
        sp500_data = sp500_base * np.cumprod(1 + sp500_returns)
        
        # 2. 10年期美债收益率 - 反映真实的货币政策周期
        treasury_base = np.concatenate([
            np.full(20, 0.7),               # 2020年超低利率
            np.linspace(0.7, 1.8, 30),      # 2021年缓慢上升
            np.linspace(1.8, 4.5, 40),      # 2022年快速加息
            np.linspace(4.5, 3.8, n_periods-90) if n_periods > 90 else [3.8] * max(1, n_periods-90)
        ])
        treasury_noise = np.random.normal(0, 0.1, n_periods)
        treasury_data = treasury_base + treasury_noise
        
        # 3. WTI原油 - 包含2020年负油价事件
        oil_base = 60
        oil_shocks = np.zeros(n_periods)
        if n_periods > 15:
            oil_shocks[15:20] = -1.2  # 2020年4月负油价
        if n_periods > 100:
            ukraine_start = int(n_periods * 0.5)
            oil_shocks[ukraine_start:ukraine_start+10] = 0.8  # 俄乌冲突
        
        oil_trend = np.random.normal(0.001, 0.03, n_periods)
        oil_returns = oil_trend + oil_shocks
        oil_data = oil_base * np.cumprod(1 + oil_returns)
        oil_data = np.clip(oil_data, 10, 150)
        
        # 4. 黄金 - 避险资产特性
        gold_base = 1800
        gold_safe_haven = np.zeros(n_periods)
        if n_periods > 25:
            gold_safe_haven[5:25] = 0.3   # COVID期间避险需求
        if n_periods > 100:
            ukraine_start = int(n_periods * 0.5)
            gold_safe_haven[ukraine_start:ukraine_start+15] = 0.2
        
        gold_trend = np.random.normal(0, 0.01, n_periods)
        gold_returns = gold_trend + gold_safe_haven * 0.1
        gold_data = gold_base * np.cumprod(1 + gold_returns)
        
        # 5. 美元指数 - 反映美国相对强势
        usd_base = 100
        usd_trend = np.concatenate([
            np.linspace(0, -0.1, min(50, n_periods//2)),       # 初期走弱
            np.linspace(-0.1, 0.2, min(60, n_periods//2)),     # 加息周期走强
            np.linspace(0.2, 0.1, max(1, n_periods-110)) if n_periods > 110 else [0.1]
        ])
        usd_noise = np.random.normal(0, 0.005, n_periods)
        usd_data = usd_base * (1 + usd_trend + usd_noise)
        
        # 6. VIX - 恐慌指数
        vix_spikes = np.zeros(n_periods)
        if n_periods > 15:
            vix_spikes[8:15] = 40    # COVID恐慌
        if n_periods > 100:
            ukraine_start = int(n_periods * 0.5)
            vix_spikes[ukraine_start:ukraine_start+5] = 20
        
        vix_baseline = 15 + 10 * np.abs(sp500_returns)
        vix_data = vix_baseline + vix_spikes + np.random.normal(0, 3, n_periods)
        vix_data = np.clip(vix_data, 10, 80)
        
        # 7. 经济指标 - 基于真实经济周期
        # PMI
        pmi_cycle = np.concatenate([
            np.linspace(50, 35, min(15, n_periods//4)),         # COVID冲击
            np.linspace(35, 60, min(25, n_periods//3)),         # 强劲恢复
            np.linspace(60, 48, max(1, n_periods-40)) if n_periods > 40 else [48]
        ])
        pmi_noise = np.random.normal(0, 2, n_periods)
        pmi_data = pmi_cycle + pmi_noise
        
        # CPI - 通胀飙升和回落
        cpi_surge = np.concatenate([
            np.full(min(30, n_periods//3), 0.01),               # 低通胀期
            np.linspace(0.01, 0.09, min(40, n_periods//2)),     # 通胀飙升
            np.linspace(0.09, 0.03, max(1, n_periods-70)) if n_periods > 70 else [0.03]
        ])
        cpi_noise = np.random.normal(0, 0.005, n_periods)
        cpi_data = cpi_surge + cpi_noise
        
        # 失业率 - COVID冲击和恢复
        unemployment_base = np.concatenate([
            np.linspace(3.5, 14.8, min(10, n_periods//6)),     # COVID失业率飙升
            np.linspace(14.8, 3.4, min(50, n_periods//2)),     # 快速恢复
            np.linspace(3.4, 4.0, max(1, n_periods-60)) if n_periods > 60 else [4.0]
        ])
        unemployment_noise = np.random.normal(0, 0.2, n_periods)
        unemployment_data = unemployment_base + unemployment_noise
        
        return pd.DataFrame({
            'SP500': sp500_data,
            'Treasury_10Y': treasury_data,
            'Oil_WTI': oil_data,
            'Gold': gold_data,
            'USD_Index': usd_data,
            'VIX': vix_data,
            'PMI': pmi_data,
            'CPI_YoY': cpi_data,
            'Unemployment': unemployment_data
        }, index=dates)
    
    def add_event_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加事件特征"""
        df_with_events = df.copy()
        
        # 初始化事件特征
        for event_type in range(10):  # 10种事件类型
            df_with_events[f'event_{event_type}'] = 0.0
        
        # 添加历史重大事件
        for date_str, event_info in self.major_events.items():
            try:
                event_date = pd.to_datetime(date_str)
                # 找到最接近的数据点
                closest_idx = df_with_events.index.get_indexer([event_date], method='nearest')[0]
                if closest_idx >= 0:
                    event_col = f'event_{event_info["type"]}'
                    df_with_events.iloc[closest_idx, df_with_events.columns.get_loc(event_col)] = event_info['intensity']
                    
                    # 事件影响持续几个周期
                    for i in range(1, 4):  # 影响后续3周
                        if closest_idx + i < len(df_with_events):
                            decay_intensity = event_info['intensity'] * (0.7 ** i)
                            df_with_events.iloc[closest_idx + i, df_with_events.columns.get_loc(event_col)] = decay_intensity
            except:
                continue
        
        return df_with_events

def main():
    """
    主函数 - 使用真实数据或高质量模拟数据
    """
    print("=== 增强版真实数据驱动的事件DKG模型 ===")
    
    # 1. 收集真实数据或使用高质量模拟数据
    data_collector = RealDataCollector()
    if 1==2:
        df = data_collector.collect_real_data(start_date='2020-01-01', end_date='2023-12-31')
    else:
        file_path = "/home/aixz/data/hxf/bigproject/FinDKG-main/fin_dkg/financial_data_2000_2024.csv"
        df = pd.read_csv(
            file_path, 
            index_col=0, 
            parse_dates=True
        )
    print(f"数据时间范围: {df.index[0]} 到 {df.index[-1]}")
    print(f"数据点数量: {len(df)}")
    print("\n数据统计:")
    print(df.describe())
    
    # 添加事件特征
    df_with_events = data_collector.add_event_features(df)
    print(f"添加事件特征后: {df_with_events.shape[1]} 个特征")
    
    # 计算收益率
    asset_columns = ['SP500', 'Treasury_10Y', 'Oil_WTI', 'Gold', 'USD_Index', 'VIX']
    returns = df_with_events[asset_columns].pct_change().dropna()
    
    # 添加经济指标变化率
    econ_columns = ['PMI', 'CPI_YoY', 'Unemployment']
    econ_changes = df_with_events[econ_columns].pct_change().dropna()
    
    # 合并所有特征
    all_features = pd.concat([returns, econ_changes], axis=1).dropna()
    
    # 添加事件特征
    event_columns = [col for col in df_with_events.columns if col.startswith('event_')]
    event_features = df_with_events[event_columns].loc[all_features.index]
    
    # 最终特征矩阵
    final_features = pd.concat([all_features, event_features], axis=1).fillna(0)
    
    print(f"最终特征维度: {final_features.shape}")
    
    # 2. 模型参数
    num_nodes = 6  # 股票, 债券, 原油, 黄金, 美元, VIX
    node_dim = 32
    hidden_dim = 64
    event_dim = final_features.shape[1]
    num_event_types = 10
    
    # 3. 初始化模型
    model = AdvancedEventDKG(num_nodes, node_dim, hidden_dim, event_dim, num_event_types)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # 多任务损失
    price_criterion = nn.BCELoss()
    volatility_criterion = nn.MSELoss()
    
    # 4. 准备训练数据
    node_ids = torch.arange(num_nodes)
    
    # 多任务目标
    sp500_targets = (returns['SP500'].shift(-1) > 0).astype(int).dropna()
    oil_targets = (returns['Oil_WTI'].shift(-1) > 0).astype(int).dropna()
    gold_targets = (returns['Gold'].shift(-1) > 0).astype(int).dropna()
    
    # 波动率目标
    volatility_targets = returns['SP500'].rolling(5).std().shift(-1).dropna()
    volatility_targets = (volatility_targets - volatility_targets.min()) / (volatility_targets.max() - volatility_targets.min())
    
    # 对齐数据长度
    min_length = min(len(sp500_targets), len(oil_targets), len(gold_targets), 
                    len(volatility_targets), len(final_features)-1)
    
    # 5. 训练
    print("\n开始训练真实历史模式模型...")
    epochs = 30
    historical_features = []
    
    # 训练/验证分割
    train_size = int(0.8 * min_length)
    
    for epoch in range(epochs):
        model.reset_memory()
        epoch_loss = 0
        epoch_sp500_acc = 0
        epoch_oil_acc = 0
        epoch_gold_acc = 0
        epoch_vol_loss = 0
        
        historical_features = []  # 重置历史特征
        
        for i in range(train_size):
            # 当前特征（包含所有资产变化和事件特征）
            current_features = torch.tensor(final_features.iloc[i].values, dtype=torch.float)
            
            historical_features.append(current_features)
            
            # 动态事件类型检测
            event_types = []
            if abs(current_features[1]) > 0.01:  # 债券收益率大幅变动
                event_types.append(RealFinancialEventTypes.MONETARY_POLICY)
            if abs(current_features[0]) > 0.03:  # 股市大幅波动
                event_types.append(RealFinancialEventTypes.MARKET_CRASH)
            if abs(current_features[2]) > 0.05:  # 原油大幅波动
                event_types.append(RealFinancialEventTypes.COMMODITY_SHOCK)
            if abs(current_features[3]) > 0.02:  # 黄金大幅波动
                event_types.append(RealFinancialEventTypes.SAFE_HAVEN_FLOW)
                
            # 前向传播
            hidden_states, cell_states, edge_index, event_info = model(
                node_ids, current_features, event_types, historical_features[-10:]
            )
            
            # 多任务预测
            sp500_pred = model.price_predictor(hidden_states[0]).squeeze()  # 股票节点
            oil_pred = model.price_predictor(hidden_states[2]).squeeze()    # 原油节点
            gold_pred = model.price_predictor(hidden_states[3]).squeeze()   # 黄金节点
            vol_pred = model.volatility_predictor(hidden_states[0]).squeeze()  # 基于股票节点预测波动率
            
            # 目标
            sp500_target = torch.tensor(sp500_targets.iloc[i], dtype=torch.float)
            oil_target = torch.tensor(oil_targets.iloc[i], dtype=torch.float)
            gold_target = torch.tensor(gold_targets.iloc[i], dtype=torch.float)
            vol_target = torch.tensor(volatility_targets.iloc[i], dtype=torch.float)
            
            # 多任务损失
            sp500_loss = price_criterion(sp500_pred, sp500_target)
            oil_loss = price_criterion(oil_pred, oil_target)
            gold_loss = price_criterion(gold_pred, gold_target)
            vol_loss = volatility_criterion(vol_pred, vol_target)
            
            total_loss = sp500_loss + 0.5 * oil_loss + 0.5 * gold_loss + 0.3 * vol_loss
            
            # 反向传播
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # 更新模型状态
            model.hidden_state = hidden_states.detach()
            model.cell_state = cell_states.detach()
            
            # 统计
            epoch_loss += total_loss.item()
            epoch_sp500_acc += int((sp500_pred > 0.5) == (sp500_target > 0.5))
            epoch_oil_acc += int((oil_pred > 0.5) == (oil_target > 0.5))
            epoch_gold_acc += int((gold_pred > 0.5) == (gold_target > 0.5))
            epoch_vol_loss += vol_loss.item()
        
        if (epoch + 1) % 10 == 0:
            sp500_acc = epoch_sp500_acc / train_size
            oil_acc = epoch_oil_acc / train_size
            gold_acc = epoch_gold_acc / train_size
            print(f"Epoch {epoch+1}:")
            print(f"  Loss: {epoch_loss:.4f}, Vol Loss: {epoch_vol_loss:.4f}")
            print(f"  Accuracy - SP500: {sp500_acc:.3f}, Oil: {oil_acc:.3f}, Gold: {gold_acc:.3f}")
    
    print("\n训练完成！")
    
    # 6. 可视化结果


if __name__ == "__main__":
    main()
