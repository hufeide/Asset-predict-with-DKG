# -*- coding: utf-8 -*-
"""
高级事件驱动动态知识图谱（Advanced Event-Driven DKG）
包含更复杂的金融事件和关系建模
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
from typing import List, Dict, Tuple
import seaborn as sns

class FinancialEventTypes:
    """金融事件类型定义"""
    MONETARY_POLICY = 0      # 货币政策
    MARKET_SHOCK = 1         # 市场冲击
    EARNINGS_RELEASE = 2     # 财报发布
    GEOPOLITICAL = 3         # 地缘政治
    ECONOMIC_INDICATOR = 4   # 经济指标
    VOLATILITY_SPIKE = 5     # 波动率飙升
    LIQUIDITY_CRISIS = 6     # 流动性危机
    SECTOR_ROTATION = 7      # 板块轮动

class AdvancedEventDKG(nn.Module):
    """
    高级事件驱动动态知识图谱
    """
    def __init__(self, num_nodes: int, node_dim: int, hidden_dim: int, 
                 event_dim: int, num_event_types: int = 8):
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
        检测复杂金融事件
        """
        events = {}
        
        # 基础事件检测
        rate_change = abs(features[0])
        market_change = abs(features[1]) 
        bond_change = abs(features[2])
        
        # 货币政策事件
        if rate_change > 0.005:  # 50个基点
            events[FinancialEventTypes.MONETARY_POLICY] = min(rate_change * 10, 1.0)
            
        # 市场冲击事件
        if market_change > 0.03:  # 3%以上变动
            events[FinancialEventTypes.MARKET_SHOCK] = min(market_change * 5, 1.0)
            
        # 波动率飙升
        if historical_features and len(historical_features) >= 5:
            recent_volatility = torch.std(torch.stack(historical_features[-5:]))
            if recent_volatility > 0.02:
                events[FinancialEventTypes.VOLATILITY_SPIKE] = min(float(recent_volatility) * 10, 1.0)
        
        # 经济指标异常
        if bond_change > 0.01:
            events[FinancialEventTypes.ECONOMIC_INDICATOR] = min(bond_change * 20, 1.0)
            
        # 流动性危机（多个指标同时异常）
        if rate_change > 0.01 and market_change > 0.05 and bond_change > 0.02:
            events[FinancialEventTypes.LIQUIDITY_CRISIS] = 0.8
            
        return events
        
    def build_dynamic_graph(self, node_states: torch.Tensor, 
                          event_info: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        构建动态图结构
        """
        num_nodes = node_states.size(0)
        
        # 基础连接矩阵
        base_connections = torch.ones(num_nodes, num_nodes) * 0.1
        
        # 事件驱动的连接调整
        for event_type, intensity in event_info.items():
            if event_type == FinancialEventTypes.MONETARY_POLICY:
                # 货币政策主要影响Fed->其他的连接
                base_connections[0, :] *= (1 + intensity)
                
            elif event_type == FinancialEventTypes.MARKET_SHOCK:
                # 市场冲击增强所有连接
                base_connections *= (1 + intensity * 0.5)
                
            elif event_type == FinancialEventTypes.VOLATILITY_SPIKE:
                # 波动率飙升增强市场内部连接
                base_connections[1, 2] *= (1 + intensity)
                base_connections[2, 1] *= (1 + intensity)
                
        # 生成边
        edge_index = []
        edge_attr = []
        
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j and base_connections[i, j] > 0.2:
                    edge_index.append([i, j])
                    edge_attr.append(base_connections[i, j].item())
                    
        if len(edge_index) == 0:
            # 保证最小连接
            edge_index = [[0, 1], [1, 0], [0, 2], [2, 0], [1, 2], [2, 1]]
            edge_attr = [0.5] * 6
            
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

class AdvancedFinancialDataGenerator:
    """
    高级金融数据生成器
    """
    def __init__(self):
        self.volatility_regimes = ['low', 'medium', 'high', 'crisis']
        
    def generate_realistic_data(self, start_date='2020-01-01', end_date='2023-12-31'):
        """
        生成更真实的金融数据
        """
        dates = pd.date_range(start=start_date, end=end_date, freq='W')  # 周频数据
        n_periods = len(dates)
        
        np.random.seed(42)
        
        # 模拟不同的市场制度
        regime_changes = np.random.choice([0, 1], size=n_periods, p=[0.95, 0.05])
        current_regime = 0
        
        # 联邦基金利率（考虑政策周期）
        rate_base = np.zeros(n_periods)
        policy_cycles = [
            (0, 50, 0.25, 0.1),      # 低利率期
            (50, 100, 0.1, 2.0),     # 加息周期
            (100, 150, 2.0, 4.5),    # 高利率期
            (150, n_periods, 4.5, 3.0)  # 降息周期
        ]
        
        for start_idx, end_idx, start_rate, end_rate in policy_cycles:
            if start_idx < n_periods:
                end_idx = min(end_idx, n_periods)
                rate_base[start_idx:end_idx] = np.linspace(start_rate, end_rate, end_idx - start_idx)
        
        rate_noise = np.random.normal(0, 0.05, n_periods)
        rate_data = rate_base + rate_noise
        
        # S&P500（包含趋势、周期和冲击）
        sp500_base = 3500
        trend = np.linspace(0, 0.3, n_periods)  # 长期上涨趋势
        
        # 添加市场周期
        cycle = 0.1 * np.sin(2 * np.pi * np.arange(n_periods) / 52)  # 年度周期
        
        # 添加随机冲击
        shocks = np.zeros(n_periods)
        shock_indices = np.random.choice(n_periods, size=5, replace=False)
        for idx in shock_indices:
            shocks[idx] = np.random.uniform(-0.15, -0.05)  # 负面冲击
            
        sp500_returns = trend + cycle + shocks + np.random.normal(0, 0.02, n_periods)
        sp500_data = sp500_base * np.cumprod(1 + sp500_returns)
        
        # 10年期国债收益率（与利率相关但有滞后）
        bond_base = rate_data * 1.2 + 0.5  # 基础关系
        bond_noise = np.random.normal(0, 0.1, n_periods)
        bond_data = bond_base + bond_noise
        
        # VIX（波动率指数）
        vix_base = 20 + 30 * np.abs(sp500_returns) + np.random.normal(0, 5, n_periods)
        vix_data = np.clip(vix_base, 10, 80)
        
        df = pd.DataFrame({
            'Rate': rate_data,
            'SP500': sp500_data,
            'Bond': bond_data,
            'VIX': vix_data
        }, index=dates)
        
        return df

def main():
    """
    主函数
    """
    print("=== 高级事件驱动动态知识图谱模型 ===")
    
    # 1. 生成高级金融数据
    data_generator = AdvancedFinancialDataGenerator()
    df = data_generator.generate_realistic_data()
    
    print(f"数据时间范围: {df.index[0]} 到 {df.index[-1]}")
    print(f"数据点数量: {len(df)}")
    print("\n数据统计:")
    print(df.describe())
    
    # 计算收益率
    returns = df.pct_change().dropna()
    
    # 2. 模型参数
    num_nodes = 4  # Fed, SP500, Bond, VIX
    node_dim = 32
    hidden_dim = 64
    event_dim = 4
    num_event_types = 8
    
    # 3. 初始化模型
    model = AdvancedEventDKG(num_nodes, node_dim, hidden_dim, event_dim, num_event_types)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # 多任务损失
    price_criterion = nn.BCELoss()
    volatility_criterion = nn.MSELoss()
    
    # 4. 准备训练数据
    node_ids = torch.arange(num_nodes)
    
    # 目标：价格方向和波动率
    price_targets = (returns['SP500'].shift(-1) > 0).astype(int).dropna()
    volatility_targets = returns['SP500'].rolling(5).std().shift(-1).dropna()
    
    # 归一化波动率目标
    volatility_targets = (volatility_targets - volatility_targets.min()) / (volatility_targets.max() - volatility_targets.min())
    
    # 5. 训练
    print("\n开始训练高级模型...")
    epochs = 30
    historical_features = []
    
    for epoch in range(epochs):
        model.reset_memory()
        epoch_loss = 0
        epoch_price_acc = 0
        epoch_vol_loss = 0
        
        historical_features = []  # 重置历史特征
        
        for i in range(len(returns) - 1):
            if i >= len(price_targets) or i >= len(volatility_targets):
                break
                
            # 当前特征
            current_features = torch.tensor([
                returns.iloc[i]['Rate'],
                returns.iloc[i]['SP500'], 
                returns.iloc[i]['Bond'],
                returns.iloc[i]['VIX']
            ], dtype=torch.float)
            
            historical_features.append(current_features)
            
            # 事件类型（简化）
            event_types = [FinancialEventTypes.ECONOMIC_INDICATOR]
            if abs(current_features[0]) > 0.01:
                event_types.append(FinancialEventTypes.MONETARY_POLICY)
            if abs(current_features[1]) > 0.03:
                event_types.append(FinancialEventTypes.MARKET_SHOCK)
                
            # 前向传播
            hidden_states, cell_states, edge_index, event_info = model(
                node_ids, current_features, event_types, historical_features[-10:]
            )
            
            # 价格预测
            sp500_state = hidden_states[1]  # SP500节点
            price_pred = model.price_predictor(sp500_state).squeeze()
            
            # 波动率预测
            volatility_pred = model.volatility_predictor(sp500_state).squeeze()
            
            # 目标
            price_target = torch.tensor(price_targets.iloc[i], dtype=torch.float)
            vol_target = torch.tensor(volatility_targets.iloc[i], dtype=torch.float)
            
            # 多任务损失
            price_loss = price_criterion(price_pred, price_target)
            vol_loss = volatility_criterion(volatility_pred, vol_target)
            total_loss = price_loss + 0.5 * vol_loss
            
            # 反向传播
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪
            optimizer.step()
            
            # 更新模型状态
            model.hidden_state = hidden_states.detach()
            model.cell_state = cell_states.detach()
            
            epoch_loss += total_loss.item()
            epoch_price_acc += int((price_pred > 0.5) == (price_target > 0.5))
            epoch_vol_loss += vol_loss.item()
        
        if (epoch + 1) % 10 == 0:
            avg_acc = epoch_price_acc / (len(returns) - 1)
            print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}, Price Acc: {avg_acc:.3f}, Vol Loss: {epoch_vol_loss:.4f}")
    
    print("\n训练完成！")
    
    # 6. 可视化结果
    plt.figure(figsize=(20, 12))
    
    # 原始数据
    plt.subplot(2, 3, 1)
    plt.plot(df.index, df['SP500'], label='S&P 500', linewidth=2)
    plt.title('S&P 500 Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 2)
    plt.plot(df.index, df['Rate'], label='Fed Rate', color='red', linewidth=2)
    plt.plot(df.index, df['Bond'], label='10Y Bond', color='blue', linewidth=2)
    plt.title('Interest Rates')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 3)
    plt.plot(df.index, df['VIX'], label='VIX', color='orange', linewidth=2)
    plt.title('Volatility Index')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 收益率分布
    plt.subplot(2, 3, 4)
    plt.hist(returns['SP500'].dropna(), bins=50, alpha=0.7, color='green')
    plt.title('S&P 500 Returns Distribution')
    plt.xlabel('Returns')
    plt.ylabel('Frequency')
    
    # 相关性热图
    plt.subplot(2, 3, 5)
    correlation_matrix = returns.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Asset Correlation Matrix')
    
    # 波动率时间序列
    plt.subplot(2, 3, 6)
    rolling_vol = returns['SP500'].rolling(20).std()
    plt.plot(returns.index, rolling_vol, label='20-day Rolling Volatility', linewidth=2)
    plt.title('S&P 500 Volatility')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('advanced_event_dkg_analysis.png', dpi=150, bbox_inches='tight')
    print("高级分析图表已保存为: advanced_event_dkg_analysis.png")
    
    plt.show()

if __name__ == "__main__":
    main()
