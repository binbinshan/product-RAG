# ProductRAG - 商品智能导购RAG系统

**Product Retrieval-Augmented Generation System**

> 让每一次商品推荐都有据可循

基于 Python + LangChain + LangGraph 实现的完整商品导购 RAG 系统

[![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)](https://github.com/yourname/ProductRAG)
[![Python](https://img.shields.io/badge/python-3.8+-green.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-orange.svg)](LICENSE)
[![LangChain](https://img.shields.io/badge/langchain-1.1.0-brightgreen.svg)](https://github.com/langchain-ai/langchain)
[![LangGraph](https://img.shields.io/badge/langgraph-1.0.4-brightgreen.svg)](https://github.com/langchain-ai/langgraph)

## 系统架构

```
QueryRewrite → HybridSearch → Rerank → ContextBuilder → RealTimeData → LLMGenerate
```

## 项目进度

| 模块 | 状态 | 进度 | 说明             |
|------|------|----|----------------|
| Module 1: Query改写 | ✅ 已完成 | 100% | 分层重写策略已实现      |
| Module 2: 混合检索 | 🚧 开发中 | 0% | ....           |
| Module 3: 重排序 | 🚧 开发中 | 0% | ....  |
| Module 4: 上下文构建 | 🚧 开发中 | 0% | ....  |
| Module 5: 实时数据 | 🚧 开发中 | 0% | .... |
| Module 6: LLM生成 | 🚧 开发中 | 0% | .... |
| Module 7: 流程编排 |  🚧 开发中 | 0% | ....  |
| 数据模型 | ✅ 已完成 | 100% | 全部Pydantic模型定义 |
| 测试覆盖 | 🚧 开发中 | 0% | 各模块单元测试        |

### 最新更新 🆕
- **Module 1**: 新增分层重写策略系统，支持规则扩展、同义词、LLM优化、检索增强四种策略

## 模块说明

### Module 1: Query改写服务 (query_rewrite.py) ✅
- **职责**: 将用户原始Query转换为可检索的结构化查询
- **输入**: 原始查询 + 用户上下文
- **输出**: 改写查询列表 + 结构化过滤条件
- **实现**: 分层重写策略 - 根据查询特征自动选择最佳策略
- **策略类型**:
  - 🔧 规则扩展: 基于领域知识的关键词扩展
  - 📚 同义词替换: 智能同义词变换
  - 🤖 LLM优化: 复杂查询的语义理解
  - 🔍 检索增强: 基于知识库的查询增强

### Module 2: Hybrid Search服务 (hybrid_search.py)
- **职责**: 多路召回 (向量 + 关键词 + 过滤)
- **子模块**:
  - VectorRetrievalService: FAISS向量检索
  - KeywordRetrievalService: BM25关键词检索
  - HybridSearchService: 结果合并去重
- **输出**: 合并后的候选商品列表

### Module 3: Rerank重排序服务 (rerank.py)
- **职责**: 对候选商品精排,选出Top-N
- **排序维度**:
  - Query-商品语义相关度
  - 商品质量和特性
  - 业务权重
- **实现**: 使用LLM或Cross-Encoder精排

### Module 4: 上下文构建服务 (context_builder.py)
- **职责**: 整理商品信息为LLM可消费格式
- **特性**:
  - Token限制控制
  - 必须包含SKU_ID
  - 只包含静态商品知识

### Module 5: 实时数据服务 (realtime_data.py)
- **职责**: 提供价格/库存/促销等实时数据
- **特性**:
  - 模拟API调用
  - 禁止缓存到向量库
  - 数据必须来自实时查询

### Module 6: LLM生成服务 (llm_generate.py)
- **职责**: 基于商品上下文生成导购回答
- **约束**:
  - 只能推荐给定商品
  - 不得编造功效/价格
  - 推荐理由可回溯
  - 实时数据必须准确

### Module 7: 流程编排 (pipeline.py)
- **职责**: LangGraph编排完整RAG流程
- **特性**:
  - 状态管理
  - 节点间数据传递
  - 可视化流程图

## 数据模型 (models.py)

所有模块间接口使用Pydantic严格定义:
- QueryRewriteInput/Output
- VectorRetrievalInput/Result
- KeywordRetrievalInput/Result
- MergedCandidate
- RerankInput/Output
- ContextBuilderInput/Output
- RealTimeDataInput/Output
- LLMGenerateInput/Output
- PipelineState

## 安装依赖

```bash
pip install -r requirements.txt
```

## 快速开始

### 1. 测试单个模块

```python
# 测试Query改写
python query_rewrite.py

# 测试混合检索
python hybrid_search.py

# 测试重排序
python rerank.py

# 测试上下文构建
python context_builder.py

# 测试实时数据
python realtime_data.py

# 测试LLM生成
python llm_generate.py
```

### 2. 运行完整流程

```python
# 使用Mock LLM测试
python pipeline.py

# 或在代码中使用
from pipeline import ProductRAGPipeline

pipeline = ProductRAGPipeline(use_mock=True)
result = pipeline.run(
    query="不上火的奶粉",
    user_context={"channel": "miniapp", "user_type": "new_user"}
)

print(result['final_response'])
```

### 3. 运行所有测试

```bash
python test_all.py
```

## 使用真实LLM

如需使用真实OpenAI API:

```python
import os
os.environ["OPENAI_API_KEY"] = "your-api-key"

pipeline = ProductRAGPipeline(
    llm_model="gpt-3.5-turbo",
    use_mock=False  # 使用真实LLM
)
```

## 系统特性

### 硬性工程约束
✅ 商品信息必须绑定SKU

✅ 向量库只存静态商品知识

✅ 价格/库存/促销只能实时查询

✅ 检索、排序、生成职责严格分离

✅ 每个模块可独立测试

### 接口规范
✅ 所有输入输出使用Pydantic DTO

✅ 严格JSON格式,可反序列化

✅ 无占位符,全部真实实现

✅ 支持LangGraph节点调用


### 检索策略
- 向量检索: FAISS + sentence-transformers
- 关键词检索: BM25
- 混合召回: 多路结果合并去重
- 重排序: LLM精排

### 生成约束
- 基于提供的商品上下文
- 不编造功效、价格、库存
- 推荐理由可回溯到字段
- 明确标注缺货商品

## 项目结构

```
.
├── models.py              # 数据模型定义
├── query_rewrite.py       # Module 1: Query改写
├── hybrid_search.py       # Module 2: 混合检索
├── rerank.py             # Module 3: 重排序
├── context_builder.py    # Module 4: 上下文构建
├── realtime_data.py      # Module 5: 实时数据
├── llm_generate.py       # Module 6: LLM生成
├── pipeline.py           # Module 7: 流程编排
├── test_all.py           # 完整测试脚本
├── requirements.txt      # 依赖包
└── README.md            # 本文档
```

## 扩展建议

1. **向量模型**: 可替换为中文优化模型 (如 text2vec-chinese)
2. **重排序**: 可使用Cross-Encoder模型 (如 bge-reranker)
3. **商品数据库**: 接入真实MySQL/PostgreSQL
4. **实时数据**: 接入真实API或Redis缓存
5. **LLM**: 支持更多模型 (Claude, OpenAI等)
6. **监控**: 添加日志、指标、追踪

## 许可证

MIT License