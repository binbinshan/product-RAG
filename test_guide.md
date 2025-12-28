# RAG系统测试指南

## 快速开始

### 1. 基础测试
```bash
# 运行完整评估
python rag_evaluator.py

# 测试单个模块
python test_dataset.py
python evaluation_metrics.py
```

### 2. 评估指标说明

#### 召回率指标
- **Precision@K**: 前K个结果中相关商品的比例
- **Recall@K**: 相关商品被召回的比例
- **F1@K**: Precision和Recall的调和平均
- **NDCG@K**: 考虑排序质量的归一化折损累积增益
- **MRR**: 第一个相关结果的倒数排名
- **MAP**: 平均精度均值

#### 准确率指标
- **关键词覆盖率**: 生成回答包含期望关键词的比例
- **SKU准确率**: 推荐SKU与标准答案匹配的比例
- **答案相关性**: 综合评估生成回答的质量

#### 耗时指标
- **各模块耗时**: Query改写、检索、重排序、生成等
- **端到端延迟**: 完整流程总耗时
- **P95/P99延迟**: 高百分位延迟，反映系统稳定性

### 3. 测试数据集结构

```python
# 测试查询示例
TestQuery(
    query_id="Q001",
    query="不上火的奶粉",
    user_context={"channel": "miniapp", "age_range": "3-6"},
    expected_skus=["SKU_1001", "SKU_3003"],      # 期望召回的SKU
    relevant_skus=["SKU_1001", "SKU_3003", ...], # 所有相关SKU
    expected_answer_keywords=["低乳糖", "益生菌"], # 期望关键词
    difficulty="easy"  # 难度等级
)
```

### 4. 运行不同类型的测试

#### A. 全面评估
```python
from test_dataset import TestDataset
from pipeline import ProductRAGPipeline
from rag_evaluator import RAGEvaluator

# 创建组件
dataset = TestDataset()
pipeline = ProductRAGPipeline(use_mock=True)
evaluator = RAGEvaluator(pipeline, dataset)

# 运行评估
results = evaluator.evaluate_all_queries()
```

#### B. 按难度分组测试
```python
# 分别测试easy/medium/hard难度
difficulty_results = evaluator.evaluate_by_difficulty()

# 查看各难度表现
for difficulty, result in difficulty_results.items():
    agg = result['aggregated_metrics']
    print(f"{difficulty}: P@5={agg['retrieval']['avg_precision_at_5']:.3f}")
```

#### C. 性能压测
```python
# 延迟基准测试
latency_stats = evaluator.benchmark_latency(num_runs=100)
print(f"平均延迟: {latency_stats['avg_latency']:.3f}s")
print(f"P95延迟: {latency_stats['p95_latency']:.3f}s")
print(f"QPS: {latency_stats['qps']:.1f}")
```

### 5. 自定义测试数据

#### 添加新的测试查询
```python
# 编辑 test_dataset.py，在 _create_test_queries 方法中添加:
TestQuery(
    query_id="Q_NEW",
    query="你的测试查询",
    expected_skus=["SKU_XXX"],
    relevant_skus=["SKU_XXX", "SKU_YYY"],
    expected_answer_keywords=["关键词1", "关键词2"],
    difficulty="medium"
)
```

#### 修改评估标准
```python
# 编辑 evaluation_metrics.py 中的计算逻辑
def calculate_custom_metric(self, ...):
    # 实现自定义评估指标
    pass
```

### 6. 结果分析

#### 输出文件
- `detailed_results_TIMESTAMP.csv` - 每个查询的详细指标
- `aggregated_results_TIMESTAMP.json` - 聚合统计指标
- `evaluation_report.md` - 可读性报告

#### 关键指标阈值
```
优秀    良好    待改进
P@5     >0.8    >0.6    <0.6
R@5     >0.7    >0.5    <0.5
NDCG@5  >0.8    >0.6    <0.6
延迟    <1s     <2s     >2s
```

### 7. 优化建议

#### 召回率低 → 改进检索
- 扩大检索范围 (增大top_k)
- 优化Query改写策略
- 使用更好的向量模型
- 调整混合检索权重

#### 准确率低 → 改进排序
- 优化重排序算法
- 改进LLM prompt
- 增强上下文构建
- 添加业务规则过滤

#### 延迟高 → 性能优化
- 向量检索加速 (量化、索引优化)
- LLM推理加速 (小模型、缓存)
- 并行处理
- 异步调用

### 8. 持续监控

建议将评估框架集成到CI/CD流程:

```bash
# 在代码提交前运行快速测试
python rag_evaluator.py --quick --queries 10

# 定期运行全面评估
python rag_evaluator.py --full --output ./daily_reports/

# 性能回归测试
python rag_evaluator.py --benchmark --baseline ./baseline_results.json
```

### 9. 问题排查

#### 常见问题
1. **召回为空** - 检查索引是否构建，向量服务是否正常
2. **延迟过高** - 分模块测量，定位瓶颈组件
3. **结果不准** - 检查测试数据标注，确认期望结果合理性

#### 调试技巧
- 开启详细日志查看中间结果
- 单步调试各模块输入输出
- 对比不同配置下的指标变化

这个测试框架为您提供了完整的RAG系统评估能力，帮助持续优化系统性能。