"""
RAG评估指标计算
包含召回率、准确率、NDCG、MRR等指标
"""
from typing import List, Dict, Any, Optional
import time
import math
from collections import defaultdict
import numpy as np
from pydantic import BaseModel


class RetrievalMetrics(BaseModel):
    """检索评估指标"""
    precision_at_k: Dict[int, float] = {}  # P@K
    recall_at_k: Dict[int, float] = {}     # R@K
    f1_at_k: Dict[int, float] = {}         # F1@K
    ndcg_at_k: Dict[int, float] = {}       # NDCG@K
    mrr: float = 0.0                       # Mean Reciprocal Rank
    map_score: float = 0.0                 # Mean Average Precision


class GenerationMetrics(BaseModel):
    """生成评估指标"""
    keyword_coverage: float = 0.0          # 关键词覆盖率
    sku_accuracy: float = 0.0             # SKU推荐准确率
    answer_relevance: float = 0.0         # 答案相关性(需要人工评估或模型评估)


class LatencyMetrics(BaseModel):
    """延迟评估指标"""
    query_rewrite_time: float = 0.0
    vector_search_time: float = 0.0
    keyword_search_time: float = 0.0
    rerank_time: float = 0.0
    context_build_time: float = 0.0
    realtime_data_time: float = 0.0
    llm_generate_time: float = 0.0
    total_time: float = 0.0


class OverallMetrics(BaseModel):
    """综合评估指标"""
    retrieval: RetrievalMetrics
    generation: GenerationMetrics
    latency: LatencyMetrics
    query_id: str
    query: str


class MetricsCalculator:
    """指标计算器"""

    def __init__(self):
        pass

    def calculate_retrieval_metrics(
        self,
        retrieved_skus: List[str],
        relevant_skus: List[str],
        k_values: List[int] = [1, 3, 5, 10]
    ) -> RetrievalMetrics:
        """
        计算检索指标

        Args:
            retrieved_skus: 检索返回的SKU列表(按相关性排序)
            relevant_skus: 相关SKU列表
            k_values: 计算P@K, R@K的K值列表
        """
        metrics = RetrievalMetrics()

        # 计算各种@K指标
        for k in k_values:
            retrieved_k = retrieved_skus[:k]
            relevant_retrieved = len(set(retrieved_k) & set(relevant_skus))

            # Precision@K
            precision_k = relevant_retrieved / k if k > 0 else 0
            metrics.precision_at_k[k] = precision_k

            # Recall@K
            recall_k = relevant_retrieved / len(relevant_skus) if relevant_skus else 0
            metrics.recall_at_k[k] = recall_k

            # F1@K
            if precision_k + recall_k > 0:
                f1_k = 2 * precision_k * recall_k / (precision_k + recall_k)
            else:
                f1_k = 0
            metrics.f1_at_k[k] = f1_k

            # NDCG@K
            ndcg_k = self._calculate_ndcg_at_k(retrieved_k, relevant_skus, k)
            metrics.ndcg_at_k[k] = ndcg_k

        # MRR (Mean Reciprocal Rank)
        metrics.mrr = self._calculate_mrr(retrieved_skus, relevant_skus)

        # MAP (Mean Average Precision)
        metrics.map_score = self._calculate_map(retrieved_skus, relevant_skus)

        return metrics

    def _calculate_ndcg_at_k(
        self,
        retrieved_skus: List[str],
        relevant_skus: List[str],
        k: int
    ) -> float:
        """计算NDCG@K"""
        # 简化版NDCG：相关为1，不相关为0
        dcg = 0.0
        for i, sku in enumerate(retrieved_skus[:k]):
            if sku in relevant_skus:
                dcg += 1.0 / math.log2(i + 2)  # i+2 because log2(1)=0

        # 理想DCG
        idcg = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(relevant_skus))))

        return dcg / idcg if idcg > 0 else 0.0

    def _calculate_mrr(self, retrieved_skus: List[str], relevant_skus: List[str]) -> float:
        """计算MRR"""
        for i, sku in enumerate(retrieved_skus):
            if sku in relevant_skus:
                return 1.0 / (i + 1)
        return 0.0

    def _calculate_map(self, retrieved_skus: List[str], relevant_skus: List[str]) -> float:
        """计算MAP"""
        if not relevant_skus:
            return 0.0

        precision_sum = 0.0
        relevant_found = 0

        for i, sku in enumerate(retrieved_skus):
            if sku in relevant_skus:
                relevant_found += 1
                precision_sum += relevant_found / (i + 1)

        return precision_sum / len(relevant_skus)

    def calculate_generation_metrics(
        self,
        generated_response: str,
        referenced_skus: List[str],
        expected_keywords: List[str],
        expected_skus: List[str]
    ) -> GenerationMetrics:
        """计算生成指标"""
        metrics = GenerationMetrics()

        # 关键词覆盖率
        response_lower = generated_response.lower()
        covered_keywords = sum(1 for kw in expected_keywords
                             if kw.lower() in response_lower)
        metrics.keyword_coverage = (covered_keywords / len(expected_keywords)
                                   if expected_keywords else 0)

        # SKU推荐准确率
        if referenced_skus:
            correct_skus = len(set(referenced_skus) & set(expected_skus))
            metrics.sku_accuracy = correct_skus / len(referenced_skus)
        else:
            metrics.sku_accuracy = 0.0

        # 答案相关性(这里简化处理，实际可以用模型评估)
        metrics.answer_relevance = (metrics.keyword_coverage + metrics.sku_accuracy) / 2

        return metrics

    def calculate_latency_metrics(self, timing_data: Dict[str, float]) -> LatencyMetrics:
        """计算延迟指标"""
        metrics = LatencyMetrics()

        # 各模块延迟
        metrics.query_rewrite_time = timing_data.get('query_rewrite', 0.0)
        metrics.vector_search_time = timing_data.get('vector_search', 0.0)
        metrics.keyword_search_time = timing_data.get('keyword_search', 0.0)
        metrics.rerank_time = timing_data.get('rerank', 0.0)
        metrics.context_build_time = timing_data.get('context_build', 0.0)
        metrics.realtime_data_time = timing_data.get('realtime_data', 0.0)
        metrics.llm_generate_time = timing_data.get('llm_generate', 0.0)

        # 总延迟
        metrics.total_time = sum(timing_data.values())

        return metrics


class TimingDecorator:
    """时间测量装饰器"""

    def __init__(self):
        self.timings = {}

    def time_function(self, func_name: str):
        """装饰器工厂"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                start_time = time.time()
                result = func(*args, **kwargs)
                end_time = time.time()
                self.timings[func_name] = end_time - start_time
                return result
            return wrapper
        return decorator

    def get_timings(self) -> Dict[str, float]:
        """获取计时结果"""
        return self.timings.copy()

    def reset(self):
        """重置计时"""
        self.timings = {}


def calculate_aggregated_metrics(metrics_list: List[OverallMetrics]) -> Dict[str, Any]:
    """计算聚合指标"""
    if not metrics_list:
        return {}

    # 检索指标聚合
    retrieval_agg = {
        'avg_precision_at_5': np.mean([m.retrieval.precision_at_k.get(5, 0) for m in metrics_list]),
        'avg_recall_at_5': np.mean([m.retrieval.recall_at_k.get(5, 0) for m in metrics_list]),
        'avg_ndcg_at_5': np.mean([m.retrieval.ndcg_at_k.get(5, 0) for m in metrics_list]),
        'avg_mrr': np.mean([m.retrieval.mrr for m in metrics_list]),
        'avg_map': np.mean([m.retrieval.map_score for m in metrics_list])
    }

    # 生成指标聚合
    generation_agg = {
        'avg_keyword_coverage': np.mean([m.generation.keyword_coverage for m in metrics_list]),
        'avg_sku_accuracy': np.mean([m.generation.sku_accuracy for m in metrics_list]),
        'avg_answer_relevance': np.mean([m.generation.answer_relevance for m in metrics_list])
    }

    # 延迟指标聚合
    latency_agg = {
        'avg_total_time': np.mean([m.latency.total_time for m in metrics_list]),
        'p95_total_time': np.percentile([m.latency.total_time for m in metrics_list], 95),
        'p99_total_time': np.percentile([m.latency.total_time for m in metrics_list], 99),
        'avg_query_rewrite_time': np.mean([m.latency.query_rewrite_time for m in metrics_list]),
        'avg_vector_search_time': np.mean([m.latency.vector_search_time for m in metrics_list]),
        'avg_rerank_time': np.mean([m.latency.rerank_time for m in metrics_list]),
        'avg_llm_generate_time': np.mean([m.latency.llm_generate_time for m in metrics_list])
    }

    return {
        'retrieval': retrieval_agg,
        'generation': generation_agg,
        'latency': latency_agg,
        'total_queries': len(metrics_list)
    }


if __name__ == "__main__":
    # 测试指标计算
    calculator = MetricsCalculator()

    # 模拟数据
    retrieved_skus = ["SKU_1001", "SKU_2002", "SKU_7001", "SKU_3003", "SKU_8001"]
    relevant_skus = ["SKU_1001", "SKU_3003", "SKU_5005"]

    # 计算检索指标
    retrieval_metrics = calculator.calculate_retrieval_metrics(
        retrieved_skus, relevant_skus, [1, 3, 5]
    )

    print("=== 检索指标测试 ===")
    print(f"Precision@5: {retrieval_metrics.precision_at_k[5]:.3f}")
    print(f"Recall@5: {retrieval_metrics.recall_at_k[5]:.3f}")
    print(f"NDCG@5: {retrieval_metrics.ndcg_at_k[5]:.3f}")
    print(f"MRR: {retrieval_metrics.mrr:.3f}")
    print(f"MAP: {retrieval_metrics.map_score:.3f}")

    # 计算生成指标
    generation_metrics = calculator.calculate_generation_metrics(
        generated_response="推荐低乳糖配方奶粉，含益生菌，适合肠胃敏感的宝宝",
        referenced_skus=["SKU_1001", "SKU_3003"],
        expected_keywords=["低乳糖", "益生菌", "肠胃"],
        expected_skus=["SKU_1001", "SKU_3003", "SKU_5005"]
    )

    print("\n=== 生成指标测试 ===")
    print(f"关键词覆盖率: {generation_metrics.keyword_coverage:.3f}")
    print(f"SKU准确率: {generation_metrics.sku_accuracy:.3f}")
    print(f"答案相关性: {generation_metrics.answer_relevance:.3f}")