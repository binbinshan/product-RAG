"""
RAG系统完整评估框架
支持召回率、准确率、耗时等全面评估
"""
import time
import json
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
import pandas as pd

from test_dataset import TestDataset, TestQuery
from evaluation_metrics import (
    MetricsCalculator, OverallMetrics, TimingDecorator,
    calculate_aggregated_metrics
)
from pipeline import ProductRAGPipeline
from models import PipelineState


class RAGEvaluator:
    """RAG系统评估器"""

    def __init__(
        self,
        pipeline: ProductRAGPipeline,
        test_dataset: TestDataset,
        output_dir: str = "./evaluation_results"
    ):
        """
        初始化评估器

        Args:
            pipeline: RAG流程实例
            test_dataset: 测试数据集
            output_dir: 结果输出目录
        """
        self.pipeline = pipeline
        self.test_dataset = test_dataset
        self.output_dir = output_dir
        self.metrics_calculator = MetricsCalculator()
        self.timing_decorator = TimingDecorator()

        # 创建输出目录
        import os
        os.makedirs(output_dir, exist_ok=True)

    def evaluate_single_query(self, test_query: TestQuery) -> OverallMetrics:
        """
        评估单个查询

        Args:
            test_query: 测试查询

        Returns:
            完整评估指标
        """
        print(f"\n评估查询: {test_query.query_id} - {test_query.query}")

        # 重置计时器
        self.timing_decorator.reset()

        # 为pipeline的各个服务添加计时
        self._add_timing_to_services()

        # 执行RAG流程
        start_time = time.time()
        try:
            result = self.pipeline.run(
                query=test_query.query,
                user_context=test_query.user_context
            )
            total_time = time.time() - start_time
        except Exception as e:
            print(f"查询执行失败: {e}")
            # 返回空指标
            return self._create_empty_metrics(test_query)

        # 获取计时数据
        timing_data = self.timing_decorator.get_timings()
        timing_data['total'] = total_time

        # 提取检索结果
        retrieved_skus = self._extract_retrieved_skus(result)

        # 计算各类指标
        retrieval_metrics = self.metrics_calculator.calculate_retrieval_metrics(
            retrieved_skus=retrieved_skus,
            relevant_skus=test_query.relevant_skus,
            k_values=[1, 3, 5, 10]
        )

        generation_metrics = self.metrics_calculator.calculate_generation_metrics(
            generated_response=result.get('final_response', ''),
            referenced_skus=result.get('referenced_skus', []),
            expected_keywords=test_query.expected_answer_keywords,
            expected_skus=test_query.expected_skus
        )

        latency_metrics = self.metrics_calculator.calculate_latency_metrics(timing_data)

        # 组装完整指标
        overall_metrics = OverallMetrics(
            retrieval=retrieval_metrics,
            generation=generation_metrics,
            latency=latency_metrics,
            query_id=test_query.query_id,
            query=test_query.query
        )

        print(f"  - P@5: {retrieval_metrics.precision_at_k.get(5, 0):.3f}")
        print(f"  - R@5: {retrieval_metrics.recall_at_k.get(5, 0):.3f}")
        print(f"  - 关键词覆盖: {generation_metrics.keyword_coverage:.3f}")
        print(f"  - 总耗时: {latency_metrics.total_time:.3f}s")

        return overall_metrics

    def evaluate_all_queries(self) -> Dict[str, Any]:
        """评估所有查询"""
        print("开始全面评估...")

        all_metrics = []
        failed_queries = []

        for test_query in self.test_dataset.queries:
            try:
                metrics = self.evaluate_single_query(test_query)
                all_metrics.append(metrics)
            except Exception as e:
                print(f"查询 {test_query.query_id} 评估失败: {e}")
                failed_queries.append(test_query.query_id)

        # 计算聚合指标
        aggregated_metrics = calculate_aggregated_metrics(all_metrics)

        # 添加失败信息
        aggregated_metrics['failed_queries'] = failed_queries
        aggregated_metrics['success_rate'] = (
            len(all_metrics) / len(self.test_dataset.queries)
        )

        # 保存结果
        self._save_results(all_metrics, aggregated_metrics)

        return {
            'individual_metrics': all_metrics,
            'aggregated_metrics': aggregated_metrics
        }

    def evaluate_by_difficulty(self) -> Dict[str, Dict[str, Any]]:
        """按难度分组评估"""
        results = {}

        for difficulty in ['easy', 'medium', 'hard']:
            print(f"\n=== 评估{difficulty.upper()}难度查询 ===")

            queries = self.test_dataset.get_queries_by_difficulty(difficulty)
            if not queries:
                continue

            difficulty_metrics = []
            for query in queries:
                metrics = self.evaluate_single_query(query)
                difficulty_metrics.append(metrics)

            aggregated = calculate_aggregated_metrics(difficulty_metrics)
            results[difficulty] = {
                'individual_metrics': difficulty_metrics,
                'aggregated_metrics': aggregated
            }

            print(f"{difficulty}难度汇总:")
            print(f"  - 平均P@5: {aggregated['retrieval']['avg_precision_at_5']:.3f}")
            print(f"  - 平均R@5: {aggregated['retrieval']['avg_recall_at_5']:.3f}")
            print(f"  - 平均耗时: {aggregated['latency']['avg_total_time']:.3f}s")

        return results

    def benchmark_latency(self, num_runs: int = 100) -> Dict[str, Any]:
        """压力测试 - 评估并发性能"""
        print(f"\n=== 延迟基准测试 ({num_runs}次) ===")

        # 选择一个中等难度的查询进行压测
        test_query = None
        for query in self.test_dataset.queries:
            if query.difficulty == 'medium':
                test_query = query
                break

        if not test_query:
            test_query = self.test_dataset.queries[0]

        latencies = []
        errors = 0

        for i in range(num_runs):
            if i % 10 == 0:
                print(f"  进度: {i+1}/{num_runs}")

            try:
                start_time = time.time()
                self.pipeline.run(
                    query=test_query.query,
                    user_context=test_query.user_context
                )
                latency = time.time() - start_time
                latencies.append(latency)
            except Exception as e:
                errors += 1

        # 计算统计数据
        import numpy as np
        stats = {
            'total_runs': num_runs,
            'successful_runs': len(latencies),
            'error_rate': errors / num_runs,
            'avg_latency': np.mean(latencies) if latencies else 0,
            'median_latency': np.median(latencies) if latencies else 0,
            'p95_latency': np.percentile(latencies, 95) if latencies else 0,
            'p99_latency': np.percentile(latencies, 99) if latencies else 0,
            'min_latency': np.min(latencies) if latencies else 0,
            'max_latency': np.max(latencies) if latencies else 0,
            'qps': len(latencies) / sum(latencies) if latencies else 0
        }

        print("基准测试结果:")
        print(f"  - 成功率: {(1-stats['error_rate'])*100:.1f}%")
        print(f"  - 平均延迟: {stats['avg_latency']:.3f}s")
        print(f"  - P95延迟: {stats['p95_latency']:.3f}s")
        print(f"  - P99延迟: {stats['p99_latency']:.3f}s")
        print(f"  - QPS: {stats['qps']:.1f}")

        return stats

    def _add_timing_to_services(self):
        """为pipeline服务添加计时装饰器"""
        # 这里简化处理，实际应该通过AOP或者中间件实现
        # 由于现有服务没有暴露细分方法，这里只能测量总时间
        pass

    def _extract_retrieved_skus(self, result: Dict[str, Any]) -> List[str]:
        """从结果中提取检索到的SKU列表"""
        # 优先使用重排序后的结果
        if result.get('ranked_skus'):
            return result['ranked_skus']

        # 否则从候选商品中提取
        if result.get('candidates'):
            return [c.sku_id for c in result['candidates']]

        # 最后使用引用的SKU
        if result.get('referenced_skus'):
            return result['referenced_skus']

        return []

    def _create_empty_metrics(self, test_query: TestQuery) -> OverallMetrics:
        """创建空指标(用于失败场景)"""
        from evaluation_metrics import RetrievalMetrics, GenerationMetrics, LatencyMetrics

        return OverallMetrics(
            retrieval=RetrievalMetrics(),
            generation=GenerationMetrics(),
            latency=LatencyMetrics(),
            query_id=test_query.query_id,
            query=test_query.query
        )

    def _save_results(
        self,
        individual_metrics: List[OverallMetrics],
        aggregated_metrics: Dict[str, Any]
    ):
        """保存评估结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 保存详细结果
        detailed_results = []
        for metrics in individual_metrics:
            detailed_results.append({
                'query_id': metrics.query_id,
                'query': metrics.query,
                'precision_at_5': metrics.retrieval.precision_at_k.get(5, 0),
                'recall_at_5': metrics.retrieval.recall_at_k.get(5, 0),
                'ndcg_at_5': metrics.retrieval.ndcg_at_k.get(5, 0),
                'mrr': metrics.retrieval.mrr,
                'keyword_coverage': metrics.generation.keyword_coverage,
                'sku_accuracy': metrics.generation.sku_accuracy,
                'total_time': metrics.latency.total_time,
                'query_rewrite_time': metrics.latency.query_rewrite_time,
                'llm_generate_time': metrics.latency.llm_generate_time
            })

        # 保存为CSV
        df = pd.DataFrame(detailed_results)
        csv_path = f"{self.output_dir}/detailed_results_{timestamp}.csv"
        df.to_csv(csv_path, index=False, encoding='utf-8')

        # 保存聚合结果为JSON
        json_path = f"{self.output_dir}/aggregated_results_{timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(aggregated_metrics, f, ensure_ascii=False, indent=2)

        print(f"\n结果已保存:")
        print(f"  - 详细结果: {csv_path}")
        print(f"  - 聚合结果: {json_path}")

    def generate_report(self, results: Dict[str, Any]) -> str:
        """生成评估报告"""
        agg = results['aggregated_metrics']

        report = f"""
# RAG系统评估报告

## 总体概况
- 总查询数: {agg['total_queries']}
- 成功率: {agg.get('success_rate', 1.0) * 100:.1f}%

## 检索性能
- 平均Precision@5: {agg['retrieval']['avg_precision_at_5']:.3f}
- 平均Recall@5: {agg['retrieval']['avg_recall_at_5']:.3f}
- 平均NDCG@5: {agg['retrieval']['avg_ndcg_at_5']:.3f}
- 平均MRR: {agg['retrieval']['avg_mrr']:.3f}
- 平均MAP: {agg['retrieval']['avg_map']:.3f}

## 生成质量
- 平均关键词覆盖率: {agg['generation']['avg_keyword_coverage']:.3f}
- 平均SKU准确率: {agg['generation']['avg_sku_accuracy']:.3f}
- 平均答案相关性: {agg['generation']['avg_answer_relevance']:.3f}

## 延迟性能
- 平均总耗时: {agg['latency']['avg_total_time']:.3f}s
- P95耗时: {agg['latency']['p95_total_time']:.3f}s
- P99耗时: {agg['latency']['p99_total_time']:.3f}s

### 模块耗时分解
- Query改写: {agg['latency']['avg_query_rewrite_time']:.3f}s
- 向量检索: {agg['latency']['avg_vector_search_time']:.3f}s
- 重排序: {agg['latency']['avg_rerank_time']:.3f}s
- LLM生成: {agg['latency']['avg_llm_generate_time']:.3f}s

## 优化建议
{self._generate_optimization_suggestions(agg)}
        """

        return report.strip()

    def _generate_optimization_suggestions(self, agg: Dict[str, Any]) -> str:
        """生成优化建议"""
        suggestions = []

        # 检索优化建议
        if agg['retrieval']['avg_precision_at_5'] < 0.6:
            suggestions.append("- 检索精度较低，建议优化向量模型或重排序算法")

        if agg['retrieval']['avg_recall_at_5'] < 0.5:
            suggestions.append("- 召回率偏低，建议扩大检索范围或改进查询改写策略")

        # 生成优化建议
        if agg['generation']['avg_keyword_coverage'] < 0.7:
            suggestions.append("- 关键词覆盖率不足，建议优化LLM prompt或增强上下文构建")

        # 性能优化建议
        if agg['latency']['avg_total_time'] > 2.0:
            suggestions.append("- 总耗时较长，建议优化慢速模块或并行处理")

        if agg['latency']['avg_llm_generate_time'] > 1.0:
            suggestions.append("- LLM生成耗时过长，建议使用更快的模型或优化prompt长度")

        return '\n'.join(suggestions) if suggestions else "- 系统性能良好，无明显优化点"


if __name__ == "__main__":
    # 创建测试数据集
    dataset = TestDataset()

    # 创建RAG流程
    pipeline = ProductRAGPipeline(use_mock=True)

    # 创建评估器
    evaluator = RAGEvaluator(pipeline, dataset)

    print("=== RAG系统评估 ===")

    # 1. 全面评估
    print("\n1. 全面评估...")
    results = evaluator.evaluate_all_queries()

    # 2. 按难度评估
    print("\n2. 按难度评估...")
    difficulty_results = evaluator.evaluate_by_difficulty()

    # 3. 延迟基准测试
    print("\n3. 延迟基准测试...")
    latency_stats = evaluator.benchmark_latency(num_runs=20)

    # 4. 生成报告
    print("\n4. 生成评估报告...")
    report = evaluator.generate_report(results)

    # 保存报告
    report_path = f"{evaluator.output_dir}/evaluation_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\n评估完成! 报告保存至: {report_path}")
    print(report)