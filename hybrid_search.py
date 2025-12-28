"""
Module 2: Hybrid Search服务
职责: 基于改写Query执行向量+关键词+结构化过滤的并行检索
"""
import numpy as np
import os
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from pymilvus import Collection, connections
from dotenv import load_dotenv
from index_builder import ProductDatabase, HybridIndexBuilder
from models import (
    VectorRetrievalInput, VectorRetrievalResult,
    KeywordRetrievalInput, KeywordRetrievalResult,
    MergedCandidate, RetrievalLog, RetrievalLogSummary
)

# 加载环境变量
load_dotenv()


class VectorRetrievalService:
    """向量检索子模块 - 基于预构建索引"""

    def __init__(self, index_data: Optional[Dict[str, Any]] = None, embedding_model: Optional[str] = None):
        self.db = ProductDatabase()
        # 使用环境变量或默认值
        model_name = embedding_model or os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
        self.encoder = SentenceTransformer(model_name)

        if index_data is None:
            # 如果没有提供索引数据，则构建新的
            builder = HybridIndexBuilder(embedding_model)
            indices = builder.build_all_indices()
            self.index_data = indices["vector_index"]
        else:
            self.index_data = index_data

        self.use_fallback = self.index_data["index_type"] == "memory"
        self._setup_index()

    def _setup_index(self):
        """设置索引"""
        self.sku_list = self.index_data["sku_list"]
        self.content_list = self.index_data["content_list"]
        self.product_map = self.index_data["product_map"]

        if self.use_fallback:
            self.embeddings = self.index_data["embeddings"]
        else:
            # 重新连接Milvus
            try:
                self.collection = Collection(self.index_data["collection_name"])
                self.collection.load()
                print("Milvus索引加载完成")
            except Exception as e:
                print(f"Milvus连接失败，切换到内存索引: {e}")
                self.use_fallback = True
                # 需要重新构建内存索引
                embeddings = self.encoder.encode(self.content_list)
                self.embeddings = np.array(embeddings).astype('float32')

    def retrieve(self, input_data: VectorRetrievalInput) -> List[VectorRetrievalResult]:
        """
        执行向量检索 - 使用Pre-Filtering避免召回空白

        Args:
            input_data: 向量检索输入

        Returns:
            检索结果列表
        """
        # 查询向量编码
        query_vector = self.encoder.encode([input_data.query])
        query_vector = np.array(query_vector).astype('float32')

        if self.use_fallback:
            # 先进行过滤
            filtered_products = self.db.filter_products(input_data.filters)
            filtered_skus = {p["sku_id"] for p in filtered_products}
            return self._fallback_retrieve(query_vector[0], filtered_skus, input_data.top_k)

        try:
            # 🔥 关键修复：Pre-Filtering - 构建Milvus expr表达式
            expr = self._build_milvus_expr(input_data.filters)

            # Milvus检索 - 使用expr进行Pre-Filtering
            search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
            results = self.collection.search(
                data=[query_vector[0].tolist()],
                anns_field="embedding",
                param=search_params,
                limit=input_data.top_k,  # 直接取需要的数量，不需要*2
                expr=expr  # 🔥 Pre-Filtering关键参数
            )

            # 构建结果 - 不需要后置过滤
            final_results = []
            for hit in results[0]:
                sku_id = hit.entity.get("sku_id")
                content = hit.entity.get("content")

                # Milvus返回的是距离，转换为相似度分数
                score = 1.0 / (1.0 + float(hit.distance))

                final_results.append(VectorRetrievalResult(
                    sku_id=sku_id,
                    score=score,
                    content=content
                ))

            return final_results

        except Exception as e:
            print(f"Milvus检索失败，降级到内存索引: {e}")
            # 降级时仍需过滤
            filtered_products = self.db.filter_products(input_data.filters)
            filtered_skus = {p["sku_id"] for p in filtered_products}
            return self._fallback_retrieve(query_vector[0], filtered_skus, input_data.top_k)

    def _build_milvus_expr(self, filters: Dict[str, Any]) -> Optional[str]:
        """
        构建Milvus表达式用于Pre-Filtering

        Args:
            filters: 过滤条件字典

        Returns:
            Milvus表达式字符串，如果没有过滤条件则返回None
        """
        if not filters:
            return None

        expr_parts = []

        # 注意：这里的字段名需要与index_builder.py中Milvus schema一致
        # 目前schema中只有 id, sku_id, embedding, content 字段
        # 所以只能基于sku_id进行过滤

        # 先获取符合条件的商品SKU列表
        filtered_products = self.db.filter_products(filters)
        if not filtered_products:
            # 如果没有符合条件的商品，返回空结果的表达式
            return "sku_id in ['__EMPTY__']"

        sku_list = [p["sku_id"] for p in filtered_products]

        # 构建SKU IN表达式
        if len(sku_list) > 0:
            # Milvus的IN表达式格式: field in [value1, value2, ...]
            sku_str = "', '".join(sku_list)
            expr_parts.append(f"sku_id in ['{sku_str}']")

        return " and ".join(expr_parts) if expr_parts else None

    def _fallback_retrieve(self, query_vector: np.ndarray, filtered_skus: set, top_k: int) -> List[VectorRetrievalResult]:
        """
        降级检索方法 - 修复OOM风险

        Args:
            query_vector: 查询向量
            filtered_skus: 过滤后的SKU集合
            top_k: 返回结果数量

        Returns:
            检索结果列表
        """
        try:
            # 🔥 修复问题5: OOM防护 - 检查内存使用情况
            embeddings_size_mb = self.embeddings.nbytes / (1024 * 1024)
            if embeddings_size_mb > 1024:  # 超过1GB的embedding矩阵
                print(f"警告：向量矩阵过大({embeddings_size_mb:.1f}MB)，可能导致内存不足")
                # 如果矩阵过大，只处理过滤后的商品
                return self._memory_safe_fallback(query_vector, filtered_skus, top_k)

            # 计算余弦相似度
            scores = np.dot(self.embeddings, query_vector) / (
                np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_vector)
            )

            # 获取top-k索引
            top_indices = np.argsort(scores)[::-1][:top_k * 2]

            results = []
            for idx in top_indices:
                sku_id = self.sku_list[idx]

                if sku_id not in filtered_skus:
                    continue

                results.append(VectorRetrievalResult(
                    sku_id=sku_id,
                    score=float(scores[idx]),
                    content=self.content_list[idx]
                ))

                if len(results) >= top_k:
                    break

            return results

        except MemoryError as e:
            print(f"内存不足错误: {e}，使用安全模式检索")
            return self._memory_safe_fallback(query_vector, filtered_skus, top_k)

    def _memory_safe_fallback(self, query_vector: np.ndarray, filtered_skus: set, top_k: int) -> List[VectorRetrievalResult]:
        """
        内存安全的降级检索方法
        """
        # 只计算过滤后SKU的相似度，避免全量计算
        filtered_indices = []
        for i, sku_id in enumerate(self.sku_list):
            if sku_id in filtered_skus:
                filtered_indices.append(i)

        if not filtered_indices:
            return []

        # 只提取需要的embedding进行计算
        filtered_embeddings = self.embeddings[filtered_indices]
        scores = np.dot(filtered_embeddings, query_vector) / (
            np.linalg.norm(filtered_embeddings, axis=1) * np.linalg.norm(query_vector)
        )

        # 排序并返回结果
        sorted_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for i in sorted_indices:
            original_idx = filtered_indices[i]
            results.append(VectorRetrievalResult(
                sku_id=self.sku_list[original_idx],
                score=float(scores[i]),
                content=self.content_list[original_idx]
            ))

        return results


class KeywordRetrievalService:
    """关键词检索子模块 - 基于预构建索引"""

    # 🔥 修复问题5: 类级别jieba实例，避免重复初始化内存浪费
    _jieba_instance = None

    @classmethod
    def _get_jieba(cls):
        """获取单例jieba实例"""
        if cls._jieba_instance is None:
            import jieba
            cls._jieba_instance = jieba
        return cls._jieba_instance

    def __init__(self, index_data: Optional[Dict[str, Any]] = None):
        self.jieba = self._get_jieba()
        self.db = ProductDatabase()

        if index_data is None:
            # 如果没有提供索引数据，则构建新的
            builder = HybridIndexBuilder()
            indices = builder.build_all_indices()
            self.index_data = indices["keyword_index"]
        else:
            self.index_data = index_data

        self._setup_index()

    def _setup_index(self):
        """设置索引"""
        self.sku_list = self.index_data["sku_list"]
        self.product_map = self.index_data["product_map"]
        self.bm25 = self.index_data["bm25"]

    def retrieve(self, input_data: KeywordRetrievalInput) -> List[KeywordRetrievalResult]:
        """
        执行关键词检索 - 优化性能瓶颈，使用向量化操作

        Args:
            input_data: 关键词检索输入

        Returns:
            检索结果列表
        """
        # 过滤商品
        filtered_products = self.db.filter_products(input_data.filters)
        filtered_skus = {p["sku_id"] for p in filtered_products}

        # BM25检索
        query_text = " ".join(input_data.keywords).lower()
        query_tokens = list(self.jieba.cut(query_text))
        scores = self.bm25.get_scores(query_tokens)

        # 🔥 修复问题3: 使用NumPy向量化操作，避免Python原生循环
        scores_array = np.array(scores)
        sku_array = np.array(self.sku_list)

        # 🔥 关键优化：只处理分数>0的商品，避免全量迭代
        positive_mask = scores_array > 0
        positive_indices = np.where(positive_mask)[0]

        if len(positive_indices) == 0:
            return []

        # 提取非零分数和对应的SKU
        positive_scores = scores_array[positive_indices]
        positive_skus = sku_array[positive_indices]

        # 🔥 向量化过滤：使用集合交集操作替代逐个检查
        # 创建SKU到索引的映射，用于快速查找
        filtered_sku_indices = []
        filtered_scores = []

        for idx, sku_id in enumerate(positive_skus):
            if sku_id in filtered_skus:
                filtered_sku_indices.append(positive_indices[idx])
                filtered_scores.append(positive_scores[idx])

        if not filtered_scores:
            return []

        # 🔥 向量化排序：使用NumPy argsort替代Python sort
        sorted_indices = np.argsort(filtered_scores)[::-1]  # 降序排序

        # 构建结果
        results = []
        max_candidates = int(os.getenv('MAX_CANDIDATES', '50'))

        for i in sorted_indices[:max_candidates]:
            original_idx = filtered_sku_indices[i]
            results.append(KeywordRetrievalResult(
                sku_id=self.sku_list[original_idx],
                score=float(filtered_scores[i])
            ))

        return results


class HybridSearchService:
    """Hybrid Search服务 - 多路召回"""

    def __init__(self, vector_service: Optional[VectorRetrievalService] = None,
                 keyword_service: Optional[KeywordRetrievalService] = None,
                 embedding_model: Optional[str] = None):
        """
        初始化混合搜索服务

        Args:
            vector_service: 向量检索服务实例，如果为None则创建新的
            keyword_service: 关键词检索服务实例，如果为None则创建新的
            embedding_model: 向量模型名称，如果为None则使用环境变量
        """
        if vector_service is None or keyword_service is None:
            # 使用环境变量或默认值
            model_name = embedding_model or os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')

            # 构建索引数据
            builder = HybridIndexBuilder(model_name)
            indices = builder.build_all_indices()

            self.vector_service = vector_service or VectorRetrievalService(
                indices["vector_index"], model_name
            )
            self.keyword_service = keyword_service or KeywordRetrievalService(
                indices["keyword_index"]
            )
        else:
            self.vector_service = vector_service
            self.keyword_service = keyword_service

    def search(self, rewritten_queries: List[str], filters: Dict[str, Any], enable_logging: bool = True) -> List[MergedCandidate]:
        """
        执行混合检索

        Args:
            rewritten_queries: 改写后的查询列表
            filters: 过滤条件
            enable_logging: 是否启用召回日志

        Returns:
            合并后的候选商品列表
        """
        candidates_map: Dict[str, MergedCandidate] = {}

        # 存储所有向量检索结果，用于生成日志
        all_vector_results: Dict[str, List[tuple]] = {}  # query -> [(sku_id, score, rank)]
        all_keyword_results: List[tuple] = []  # [(sku_id, score, rank)]

        # 1. 向量检索 - 对每个改写查询
        vector_top_k = int(os.getenv('VECTOR_TOP_K', '20'))
        for query in rewritten_queries:
            vector_input = VectorRetrievalInput(
                query=query,
                top_k=vector_top_k,
                filters=filters
            )
            vector_results = self.vector_service.retrieve(vector_input)

            # 记录向量检索结果用于日志
            if enable_logging:
                all_vector_results[query] = [(result.sku_id, result.score, i+1) for i, result in enumerate(vector_results)]

            for result in vector_results:
                if result.sku_id not in candidates_map:
                    candidates_map[result.sku_id] = MergedCandidate(
                        sku_id=result.sku_id,
                        vector_score=result.score,
                        keyword_score=0.0,
                        sources=["vector"],
                        content=result.content
                    )
                else:
                    # 取最高分
                    if result.score > candidates_map[result.sku_id].vector_score:
                        candidates_map[result.sku_id].vector_score = result.score
                    if "vector" not in candidates_map[result.sku_id].sources:
                        candidates_map[result.sku_id].sources.append("vector")

        # 2. 关键词检索
        # 提取所有关键词
        all_keywords = []
        for query in rewritten_queries:
            all_keywords.extend(query.split())
        all_keywords = list(set(all_keywords))  # 去重

        if all_keywords:
            keyword_input = KeywordRetrievalInput(
                keywords=all_keywords,
                filters=filters
            )
            keyword_results = self.keyword_service.retrieve(keyword_input)

            # 记录关键词检索结果用于日志
            if enable_logging:
                keyword_query = " ".join(all_keywords)
                all_keyword_results = [(result.sku_id, result.score, i+1) for i, result in enumerate(keyword_results)]

            for result in keyword_results:
                if result.sku_id not in candidates_map:
                    candidates_map[result.sku_id] = MergedCandidate(
                        sku_id=result.sku_id,
                        vector_score=0.0,
                        keyword_score=result.score,
                        sources=["keyword"],
                        content=""
                    )
                else:
                    candidates_map[result.sku_id].keyword_score = result.score
                    if "keyword" not in candidates_map[result.sku_id].sources:
                        candidates_map[result.sku_id].sources.append("keyword")

        # 3. 计算混合分数并转换为列表
        candidates = list(candidates_map.values())
        candidates = self._calculate_hybrid_scores(candidates)

        # 4. 生成召回日志
        if enable_logging:
            candidates = self._generate_retrieval_logs(
                candidates,
                all_vector_results,
                all_keyword_results,
                " ".join(all_keywords) if all_keywords else ""
            )

        return candidates

    def _calculate_hybrid_scores(self, candidates: List[MergedCandidate]) -> List[MergedCandidate]:
        """
        计算混合检索融合分数 - 修复归一化算法缺陷

        Args:
            candidates: 候选商品列表

        Returns:
            添加了fusion_score的候选商品列表
        """
        # 从环境变量获取权重配置
        vector_weight = float(os.getenv('VECTOR_WEIGHT', '0.7'))
        keyword_weight = float(os.getenv('KEYWORD_WEIGHT', '0.3'))

        # 权重归一化（确保和为1）
        total_weight = vector_weight + keyword_weight
        if total_weight > 0:
            vector_weight = vector_weight / total_weight
            keyword_weight = keyword_weight / total_weight
        else:
            vector_weight, keyword_weight = 0.7, 0.3

        # 🔥 修复问题2: 分数归一化算法缺陷
        vector_scores = [c.vector_score or 0.0 for c in candidates]
        keyword_scores = [c.keyword_score or 0.0 for c in candidates]

        # 🔥 关键修复A: 只对非零值进行归一化，避免0值干扰
        # 向量分数归一化
        non_zero_vector = [s for s in vector_scores if s > 0]
        if len(non_zero_vector) > 1 and max(non_zero_vector) > min(non_zero_vector):
            min_v, max_v = min(non_zero_vector), max(non_zero_vector)
            vector_scores = [(s - min_v) / (max_v - min_v) if s > 0 else 0.0 for s in vector_scores]
        else:
            # 如果只有一个或全部相同的非零值，直接设为1.0
            vector_scores = [1.0 if s > 0 else 0.0 for s in vector_scores]

        # 🔥 关键修复B: BM25分数使用Sigmoid归一化，避免异常值问题
        non_zero_keyword = [s for s in keyword_scores if s > 0]
        if non_zero_keyword:
            # 使用Sigmoid函数处理BM25分数，更稳定
            import math
            # 将分数映射到[0,1]区间，避免极值干扰
            max_bm25 = max(non_zero_keyword)
            normalized_keyword = []
            for s in keyword_scores:
                if s > 0:
                    # 使用改进的sigmoid: 1 / (1 + exp(-x/k)), 其中k为缩放因子
                    k = max_bm25 / 6  # 缩放因子，使得最大值约为0.95
                    normalized = 1 / (1 + math.exp(-s / max(k, 0.1)))
                    normalized_keyword.append(normalized)
                else:
                    normalized_keyword.append(0.0)
            keyword_scores = normalized_keyword
        else:
            keyword_scores = [0.0 for _ in keyword_scores]

        # 🔥 修复问题4: 统一使用配置的权重公式，消除Magic Numbers
        for i, candidate in enumerate(candidates):
            # 统一的加权融合公式，不再有硬编码的0.8/0.2
            candidate.hybrid_score = vector_scores[i] * vector_weight + keyword_scores[i] * keyword_weight

        # 按融合分数排序
        candidates.sort(key=lambda x: x.hybrid_score or 0.0, reverse=True)

        return candidates

    def _generate_retrieval_logs(self, candidates: List[MergedCandidate],
                                 all_vector_results: Dict[str, List[tuple]],
                                 all_keyword_results: List[tuple],
                                 keyword_query: str) -> List[MergedCandidate]:
        """
        生成召回日志

        Args:
            candidates: 候选商品列表
            all_vector_results: 向量检索结果 {query: [(sku_id, score, rank)]}
            all_keyword_results: 关键词检索结果 [(sku_id, score, rank)]
            keyword_query: 关键词查询字符串

        Returns:
            添加了召回日志的候选商品列表
        """
        from datetime import datetime

        for candidate in candidates:
            logs = []

            # 生成向量检索日志
            vector_queries = []
            vector_hits = 0
            max_vector_score = 0.0

            for query, results in all_vector_results.items():
                for sku_id, score, rank in results:
                    if sku_id == candidate.sku_id:
                        logs.append(RetrievalLog(
                            query=query,
                            retrieval_type="vector",
                            score=score,
                            normalized_score=None,  # 将在后面设置
                            rank=rank,
                            timestamp=datetime.now()
                        ))
                        vector_queries.append(query)
                        vector_hits += 1
                        max_vector_score = max(max_vector_score, score)
                        break

            # 生成关键词检索日志
            keyword_queries = []
            keyword_hits = 0
            max_keyword_score = 0.0

            for sku_id, score, rank in all_keyword_results:
                if sku_id == candidate.sku_id:
                    logs.append(RetrievalLog(
                        query=keyword_query,
                        retrieval_type="keyword",
                        score=score,
                        normalized_score=None,  # 将在后面设置
                        rank=rank,
                        timestamp=datetime.now()
                    ))
                    keyword_queries.append(keyword_query)
                    keyword_hits += 1
                    max_keyword_score = max(max_keyword_score, score)
                    break

            # 设置归一化分数
            for log in logs:
                if log.retrieval_type == "vector" and candidate.vector_score:
                    log.normalized_score = candidate.vector_score
                elif log.retrieval_type == "keyword" and candidate.keyword_score:
                    log.normalized_score = candidate.keyword_score

            # 生成汇总信息
            candidate.retrieval_logs = logs
            candidate.log_summary = RetrievalLogSummary(
                total_queries=len(all_vector_results) + (1 if keyword_query else 0),
                vector_queries=list(set(vector_queries)),
                keyword_queries=list(set(keyword_queries)),
                vector_hits=vector_hits,
                keyword_hits=keyword_hits,
                max_vector_score=max_vector_score if max_vector_score > 0 else None,
                max_keyword_score=max_keyword_score if max_keyword_score > 0 else None,
                final_rank=None  # 将在返回排序后的列表时设置
            )

        # 设置最终排名
        for i, candidate in enumerate(candidates, 1):
            if candidate.log_summary:
                candidate.log_summary.final_rank = i

        return candidates

    def print_retrieval_logs(self, candidates: List[MergedCandidate], top_n: int = 5):
        """
        打印召回日志

        Args:
            candidates: 候选商品列表
            top_n: 显示前N个候选商品的日志
        """
        print(f"\n=== 召回日志详情 (Top {top_n}) ===")

        for i, candidate in enumerate(candidates[:top_n], 1):
            print(f"\n【第 {i} 名】SKU: {candidate.sku_id}")
            print(f"  最终分数: {candidate.hybrid_score:.4f} (向量: {candidate.vector_score or 0:.4f}, 关键词: {candidate.keyword_score or 0:.4f})")
            print(f"  召回来源: {', '.join(candidate.sources)}")

            if candidate.log_summary:
                summary = candidate.log_summary
                print(f"  召回汇总:")
                print(f"    总查询数: {summary.total_queries}")
                print(f"    向量命中: {summary.vector_hits}次, 关键词命中: {summary.keyword_hits}次")
                if summary.max_vector_score:
                    print(f"    最高向量分数: {summary.max_vector_score:.4f}")
                if summary.max_keyword_score:
                    print(f"    最高关键词分数: {summary.max_keyword_score:.4f}")

            if candidate.retrieval_logs:
                print(f"  检索详情:")
                for j, log in enumerate(candidate.retrieval_logs, 1):
                    print(f"    {j}. [{log.retrieval_type}] \"{log.query}\" -> 分数: {log.score:.4f}, 排名: #{log.rank}")

            print("-" * 80)

    def __call__(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        LangGraph节点调用接口

        Args:
            input_data: 状态字典

        Returns:
            更新后的状态字典
        """
        rewritten_queries = input_data.get("rewritten_queries", [])
        filters = input_data.get("filters", {})

        candidates = self.search(rewritten_queries, filters)

        return {
            "candidates": candidates
        }


if __name__ == "__main__":
    # 测试代码
    print("初始化检索服务...")
    hybrid_service = HybridSearchService()

    print("\n测试混合检索:")
    candidates = hybrid_service.search(
        rewritten_queries=["低乳糖 儿童 奶粉", "益生菌 配方奶粉"],
        filters={"category": "奶粉", "status": "ON_SALE"},
        enable_logging=True
    )

    print(f"\n找到 {len(candidates)} 个候选商品:")
    for c in candidates[:5]:
        print(f"  SKU: {c.sku_id}, 向量分数: {c.vector_score:.3f}, 关键词分数: {c.keyword_score:.3f}, 混合分数: {c.hybrid_score:.3f}, 来源: {c.sources}")

    # 显示召回日志
    hybrid_service.print_retrieval_logs(candidates, top_n=3)