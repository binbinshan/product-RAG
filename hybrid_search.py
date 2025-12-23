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
    MergedCandidate
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
        执行向量检索

        Args:
            input_data: 向量检索输入

        Returns:
            检索结果列表
        """
        # 先进行过滤
        filtered_products = self.db.filter_products(input_data.filters)
        filtered_skus = {p["sku_id"] for p in filtered_products}

        # 查询向量编码
        query_vector = self.encoder.encode([input_data.query])
        query_vector = np.array(query_vector).astype('float32')

        if self.use_fallback:
            return self._fallback_retrieve(query_vector[0], filtered_skus, input_data.top_k)

        try:
            # Milvus检索
            search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
            results = self.collection.search(
                data=[query_vector[0].tolist()],
                anns_field="embedding",
                param=search_params,
                limit=input_data.top_k * 2
            )

            # 构建结果
            final_results = []
            for hit in results[0]:
                sku_id = hit.entity.get("sku_id")
                content = hit.entity.get("content")

                # 过滤检查
                if sku_id not in filtered_skus:
                    continue

                # Milvus返回的是距离，转换为相似度分数
                score = 1.0 / (1.0 + float(hit.distance))

                final_results.append(VectorRetrievalResult(
                    sku_id=sku_id,
                    score=score,
                    content=content
                ))

                if len(final_results) >= input_data.top_k:
                    break

            return final_results

        except Exception as e:
            print(f"Milvus检索失败: {e}")
            return self._fallback_retrieve(query_vector[0], filtered_skus, input_data.top_k)

    def _fallback_retrieve(self, query_vector: np.ndarray, filtered_skus: set, top_k: int) -> List[VectorRetrievalResult]:
        """降级检索方法"""
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


class KeywordRetrievalService:
    """关键词检索子模块 - 基于预构建索引"""

    def __init__(self, index_data: Optional[Dict[str, Any]] = None):
        import jieba
        self.jieba = jieba
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
        执行关键词检索

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

        # 构建结果
        results = []
        for idx, score in enumerate(scores):
            sku_id = self.sku_list[idx]

            # 过滤检查
            if sku_id not in filtered_skus:
                continue

            if score > 0:  # 只保留有分数的结果
                results.append(KeywordRetrievalResult(
                    sku_id=sku_id,
                    score=float(score)
                ))

        # 按分数排序
        results.sort(key=lambda x: x.score, reverse=True)

        max_candidates = int(os.getenv('MAX_CANDIDATES', '50'))
        return results[:max_candidates]


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

    def search(self, rewritten_queries: List[str], filters: Dict[str, Any]) -> List[MergedCandidate]:
        """
        执行混合检索

        Args:
            rewritten_queries: 改写后的查询列表
            filters: 过滤条件

        Returns:
            合并后的候选商品列表
        """
        candidates_map: Dict[str, MergedCandidate] = {}

        # 1. 向量检索 - 对每个改写查询
        vector_top_k = int(os.getenv('VECTOR_TOP_K', '20'))
        for query in rewritten_queries:
            vector_input = VectorRetrievalInput(
                query=query,
                top_k=vector_top_k,
                filters=filters
            )
            vector_results = self.vector_service.retrieve(vector_input)

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

        # 3. 转换为列表
        candidates = list(candidates_map.values())

        return candidates

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
        filters={"category": "奶粉", "status": "ON_SALE"}
    )

    print(f"\n找到 {len(candidates)} 个候选商品:")
    for c in candidates[:5]:
        print(f"  SKU: {c.sku_id}, 向量分数: {c.vector_score}, 关键词分数: {c.keyword_score}, 来源: {c.sources}")