"""
Module 3: Rerank重排序服务
职责: 对候选商品进行精排,选出最终Top-N商品
支持LLM重排和Cross-encoder重排（BGE）
使用 LangChain 1.1.0 API
"""
import json
import os
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv
from models import RerankInput, RerankOutput, MergedCandidate, RankedCandidate

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# 加载环境变量
load_dotenv()

# 尝试导入FlagEmbedding
try:
    from FlagEmbedding import FlagReranker
    BGE_AVAILABLE = True
except ImportError:
    print("警告: FlagEmbedding未安装，将使用LLM重排序")
    BGE_AVAILABLE = False

# 模块级缓存，避免重复创建 ProductDatabase
_cached_product_map = None

def _get_product_map() -> Dict[str, Dict[str, Any]]:
    """获取缓存的product_map"""
    global _cached_product_map
    if _cached_product_map is None:
        from hybrid_search import ProductDatabase
        db = ProductDatabase()
        _cached_product_map = {p["sku_id"]: p for p in db.products}
    return _cached_product_map


def _unified_fallback_rerank(candidates: List[MergedCandidate], top_n: int = 5) -> RerankOutput:
    """
    统一的fallback排序逻辑
    基于商品质量而非召回分数进行排序
    """
    if not candidates:
        return RerankOutput(ranked_candidates=[], rerank_type="fallback")

    product_map = _get_product_map()

    candidates_with_score = []
    for candidate in candidates:
        product = product_map.get(candidate.sku_id, {})

        # 基于商品质量的分数计算
        quality_score = 0.0

        # 标题完整度 (0-1)
        title_score = min(len(product.get('title', '')), 100) / 100.0

        # 描述丰富度 (0-1)
        desc_score = min(len(product.get('description', '')), 500) / 500.0

        # 标签数量 (0-1)
        tag_score = min(len(product.get('tags', [])), 10) / 10.0

        # 综合质量分数
        quality_score = 0.5 * title_score + 0.3 * desc_score + 0.2 * tag_score

        candidates_with_score.append((candidate, quality_score))

    # 按质量分数排序
    candidates_with_score.sort(key=lambda x: x[1], reverse=True)

    # 构造排序结果
    ranked_candidates = []
    for candidate, score in candidates_with_score[:top_n]:
        ranked_candidates.append(RankedCandidate(
            sku_id=candidate.sku_id,
            rerank_score=score,
            original_candidate=candidate
        ))

    return RerankOutput(ranked_candidates=ranked_candidates, rerank_type="fallback")


class LLMRerankResponse(BaseModel):
    """结构化LLM重排序输出"""
    ranked_skus: List[str] = Field(description="按相关度排序的SKU ID列表")


class BGERerankService:
    """BGE Cross-encoder重排序服务"""

    def __init__(self, model_name: Optional[str] = None, top_n: int = 5):
        if not BGE_AVAILABLE:
            raise ImportError("FlagEmbedding未安装，无法使用BGE重排序服务")

        self.model_name = model_name or os.getenv('RERANKER_MODEL', 'BAAI/bge-reranker-base')
        self.top_n = top_n

        # 初始化重排模型
        try:
            print(f"正在加载BGE重排模型: {self.model_name}")
            self.reranker = FlagReranker(self.model_name, use_fp16=True)
            print("BGE重排模型加载完成")
        except Exception as e:
            print(f"BGE模型加载失败: {e}")
            self.reranker = None

        # 使用缓存的product_map提升性能
        self.product_map = _get_product_map()

    def _prepare_query_doc_pairs(self, query: str, candidates: List[MergedCandidate]) -> Tuple[List[List[str]], List[MergedCandidate]]:
        """准备查询-文档对，返回有效的pairs和对应的candidates"""
        pairs = []
        valid_candidates = []

        for candidate in candidates:
            product = self.product_map.get(candidate.sku_id, {})

            # 构建文档文本：标题 + 描述 + 标签
            doc_text = f"{product.get('title', '')} {product.get('description', '')} {' '.join(product.get('tags', []))}"
            doc_text = doc_text.strip()

            if doc_text:
                pairs.append([query, doc_text])
                valid_candidates.append(candidate)

        return pairs, valid_candidates

    def rerank(self, input_data: RerankInput) -> RerankOutput:
        """
        执行BGE重排序

        Args:
            input_data: 重排序输入

        Returns:
            RerankOutput: 排序结果
        """
        if not input_data.candidates:
            return RerankOutput(ranked_candidates=[], rerank_type="bge")

        if not self.reranker:
            print("BGE模型未加载，使用基于分数的后备排序")
            return _unified_fallback_rerank(input_data.candidates, self.top_n)

        try:
            # 准备查询-文档对，确保 pairs 与 candidates 数量匹配
            pairs, valid_candidates = self._prepare_query_doc_pairs(input_data.query, input_data.candidates)

            if not pairs:
                print("没有有效的查询-文档对")
                return _unified_fallback_rerank(input_data.candidates, self.top_n)

            # BGE重排序
            scores = self.reranker.compute_score(pairs)

            # 处理单个分数的情况
            if isinstance(scores, (int, float)):
                scores = [scores]

            # 排序 - 现在 pairs 和 valid_candidates 数量匹配
            candidate_scores = list(zip(valid_candidates, scores))
            candidate_scores.sort(key=lambda x: x[1], reverse=True)

            # 获取Top-N
            top_candidates = candidate_scores[:self.top_n]
            ranked_candidates = [
                RankedCandidate(
                    sku_id=candidate.sku_id,
                    rerank_score=score,
                    original_candidate=candidate
                ) for candidate, score in top_candidates
            ]

            return RerankOutput(ranked_candidates=ranked_candidates, rerank_type="bge")

        except Exception as e:
            print(f"BGE重排序失败: {e}, 使用后备排序")
            return _unified_fallback_rerank(input_data.candidates, self.top_n)

    def __call__(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        LangGraph节点调用接口
        """
        rerank_input = RerankInput(
            query=input_data.get("raw_query", ""),
            candidates=input_data.get("candidates", [])
        )

        output = self.rerank(rerank_input)

        return {
            "ranked_candidates": output.ranked_candidates,
            "ranked_skus": output.ranked_skus,  # 向后兼容
            "rerank_type": output.rerank_type
        }


class RerankService:
    """重排序服务 - 使用LLM进行精排"""

    def __init__(self, llm: ChatOpenAI, top_n: int = 5):
        self.llm = llm
        self.top_n = top_n

        # 使用缓存的product_map提升性能
        self.product_map = _get_product_map()

        # 使用 with_structured_output 的 LLM
        self.structured_llm = llm.with_structured_output(LLMRerankResponse)

        # 优化后的简洁prompt，控制token消耗
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """你是商品排序专家。根据用户查询对候选商品进行精准排序。

排序优先级：
1. 语义相关度 (最重要)
2. 商品质量匹配度
3. 文本丰富度

只返回top {top_n}个最相关SKU。"""),
            ("human", "查询: {query}\n\n候选商品:\n{candidates_info}")
        ])

    def _format_candidates_compact(self, candidates: List[MergedCandidate]) -> str:
        """简洁格式化候选商品，减少token消耗"""
        info_list = []
        for i, candidate in enumerate(candidates, 1):
            product = self.product_map.get(candidate.sku_id, {})

            # 只包含关键信息，去除召回分数
            title = product.get('title', 'N/A')[:100]  # 限制标题长度
            desc = product.get('description', 'N/A')[:200]  # 限制描述长度
            tags = ', '.join(product.get('tags', [])[:5])  # 限制标签数量

            info = f"{i}. {candidate.sku_id}: {title}"
            if desc != 'N/A':
                info += f" | {desc}"
            if tags:
                info += f" | 标签: {tags}"

            info_list.append(info)

        return "\n".join(info_list)


    def rerank(self, input_data: RerankInput) -> RerankOutput:
        """
        执行重排序

        Args:
            input_data: 重排序输入

        Returns:
            RerankOutput: 排序结果
        """
        if not input_data.candidates:
            return RerankOutput(ranked_candidates=[], rerank_type="llm")

        # 简洁格式化候选商品信息，减少token消耗
        candidates_info = self._format_candidates_compact(input_data.candidates)

        # 使用 structured output 调用LLM
        try:
            messages = self.prompt.invoke({
                "query": input_data.query,
                "candidates_info": candidates_info,
                "top_n": self.top_n
            })

            # 使用结构化输出，去除JSON解析失败风险
            response: LLMRerankResponse = self.structured_llm.invoke(messages)

            # 构建结构化结果
            ranked_candidates = []
            for i, sku_id in enumerate(response.ranked_skus[:self.top_n]):
                # 找到对应的candidate
                candidate = next((c for c in input_data.candidates if c.sku_id == sku_id), None)
                if candidate:
                    ranked_candidates.append(RankedCandidate(
                        sku_id=sku_id,
                        rerank_score=1.0 - (i * 0.1),  # 位置分数（仅表示排序，非绝对分数）
                        original_candidate=candidate
                    ))

            return RerankOutput(ranked_candidates=ranked_candidates, rerank_type="llm")

        except Exception as e:
            print(f"LLM重排序失败: {e}, 使用后备排序")
            return _unified_fallback_rerank(input_data.candidates, self.top_n)

    def __call__(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        LangGraph节点调用接口

        Args:
            input_data: 状态字典

        Returns:
            更新后的状态字典
        """
        rerank_input = RerankInput(
            query=input_data.get("raw_query", ""),
            candidates=input_data.get("candidates", [])
        )

        output = self.rerank(rerank_input)

        return {
            "ranked_candidates": output.ranked_candidates,
            "ranked_skus": output.ranked_skus,  # 向后兼容
            "rerank_type": output.rerank_type
        }


def create_reranker(reranker_type: Optional[str] = None, top_n: Optional[int] = None):
    """
    工厂函数：根据环境变量配置创建重排序服务


    Args:
        reranker_type: 重排序类型，'bge' 或 'llm'，如果为None则使用环境变量
        top_n: 返回的Top-N数量，如果为None则使用环境变量

    Returns:
        重排序服务实例
    """
    # 使用环境变量或传入参数
    reranker_type = (reranker_type or os.getenv('RERANKER_TYPE', 'bge')).lower()
    top_n = top_n or int(os.getenv('RERANK_TOP_N', '5'))

    if reranker_type == "bge" and BGE_AVAILABLE:
        return BGERerankService(top_n=top_n)
    elif reranker_type == "llm":
        from query_rewrite import create_llm
        llm = create_llm()
        return RerankService(llm=llm, top_n=top_n)
    else:
        # 降级到基于分数的排序
        print("使用基于分数的后备重排序")

        class FallbackReranker:
            def __init__(self, top_n: int = 5):
                self.top_n = top_n

            def rerank(self, input_data: RerankInput) -> RerankOutput:
                return _unified_fallback_rerank(input_data.candidates, self.top_n)

            def __call__(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
                rerank_input = RerankInput(
                    query=input_data.get("raw_query", ""),
                    candidates=input_data.get("candidates", [])
                )
                output = self.rerank(rerank_input)
                return {
                    "ranked_candidates": output.ranked_candidates,
                    "ranked_skus": output.ranked_skus,  # 向后兼容
                    "rerank_type": output.rerank_type
                }

        return FallbackReranker(top_n=top_n)


if __name__ == "__main__":
    # 测试代码
    from hybrid_search import VectorRetrievalService, KeywordRetrievalService, HybridSearchService

    print("=== 重排序服务测试 ===\n")

    # 初始化检索服务
    print("初始化检索服务...")
    vector_service = VectorRetrievalService()
    keyword_service = KeywordRetrievalService()
    hybrid_service = HybridSearchService(vector_service, keyword_service)

    # 获取候选商品
    print("获取候选商品...")
    candidates = hybrid_service.search(
        rewritten_queries=["低乳糖 儿童 奶粉", "益生菌 奶粉"],
        filters={"category": "奶粉"}
    )

    if not candidates:
        print("没有找到候选商品")
        exit(1)

    print(f"找到 {len(candidates)} 个候选商品")

    # 测试查询
    test_query = "不上火的奶粉"
    rerank_input = RerankInput(
        query=test_query,
        candidates=candidates[:10]  # 限制候选数量用于测试
    )

    # 测试BGE重排序
    print("\n1. 测试BGE重排序:")
    try:
        bge_reranker = create_reranker("bge", top_n=3)
        bge_result = bge_reranker.rerank(rerank_input)
        print("BGE重排序结果:")
        for i, sku in enumerate(bge_result.ranked_skus, 1):
            print(f"  {i}. {sku}")
    except Exception as e:
        print(f"BGE重排序失败: {e}")

    # 测试LLM重排序
    print("\n2. 测试LLM重排序:")
    try:
        llm_reranker = create_reranker("llm", top_n=3)
        llm_result = llm_reranker.rerank(rerank_input)
        print("LLM重排序结果:")
        for i, sku in enumerate(llm_result.ranked_skus, 1):
            print(f"  {i}. {sku}")
    except Exception as e:
        print(f"LLM重排序失败: {e}")

    print("\n测试完成!")