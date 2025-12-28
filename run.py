"""
ProductRAG - 商品智能导购RAG系统
Product Retrieval-Augmented Generation System

让每一次商品推荐都有据可循

独立运行版本 - 无需安装外部依赖,可直接运行
版本: v1.1.0
"""
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import random
import math


# ============ 数据模型 ============
@dataclass
class QueryRewriteInput:
    raw_query: str
    user_context: Dict[str, str] = field(default_factory=dict)


@dataclass
class QueryRewriteOutput:
    rewritten_queries: List[str]
    filters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VectorRetrievalInput:
    query: str
    top_k: int = 50
    filters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VectorRetrievalResult:
    sku_id: str
    score: float
    content: str


@dataclass
class KeywordRetrievalInput:
    keywords: List[str]
    filters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class KeywordRetrievalResult:
    sku_id: str
    score: float


@dataclass
class MergedCandidate:
    sku_id: str
    vector_score: Optional[float] = None
    keyword_score: Optional[float] = None
    sources: List[str] = field(default_factory=list)
    content: Optional[str] = None


@dataclass
class RerankInput:
    query: str
    candidates: List[MergedCandidate]


@dataclass
class RerankOutput:
    ranked_skus: List[str]


@dataclass
class ContextBuilderInput:
    sku_ids: List[str]


@dataclass
class ProductContext:
    sku_id: str
    title: str
    highlights: List[str] = field(default_factory=list)
    instructions: str = ""
    description: str = ""


@dataclass
class ContextBuilderOutput:
    context: List[ProductContext]


@dataclass
class RealTimeDataInput:
    sku_ids: List[str]


@dataclass
class ProductRealTimeData:
    price: float
    stock: int
    promotion: str = ""


@dataclass
class RealTimeDataOutput:
    data: Dict[str, ProductRealTimeData]


@dataclass
class LLMGenerateInput:
    query: str
    product_context: List[ProductContext]
    real_time_data: Dict[str, ProductRealTimeData]


@dataclass
class LLMGenerateOutput:
    response: str
    referenced_skus: List[str] = field(default_factory=list)


# ============ 商品数据库 ============
class ProductDatabase:
    """模拟商品数据库"""

    def __init__(self):
        self.products = [
            {
                "sku_id": "SKU_1001",
                "title": "儿童低乳糖配方奶粉",
                "category": "奶粉",
                "age_range": "3-6",
                "tags": ["低乳糖", "益生菌", "儿童"],
                "description": "低乳糖配方,适合肠胃敏感儿童,添加益生菌促进消化",
                "status": "ON_SALE"
            },
            {
                "sku_id": "SKU_2002",
                "title": "有机全脂儿童奶粉",
                "category": "奶粉",
                "age_range": "3-6",
                "tags": ["有机", "全脂", "儿童"],
                "description": "100%有机奶源,富含DHA,促进大脑发育",
                "status": "ON_SALE"
            },
            {
                "sku_id": "SKU_3003",
                "title": "肠胃友好型配方奶粉",
                "category": "奶粉",
                "age_range": "3-6",
                "tags": ["易消化", "益生元", "儿童"],
                "description": "特别添加益生元,呵护肠胃健康,减少上火",
                "status": "ON_SALE"
            },
            {
                "sku_id": "SKU_4004",
                "title": "成人高钙奶粉",
                "category": "奶粉",
                "age_range": "18+",
                "tags": ["高钙", "成人"],
                "description": "适合成人补钙,添加维生素D",
                "status": "ON_SALE"
            },
            {
                "sku_id": "SKU_5005",
                "title": "无糖益生菌奶粉",
                "category": "奶粉",
                "age_range": "1-3",
                "tags": ["无糖", "益生菌", "幼儿"],
                "description": "0蔗糖添加,10亿活性益生菌,保护幼儿肠道",
                "status": "ON_SALE"
            }
        ]
        self.product_map = {p["sku_id"]: p for p in self.products}

    def filter_products(self, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """根据条件过滤商品"""
        results = self.products

        if "category" in filters:
            results = [p for p in results if p["category"] == filters["category"]]

        if "status" in filters:
            results = [p for p in results if p["status"] == filters["status"]]

        if "age_range" in filters:
            results = [p for p in results if p["age_range"] == filters["age_range"]]

        if "exclude_tags" in filters:
            exclude = filters["exclude_tags"]
            results = [p for p in results if not any(tag in p["tags"] for tag in exclude)]

        return results


# ============ Module 1: Query改写服务 ============
class QueryRewriteService:
    """Query改写服务 - 使用规则+模板"""

    def rewrite(self, input_data: QueryRewriteInput) -> QueryRewriteOutput:
        """执行Query改写"""
        query = input_data.raw_query

        # 简单规则改写
        rewritten_queries = []
        filters = {}

        # 提取关键词
        keywords = query.split()

        # 生成改写查询
        if "奶粉" in query:
            filters["category"] = "奶粉"
            filters["status"] = "ON_SALE"

            if "不上火" in query or "易消化" in query:
                rewritten_queries.extend([
                    "低乳糖 儿童 奶粉",
                    "含益生菌 配方奶粉",
                    "肠胃友好型 奶粉"
                ])
                filters["age_range"] = "3-6"
                filters["exclude_tags"] = ["成人"]

            elif "有机" in query:
                rewritten_queries.extend([
                    "有机 儿童 奶粉",
                    "全脂 DHA 奶粉"
                ])

            else:
                rewritten_queries.append(" ".join(keywords))

        else:
            rewritten_queries = [query]

        return QueryRewriteOutput(
            rewritten_queries=rewritten_queries,
            filters=filters
        )


# ============ Module 2: 检索服务 ============
class SimpleVectorRetrieval:
    """简单向量检索 - 基于关键词匹配"""

    def __init__(self):
        self.db = ProductDatabase()

    def retrieve(self, input_data: VectorRetrievalInput) -> List[VectorRetrievalResult]:
        """执行检索"""
        filtered = self.db.filter_products(input_data.filters)

        results = []
        query_words = set(input_data.query.lower().split())

        for product in filtered:
            # 计算匹配度
            text = f"{product['title']} {product['description']} {' '.join(product['tags'])}"
            text_words = set(text.lower().split())

            # 计算交集
            common = query_words & text_words
            if common:
                score = len(common) / len(query_words)
                results.append(VectorRetrievalResult(
                    sku_id=product["sku_id"],
                    score=score,
                    content=product["description"]
                ))

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:input_data.top_k]


class SimpleKeywordRetrieval:
    """简单关键词检索"""

    def __init__(self):
        self.db = ProductDatabase()

    def retrieve(self, input_data: KeywordRetrievalInput) -> List[KeywordRetrievalResult]:
        """执行检索"""
        filtered = self.db.filter_products(input_data.filters)

        results = []
        for product in filtered:
            text = f"{product['title']} {product['description']} {' '.join(product['tags'])}"

            # 计算关键词出现次数
            score = sum(text.lower().count(kw.lower()) for kw in input_data.keywords)

            if score > 0:
                results.append(KeywordRetrievalResult(
                    sku_id=product["sku_id"],
                    score=float(score)
                ))

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:50]


class HybridSearchService:
    """混合检索服务"""

    def __init__(self):
        self.vector_service = SimpleVectorRetrieval()
        self.keyword_service = SimpleKeywordRetrieval()

    def search(self, rewritten_queries: List[str], filters: Dict[str, Any]) -> List[MergedCandidate]:
        """执行混合检索"""
        candidates_map = {}

        # 向量检索
        for query in rewritten_queries:
            vector_input = VectorRetrievalInput(query=query, top_k=20, filters=filters)
            vector_results = self.vector_service.retrieve(vector_input)

            for result in vector_results:
                if result.sku_id not in candidates_map:
                    candidates_map[result.sku_id] = MergedCandidate(
                        sku_id=result.sku_id,
                        vector_score=result.score,
                        sources=["vector"],
                        content=result.content
                    )
                else:
                    if result.score > candidates_map[result.sku_id].vector_score:
                        candidates_map[result.sku_id].vector_score = result.score
                    if "vector" not in candidates_map[result.sku_id].sources:
                        candidates_map[result.sku_id].sources.append("vector")

        # 关键词检索
        all_keywords = []
        for query in rewritten_queries:
            all_keywords.extend(query.split())

        if all_keywords:
            keyword_input = KeywordRetrievalInput(keywords=list(set(all_keywords)), filters=filters)
            keyword_results = self.keyword_service.retrieve(keyword_input)

            for result in keyword_results:
                if result.sku_id not in candidates_map:
                    candidates_map[result.sku_id] = MergedCandidate(
                        sku_id=result.sku_id,
                        keyword_score=result.score,
                        sources=["keyword"]
                    )
                else:
                    candidates_map[result.sku_id].keyword_score = result.score
                    if "keyword" not in candidates_map[result.sku_id].sources:
                        candidates_map[result.sku_id].sources.append("keyword")

        return list(candidates_map.values())


# ============ Module 3: 重排序服务 ============
class RerankService:
    """重排序服务 - 基于分数"""

    def __init__(self, top_n: int = 5):
        self.top_n = top_n
        self.db = ProductDatabase()

    def rerank(self, input_data: RerankInput) -> RerankOutput:
        """执行重排序"""
        if not input_data.candidates:
            return RerankOutput(ranked_skus=[])

        # 基于混合分数排序
        candidates_with_score = []
        for c in input_data.candidates:
            vector_score = c.vector_score or 0.0
            keyword_score = (c.keyword_score or 0.0) / 10.0  # 归一化
            combined_score = 0.6 * vector_score + 0.4 * keyword_score
            candidates_with_score.append((c.sku_id, combined_score))

        candidates_with_score.sort(key=lambda x: x[1], reverse=True)
        ranked_skus = [sku for sku, _ in candidates_with_score[:self.top_n]]

        return RerankOutput(ranked_skus=ranked_skus)


# ============ Module 4: 上下文构建 ============
class ContextBuilderService:
    """上下文构建服务"""

    def __init__(self, max_tokens: int = 2000):
        self.max_tokens = max_tokens
        self.db = ProductDatabase()

    def build_context(self, input_data: ContextBuilderInput) -> ContextBuilderOutput:
        """构建商品上下文"""
        contexts = []

        for sku_id in input_data.sku_ids:
            product = self.db.product_map.get(sku_id)
            if not product:
                continue

            context = ProductContext(
                sku_id=sku_id,
                title=product.get("title", ""),
                highlights=product.get("tags", []),
                instructions=f"适合{product.get('age_range', '')}岁使用" if product.get('age_range') else "",
                description=product.get("description", "")
            )
            contexts.append(context)

        return ContextBuilderOutput(context=contexts)


# ============ Module 5: 实时数据服务 ============
class RealTimeDataService:
    """实时商品数据服务"""

    def __init__(self):
        self._cache = {
            "SKU_1001": ProductRealTimeData(price=299.0, stock=120, promotion="满299减50"),
            "SKU_2002": ProductRealTimeData(price=358.0, stock=85, promotion=""),
            "SKU_3003": ProductRealTimeData(price=279.0, stock=200, promotion="买2送1"),
            "SKU_4004": ProductRealTimeData(price=189.0, stock=50, promotion=""),
            "SKU_5005": ProductRealTimeData(price=329.0, stock=0, promotion="预售中")
        }

    def get_real_time_data(self, input_data: RealTimeDataInput) -> RealTimeDataOutput:
        """获取实时数据"""
        data = {}
        for sku_id in input_data.sku_ids:
            if sku_id in self._cache:
                data[sku_id] = self._cache[sku_id]
        return RealTimeDataOutput(data=data)


# ============ Module 6: LLM生成服务 ============
class LLMGenerateService:
    """LLM生成服务 - 使用模板"""

    def generate(self, input_data: LLMGenerateInput) -> LLMGenerateOutput:
        """生成导购回答"""
        if not input_data.product_context:
            return LLMGenerateOutput(
                response="抱歉,没有找到符合您需求的商品。",
                referenced_skus=[]
            )

        # 使用模板生成
        response = f'根据您的查询"{input_data.query}",为您推荐以下商品:\n\n'

        for i, ctx in enumerate(input_data.product_context[:3], 1):
            rt_data = input_data.real_time_data.get(ctx.sku_id)

            response += f"{i}. **{ctx.title}** (SKU: {ctx.sku_id})\n"
            response += f"   产品特点: {', '.join(ctx.highlights)}\n"
            response += f"   {ctx.description}\n"

            if rt_data:
                response += f"   价格: ¥{rt_data.price}"
                if rt_data.promotion:
                    response += f" ({rt_data.promotion})"
                response += "\n"

                if rt_data.stock > 0:
                    response += f"   库存: 充足({rt_data.stock}件)\n"
                else:
                    response += f"   库存: 暂时缺货\n"

            response += "\n"

        # 添加建议
        if len(input_data.product_context) > 0:
            response += f"推荐优先选择第一款{input_data.product_context[0].title},最符合您的需求。"

        referenced_skus = [ctx.sku_id for ctx in input_data.product_context]

        return LLMGenerateOutput(
            response=response.strip(),
            referenced_skus=referenced_skus
        )


# ============ Module 7: 完整流程 ============
class ProductRAGPipeline:
    """商品RAG完整流程"""

    def __init__(self):
        print("初始化RAG系统...")
        self.query_rewrite = QueryRewriteService()
        self.hybrid_search = HybridSearchService()
        self.rerank = RerankService(top_n=3)
        self.context_builder = ContextBuilderService()
        self.realtime_data = RealTimeDataService()
        self.llm_generate = LLMGenerateService()
        print("初始化完成!\n")

    def run(self, query: str, user_context: dict = None) -> dict:
        """执行完整RAG流程"""
        print(f"{'=' * 60}")
        print(f"处理查询: {query}")
        print(f"{'=' * 60}\n")

        # 1. Query改写
        print("1. Query改写...")
        rewrite_output = self.query_rewrite.rewrite(
            QueryRewriteInput(raw_query=query, user_context=user_context or {})
        )
        print(f"   改写查询: {rewrite_output.rewritten_queries}")
        print(f"   过滤条件: {rewrite_output.filters}\n")

        # 2. 混合检索
        print("2. 混合检索...")
        candidates = self.hybrid_search.search(
            rewrite_output.rewritten_queries,
            rewrite_output.filters
        )
        print(f"   找到 {len(candidates)} 个候选商品\n")

        # 3. 重排序
        print("3. 重排序...")
        rerank_output = self.rerank.rerank(
            RerankInput(query=query, candidates=candidates)
        )
        print(f"   Top商品: {rerank_output.ranked_skus}\n")

        # 4. 构建上下文
        print("4. 构建上下文...")
        context_output = self.context_builder.build_context(
            ContextBuilderInput(sku_ids=rerank_output.ranked_skus)
        )
        print(f"   准备 {len(context_output.context)} 个商品上下文\n")

        # 5. 获取实时数据
        print("5. 获取实时数据...")
        realtime_output = self.realtime_data.get_real_time_data(
            RealTimeDataInput(sku_ids=rerank_output.ranked_skus)
        )
        print(f"   获取 {len(realtime_output.data)} 个商品的实时数据\n")

        # 6. LLM生成
        print("6. 生成回答...")
        generate_output = self.llm_generate.generate(
            LLMGenerateInput(
                query=query,
                product_context=context_output.context,
                real_time_data=realtime_output.data
            )
        )
        print("   生成完成\n")

        return {
            "raw_query": query,
            "rewritten_queries": rewrite_output.rewritten_queries,
            "filters": rewrite_output.filters,
            "candidates_count": len(candidates),
            "ranked_skus": rerank_output.ranked_skus,
            "final_response": generate_output.response,
            "referenced_skus": generate_output.referenced_skus
        }


# ============ 主程序 ============
def main():
    """运行示例"""
    print("\n" + "=" * 80)
    print(" " * 22 + "ProductRAG v1.1.0")
    print(" " * 16 + "商品智能导购RAG系统演示")
    print(" " * 14 + "让每一次商品推荐都有据可循")
    print("=" * 80 + "\n")

    # 创建流程
    pipeline = ProductRAGPipeline()

    # 测试查询
    test_queries = [
        "不上火的奶粉",
        "适合3岁宝宝的有机奶粉"
    ]

    for query in test_queries:
        result = pipeline.run(query, {"channel": "miniapp", "user_type": "new_user"})

        print("\n" + "=" * 80)
        print("最终结果")
        print("=" * 80)
        print(f"\n{result['final_response']}\n")
        print(f"引用商品: {result['referenced_skus']}")
        print("=" * 80 + "\n")


if __name__ == "__main__":
    main()