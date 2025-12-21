"""
Module 1: Query改写服务 - 分层重写策略
职责: 将用户原始Query转换为可检索的结构化查询表达
支持多种重写策略: 规则扩展、同义词、LLM优化、检索增强
使用 LangChain 1.1.0 API
"""
import json
import re
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass
from models import QueryRewriteInput, QueryRewriteOutput
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


class QueryType(Enum):
    """查询类型枚举"""
    SIMPLE = "simple"  # 简单关键词查询
    COMPLEX = "complex"  # 复杂描述查询
    BRANDED = "branded"  # 品牌相关查询
    CATEGORY = "category"  # 品类查询
    ATTRIBUTE = "attribute"  # 属性查询
    FUZZY = "fuzzy"  # 模糊描述查询


class RewriteStrategy(Enum):
    """重写策略枚举"""
    RULE_EXPANSION = "rule_expansion"  # 规则扩展
    SYNONYM = "synonym"  # 同义词替换
    LLM_OPTIMIZATION = "llm_optimization"  # LLM优化
    RETRIEVAL_AUGMENTED = "retrieval_augmented"  # 检索增强


@dataclass
class QueryCharacteristics:
    """查询特征分析结果"""
    query_type: QueryType
    complexity_score: float
    has_brand: bool
    has_category: bool
    has_attributes: bool
    word_count: int
    recommended_strategies: List[RewriteStrategy]


class BaseRewriteStrategy(ABC):
    """重写策略基类"""

    @abstractmethod
    def rewrite(self, query: str, context: Dict[str, Any], characteristics: QueryCharacteristics) -> List[str]:
        """执行查询重写"""
        pass

    @abstractmethod
    def extract_filters(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """提取过滤条件"""
        pass


class QueryAnalyzer:
    """查询特征分析器"""

    def __init__(self):
        self.brand_keywords = {
            "苹果", "华为", "小米", "oppo", "vivo", "三星", "iphone", "huawei",
            "nike", "adidas", "uniqlo", "优衣库", "zara", "h&m"
        }
        self.category_keywords = {
            "手机", "电脑", "笔记本", "平板", "耳机", "音箱", "相机",
            "衣服", "裤子", "鞋子", "包包", "化妆品", "护肤品", "奶粉", "零食"
        }
        self.attribute_keywords = {
            "颜色": ["红色", "蓝色", "黑色", "白色", "粉色", "绿色"],
            "尺寸": ["大号", "中号", "小号", "xl", "l", "m", "s"],
            "材质": ["棉质", "丝绸", "皮革", "金属", "塑料"],
            "功能": ["防水", "续航", "快充", "无线", "蓝牙"]
        }

    def analyze(self, query: str) -> QueryCharacteristics:
        """分析查询特征"""
        query_lower = query.lower()
        words = list(query.strip())

        # 基础特征
        word_count = len(words)
        has_brand = any(brand in query_lower for brand in self.brand_keywords)
        has_category = any(cat in query_lower for cat in self.category_keywords)
        has_attributes = any(
            any(attr in query_lower for attr in attrs)
            for attrs in self.attribute_keywords.values()
        )

        # 复杂度评分
        complexity_score = self._calculate_complexity(query, word_count, has_brand, has_category, has_attributes)

        # 查询类型判断
        query_type = self._determine_query_type(query_lower, word_count, has_brand, has_category, complexity_score)

        # 推荐策略
        recommended_strategies = self._recommend_strategies(query_type, complexity_score, word_count)

        return QueryCharacteristics(
            query_type=query_type,
            complexity_score=complexity_score,
            has_brand=has_brand,
            has_category=has_category,
            has_attributes=has_attributes,
            word_count=word_count,
            recommended_strategies=recommended_strategies
        )

    def _calculate_complexity(self, query: str, word_count: int, has_brand: bool, has_category: bool, has_attributes: bool) -> float:
        """计算查询复杂度"""
        score = 0.0

        # 长度因子
        if word_count <= 3:
            score += 0.2
        elif word_count <= 6:
            score += 0.5
        else:
            score += 0.8

        # 结构化程度
        if has_brand:
            score += 0.1
        if has_category:
            score += 0.1
        if has_attributes:
            score += 0.2

        return min(score, 1.0)

    def _determine_query_type(self, query_lower: str, word_count: int, has_brand: bool, has_category: bool, complexity_score: float) -> QueryType:
        """确定查询类型"""
        if has_brand and word_count <= 4:
            return QueryType.BRANDED
        elif has_category and not has_brand:
            return QueryType.CATEGORY
        elif complexity_score < 0.3:
            return QueryType.SIMPLE
        elif complexity_score > 0.7:
            return QueryType.COMPLEX
        elif "什么" in query_lower or "推荐" in query_lower or "好用" in query_lower:
            return QueryType.FUZZY
        else:
            return QueryType.ATTRIBUTE

    def _recommend_strategies(self, query_type: QueryType, complexity_score: float, word_count: int) -> List[RewriteStrategy]:
        """根据查询特征推荐重写策略"""
        strategies = []

        if query_type == QueryType.SIMPLE or word_count <= 3:
            strategies.extend([RewriteStrategy.RULE_EXPANSION, RewriteStrategy.SYNONYM])

        if query_type in [QueryType.COMPLEX, QueryType.FUZZY] or complexity_score > 0.6:
            strategies.append(RewriteStrategy.LLM_OPTIMIZATION)

        if query_type in [QueryType.BRANDED, QueryType.CATEGORY]:
            strategies.append(RewriteStrategy.RETRIEVAL_AUGMENTED)

        # 默认总是包含同义词策略
        if RewriteStrategy.SYNONYM not in strategies:
            strategies.append(RewriteStrategy.SYNONYM)

        return strategies


class RuleExpansionStrategy(BaseRewriteStrategy):
    """规则扩展策略"""

    def __init__(self):
        self.expansion_rules = {
            "手机": ["智能手机", "移动电话", "手机设备"],
            "电脑": ["计算机", "pc", "台式机", "笔记本电脑"],
            "奶粉": ["婴幼儿奶粉", "配方奶粉", "牛奶粉"],
            "不上火": ["温和配方", "低热量", "清淡型", "易消化"],
            "好用": ["实用", "性价比高", "质量好", "推荐"]
        }

    def rewrite(self, query: str, context: Dict[str, Any], characteristics: QueryCharacteristics) -> List[str]:
        """基于规则扩展查询"""
        rewrites = []

        for keyword, expansions in self.expansion_rules.items():
            if keyword in query:
                for expansion in expansions:
                    new_query = query.replace(keyword, expansion)
                    if new_query != query:
                        rewrites.append(new_query)

        # 如果没有匹配的规则，返回原查询的变体
        if not rewrites:
            rewrites = [f"{query} 产品", f"优质 {query}", f"{query} 推荐"]

        return rewrites[:3]

    def extract_filters(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """基于规则提取过滤条件"""
        filters = {}

        if "奶粉" in query:
            filters["category"] = "奶粉"
        if "手机" in query:
            filters["category"] = "手机"
        if "不上火" in query or "温和" in query:
            filters["features"] = "温和配方"

        return filters


class SynonymStrategy(BaseRewriteStrategy):
    """同义词替换策略"""

    def __init__(self):
        self.synonym_dict = {
            "手机": ["手机", "智能手机", "移动电话"],
            "电脑": ["电脑", "计算机", "PC"],
            "好": ["好", "优秀", "不错", "优质"],
            "便宜": ["便宜", "实惠", "性价比高", "经济"],
            "奶粉": ["奶粉", "配方奶", "牛奶粉"],
            "上火": ["上火", "燥热", "热气"]
        }

    def rewrite(self, query: str, context: Dict[str, Any], characteristics: QueryCharacteristics) -> List[str]:
        """基于同义词重写查询"""
        rewrites = []
        original_words = list(query.strip())

        # 生成同义词变体
        for i, word in enumerate(original_words):
            for key, synonyms in self.synonym_dict.items():
                if word in key or key in word:
                    for synonym in synonyms:
                        if synonym != word:
                            new_words = original_words.copy()
                            new_words[i] = synonym
                            rewrites.append("".join(new_words))

        # 如果没有同义词，生成语义相近的表达
        if not rewrites:
            rewrites = [
                f"{query}产品",
                f"高品质{query}",
                f"{query}推荐款"
            ]

        return list(set(rewrites))[:3]

    def extract_filters(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """提取基于同义词的过滤条件"""
        filters = {}

        # 检查价格相关词汇
        if any(word in query for word in ["便宜", "实惠", "经济"]):
            filters["price_range"] = "low"

        return filters


class LLMOptimizationStrategy(BaseRewriteStrategy):
    """LLM优化策略 - 使用原有的LLM逻辑"""

    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个电商商品查询改写专家。你的任务是:
1. 理解用户的原始查询意图
2. 生成3个不同角度的改写查询,用于检索
3. 提取结构化的过滤条件

改写原则:
- 提取核心需求和关键特征
- 考虑同义词和相关术语
- 保持简洁,每个改写3-6个词

输出必须是严格的JSON格式,包含:
{{
  "rewritten_queries": ["改写1", "改写2", "改写3"]
}}

只返回JSON,不要有任何其他文字。"""),
            ("human", """原始查询: {raw_query}
用户上下文: {user_context}
查询特征: {characteristics}

请生成改写结果:""")
        ])

    def rewrite(self, query: str, context: Dict[str, Any], characteristics: QueryCharacteristics) -> List[str]:
        """使用LLM优化查询"""
        try:
            messages = self.prompt.invoke({
                "raw_query": query,
                "user_context": json.dumps(context, ensure_ascii=False),
                "characteristics": f"类型:{characteristics.query_type.value}, 复杂度:{characteristics.complexity_score}"
            })

            response = self.llm.invoke(messages)
            content = response.content.strip()

            # 清理markdown标记
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

            result = json.loads(content)
            return result.get("rewritten_queries", [query])
        except Exception as e:
            print(f"LLM优化失败: {e}")
            return [query]

    def extract_filters(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """LLM提取过滤条件"""
        # 可以扩展为使用LLM提取更复杂的过滤条件
        return {}


class RetrievalAugmentedStrategy(BaseRewriteStrategy):
    """检索增强重写策略"""

    def __init__(self, knowledge_base: Optional[Dict[str, Any]] = None):
        self.knowledge_base = knowledge_base or {
            "品牌映射": {
                "苹果": ["iPhone", "iPad", "MacBook", "Apple"],
                "华为": ["Huawei", "HUAWEI", "华为手机", "华为笔记本"]
            },
            "品类扩展": {
                "奶粉": ["婴幼儿奶粉", "成人奶粉", "配方奶粉", "有机奶粉"],
                "手机": ["智能手机", "5G手机", "拍照手机", "游戏手机"]
            }
        }

    def rewrite(self, query: str, context: Dict[str, Any], characteristics: QueryCharacteristics) -> List[str]:
        """基于知识库增强重写"""
        rewrites = []

        # 品牌扩展
        for brand, variants in self.knowledge_base.get("品牌映射", {}).items():
            if brand in query:
                for variant in variants:
                    new_query = query.replace(brand, variant)
                    rewrites.append(new_query)

        # 品类扩展
        for category, variants in self.knowledge_base.get("品类扩展", {}).items():
            if category in query:
                for variant in variants:
                    rewrites.append(f"{variant} {query.replace(category, '').strip()}")

        return rewrites[:3] if rewrites else [query]

    def extract_filters(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """基于知识库提取过滤条件"""
        filters = {}

        # 基于知识库提取品类和品牌信息
        for brand in self.knowledge_base.get("品牌映射", {}).keys():
            if brand in query:
                filters["brand"] = brand
                break

        return filters


class HierarchicalQueryRewriteService:
    """分层查询重写服务"""

    def __init__(self, llm: ChatOpenAI, knowledge_base: Optional[Dict[str, Any]] = None):
        self.analyzer = QueryAnalyzer()
        self.strategies = {
            RewriteStrategy.RULE_EXPANSION: RuleExpansionStrategy(),
            RewriteStrategy.SYNONYM: SynonymStrategy(),
            RewriteStrategy.LLM_OPTIMIZATION: LLMOptimizationStrategy(llm),
            RewriteStrategy.RETRIEVAL_AUGMENTED: RetrievalAugmentedStrategy(knowledge_base)
        }

    def rewrite(self, input_data: QueryRewriteInput) -> QueryRewriteOutput:
        """执行分层查询重写"""
        query = input_data.raw_query
        context = input_data.user_context

        # 分析查询特征
        characteristics = self.analyzer.analyze(query)

        # 收集所有重写结果
        all_rewrites = []
        all_filters = {}

        # 根据推荐策略执行重写
        for strategy_type in characteristics.recommended_strategies:
            strategy = self.strategies[strategy_type]
            rewrites = strategy.rewrite(query, context, characteristics)
            filters = strategy.extract_filters(query, context)

            all_rewrites.extend(rewrites)
            all_filters.update(filters)

        # 去重并限制数量
        unique_rewrites = list(dict.fromkeys(all_rewrites))  # 保持顺序的去重
        final_rewrites = unique_rewrites[:3] if unique_rewrites else [query]

        return QueryRewriteOutput(
            rewritten_queries=final_rewrites,
            filters=all_filters
        )

    def __call__(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """LangGraph节点调用接口"""
        query_input = QueryRewriteInput(
            raw_query=input_data.get("raw_query", ""),
            user_context=input_data.get("user_context", {})
        )
        output = self.rewrite(query_input)

        return {
            "rewritten_queries": output.rewritten_queries,
            "filters": output.filters
        }


class QueryRewriteService:
    """Query改写服务 - 兼容原有接口的封装类"""

    def __init__(self, llm: ChatOpenAI, use_hierarchical: bool = True, knowledge_base: Optional[Dict[str, Any]] = None):
        """
        初始化Query改写服务

        Args:
            llm: LangChain LLM实例
            use_hierarchical: 是否使用分层重写策略，默认True
            knowledge_base: 知识库，用于检索增强重写
        """
        self.llm = llm
        self.use_hierarchical = use_hierarchical

        if use_hierarchical:
            # 使用新的分层重写系统
            self.hierarchical_service = HierarchicalQueryRewriteService(llm, knowledge_base)
        else:
            # 保持原有的LLM-only实现
            self.prompt = ChatPromptTemplate.from_messages([
                ("system", """你是一个电商商品查询改写专家。你的任务是:
1. 理解用户的原始查询意图
2. 生成3个不同角度的改写查询,用于检索
3. 提取结构化的过滤条件

改写原则:
- 提取核心需求和关键特征
- 考虑同义词和相关术语
- 保持简洁,每个改写3-6个词

输出必须是严格的JSON格式,包含:
{{
  "rewritten_queries": ["改写1", "改写2", "改写3"],
  "filters": {{
    "category": "品类",
    "其他过滤条件": "值"
  }}
}}

只返回JSON,不要有任何其他文字。"""),
                ("human", """原始查询: {raw_query}
用户上下文: {user_context}

请生成改写结果:""")
            ])

    def rewrite(self, input_data: QueryRewriteInput) -> QueryRewriteOutput:
        """
        执行Query改写

        Args:
            input_data: 查询改写输入

        Returns:
            QueryRewriteOutput: 改写结果
        """
        if self.use_hierarchical:
            # 使用分层重写系统
            return self.hierarchical_service.rewrite(input_data)
        else:
            # 使用原有的LLM-only实现
            return self._llm_only_rewrite(input_data)

    def _llm_only_rewrite(self, input_data: QueryRewriteInput) -> QueryRewriteOutput:
        """原有的LLM-only重写实现"""
        # 使用最新的invoke API (LangChain 1.1.0)
        messages = self.prompt.invoke({
            "raw_query": input_data.raw_query,
            "user_context": json.dumps(input_data.user_context, ensure_ascii=False)
        })

        # 调用LLM
        response = self.llm.invoke(messages)

        # 解析JSON响应
        try:
            # 清理可能的markdown标记
            content = response.content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

            result = json.loads(content)
            return QueryRewriteOutput(**result)
        except json.JSONDecodeError as e:
            # 如果解析失败,返回基础改写
            print(f"JSON解析失败: {e}, 原始响应: {response.content}")
            return QueryRewriteOutput(
                rewritten_queries=[input_data.raw_query],
                filters={}
            )

    def __call__(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        LangGraph节点调用接口

        Args:
            input_data: 状态字典

        Returns:
            更新后的状态字典
        """
        query_input = QueryRewriteInput(
            raw_query=input_data.get("raw_query", ""),
            user_context=input_data.get("user_context", {})
        )
        output = self.rewrite(query_input)

        return {
            "rewritten_queries": output.rewritten_queries,
            "filters": output.filters
        }


if __name__ == "__main__":
    # 测试代码 - 展示分层重写策略
    import os

    # 模拟LLM (实际使用需要API key)
    class MockLLM:
        def invoke(self, messages):
            class Response:
                content = json.dumps({
                    "rewritten_queries": [
                        "低乳糖 儿童 奶粉",
                        "含益生菌 配方奶粉",
                        "肠胃友好型 奶粉"
                    ]
                }, ensure_ascii=False)
            return Response()

    # 测试不同类型的查询
    test_cases = [
        {
            "name": "简单查询 - 触发规则扩展+同义词策略",
            "query": "不上火的奶粉",
            "context": {"channel": "miniapp", "user_type": "new_user"}
        },
        {
            "name": "品牌查询 - 触发检索增强策略",
            "query": "苹果手机推荐",
            "context": {"channel": "app", "user_type": "regular"}
        },
        {
            "name": "复杂查询 - 触发LLM优化策略",
            "query": "什么牌子的护肤品适合敏感肌肤且价格实惠",
            "context": {"channel": "web", "user_type": "premium"}
        },
        {
            "name": "简单关键词 - 触发规则扩展策略",
            "query": "手机",
            "context": {"channel": "miniapp"}
        }
    ]

    # 创建服务实例
    service = QueryRewriteService(llm=MockLLM(), use_hierarchical=True)

    print("=== 分层查询重写策略测试 ===\n")

    for i, test_case in enumerate(test_cases, 1):
        print(f"测试 {i}: {test_case['name']}")
        print(f"原始查询: {test_case['query']}")

        input_data = QueryRewriteInput(
            raw_query=test_case['query'],
            user_context=test_case['context']
        )

        result = service.rewrite(input_data)

        print("重写结果:")
        for j, rewrite in enumerate(result.rewritten_queries, 1):
            print(f"  {j}. {rewrite}")

        if result.filters:
            print("提取的过滤条件:")
            for key, value in result.filters.items():
                print(f"  {key}: {value}")

        print("-" * 50)

    # 测试原有接口兼容性
    print("\n=== 原有接口兼容性测试 ===")
    legacy_service = QueryRewriteService(llm=MockLLM(), use_hierarchical=False)

    legacy_input = QueryRewriteInput(
        raw_query="不上火的奶粉",
        user_context={"channel": "miniapp"}
    )

    legacy_result = legacy_service.rewrite(legacy_input)
    print("LLM-only模式结果:")
    print(json.dumps(legacy_result.model_dump(), ensure_ascii=False, indent=2))

    print("\n=== 查询特征分析示例 ===")
    analyzer = QueryAnalyzer()

    analysis_cases = [
        "不上火的奶粉",
        "苹果iPhone 14",
        "什么牌子的护肤品好用又便宜",
        "手机"
    ]

    for query in analysis_cases:
        characteristics = analyzer.analyze(query)
        print(f"\n查询: {query}")
        print(f"类型: {characteristics.query_type.value}")
        print(f"复杂度: {characteristics.complexity_score:.2f}")
        print(f"推荐策略: {[s.value for s in characteristics.recommended_strategies]}")
        print(f"特征: 品牌={characteristics.has_brand}, 品类={characteristics.has_category}, 属性={characteristics.has_attributes}")