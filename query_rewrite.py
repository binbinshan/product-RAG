"""
Module 1: Query改写服务 - 分层重写策略
职责: 将用户原始Query转换为可检索的结构化查询表达
支持多种重写策略: 规则扩展、同义词、LLM优化、检索增强
使用 LangChain 1.1.0 API
"""
import json
import os
import re
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass
from typing_extensions import Literal
from dotenv import load_dotenv
from models import QueryRewriteInput, QueryRewriteOutput
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# 加载环境变量
load_dotenv()


def create_llm() -> ChatOpenAI:
    """
    从环境变量创建LLM实例

    Returns:
        ChatOpenAI实例
    """
    return ChatOpenAI(
        model=os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo'),
        temperature=float(os.getenv('OPENAI_TEMPERATURE', '0.1')),
        base_url=os.getenv("BASE_URL"),
        api_key=os.getenv("DEEP_SEEK_API")
        # api_key和base_url会自动从环境变量OPENAI_API_KEY和OPENAI_BASE_URL读取
    )


class QueryMainType(Enum):
    """查询主类型枚举"""
    SIMPLE = "simple"  # 简单关键词查询
    COMPLEX = "complex"  # 复杂描述查询
    CONVERSATIONAL = "conversational"  # 对话式查询


class IntentTag(Enum):
    """查询意图标签枚举"""
    BRANDED = "branded"  # 品牌相关
    CATEGORY = "category"  # 品类查询
    ATTRIBUTE = "attribute"  # 属性查询
    PRICE = "price"  # 价格相关
    RECOMMENDATION = "recommendation"  # 推荐需求
    COMPARISON = "comparison"  # 比较需求
    REFERENCE = "reference"  # 指代查询
    QUESTION = "question"  # 疑问查询


class RewriteStrategy(Enum):
    """重写策略枚举"""
    RULE_EXPANSION = "rule_expansion"  # 规则扩展
    SYNONYM = "synonym"  # 同义词替换
    LLM_OPTIMIZATION = "llm_optimization"  # LLM优化
    RETRIEVAL_AUGMENTED = "retrieval_augmented"  # 检索增强
    CONTEXT_REFERENCE = "context_reference"  # 上下文指代处理


@dataclass
class RewriteCandidate:
    """重写候选结果"""
    text: str
    source: RewriteStrategy
    confidence: float
    level: Literal["lexical", "semantic", "entity", "context"]
    filters: Dict[str, Any] = None

    def __post_init__(self):
        if self.filters is None:
            self.filters = {}


@dataclass
class FilterSource:
    """过滤条件来源"""
    value: Any
    source: RewriteStrategy
    confidence: float


@dataclass
class QueryCharacteristics:
    """查询特征分析结果"""
    main_type: QueryMainType
    intent_tags: List[IntentTag]
    complexity_score: float
    word_count: int
    recommended_strategies: List[RewriteStrategy]

    # 便捷属性
    @property
    def has_brand(self) -> bool:
        return IntentTag.BRANDED in self.intent_tags

    @property
    def has_category(self) -> bool:
        return IntentTag.CATEGORY in self.intent_tags

    @property
    def has_attributes(self) -> bool:
        return IntentTag.ATTRIBUTE in self.intent_tags

    @property
    def has_reference(self) -> bool:
        return IntentTag.REFERENCE in self.intent_tags


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
        # 品牌关键词
        self.brand_keywords = {
            "苹果", "华为", "小米", "oppo", "vivo", "三星", "iphone", "huawei",
            "nike", "adidas", "uniqlo", "优衣库", "zara", "h&m"
        }
        # 类别关键词
        self.category_keywords = {
            "手机", "电脑", "笔记本", "平板", "耳机", "音箱", "相机",
            "衣服", "裤子", "鞋子", "包包", "化妆品", "护肤品", "奶粉", "零食"
        }
        # 属性关键词
        self.attribute_keywords = {
            "颜色": ["红色", "蓝色", "黑色", "白色", "粉色", "绿色"],
            "尺寸": ["大号", "中号", "小号", "xl", "l", "m", "s"],
            "材质": ["棉质", "丝绸", "皮革", "金属", "塑料"],
            "功能": ["防水", "续航", "快充", "无线", "蓝牙"]
        }
        # 指代词关键词
        self.reference_keywords = {
            "这个", "这", "那个", "那", "它", "这款", "那款", "该", "上面的",
            "刚才的", "之前的", "这种", "那种", "这类", "那类"
        }

    def analyze(self, query: str) -> QueryCharacteristics:
        """分析查询特征"""
        query_lower = query.lower()
        import jieba
        words = list(jieba.cut(query.strip()))

        # 基础特征
        word_count = len(words)

        # 分析意图标签
        intent_tags = self._extract_intent_tags(query_lower)

        # 复杂度评分
        complexity_score = self._calculate_complexity(query, word_count, intent_tags)

        # 主查询类型判断
        main_type = self._determine_main_type(query_lower, word_count, complexity_score, intent_tags)

        # 推荐策略
        recommended_strategies = self._recommend_strategies(main_type, complexity_score, word_count, intent_tags)

        return QueryCharacteristics(
            main_type=main_type,
            intent_tags=intent_tags,
            complexity_score=complexity_score,
            word_count=word_count,
            recommended_strategies=recommended_strategies
        )

    def _extract_intent_tags(self, query_lower: str) -> List[IntentTag]:
        """提取意图标签"""
        tags = []

        # 品牌标签
        if any(brand in query_lower for brand in self.brand_keywords):
            tags.append(IntentTag.BRANDED)

        # 品类标签
        if any(cat in query_lower for cat in self.category_keywords):
            tags.append(IntentTag.CATEGORY)

        # 属性标签
        if any(
            any(attr in query_lower for attr in attrs)
            for attrs in self.attribute_keywords.values()
        ):
            tags.append(IntentTag.ATTRIBUTE)

        # 价格标签
        if any(word in query_lower for word in ["便宜", "实惠", "经济", "价格", "多少钱"]):
            tags.append(IntentTag.PRICE)

        # 推荐标签 - 精确识别，避免知识型查询误判
        if self._is_recommendation_query(query_lower):
            tags.append(IntentTag.RECOMMENDATION)

        # 比较标签
        if any(word in query_lower for word in ["比较", "对比", "哪个好", "vs", "还是"]):
            tags.append(IntentTag.COMPARISON)

        # 指代标签
        if any(ref in query_lower for ref in self.reference_keywords):
            tags.append(IntentTag.REFERENCE)

        # 疑问标签 - 区分推荐和纯疑问
        if self._is_question_query(query_lower) and IntentTag.RECOMMENDATION not in tags:
            tags.append(IntentTag.QUESTION)

        return tags

    def _calculate_complexity(self, query: str, word_count: int, intent_tags: List[IntentTag]) -> float:
        """计算查询复杂度"""
        score = 0.0

        # 长度因子
        if word_count <= 3:
            score += 0.2
        elif word_count <= 6:
            score += 0.5
        else:
            score += 0.8

        # 意图复杂度
        tag_count = len(intent_tags)
        if tag_count <= 1:
            score += 0.1
        elif tag_count <= 3:
            score += 0.2
        else:
            score += 0.3

        return min(score, 1.0)

    def _determine_main_type(self, query_lower: str, word_count: int, complexity_score: float, intent_tags: List[IntentTag]) -> QueryMainType:
        """确定主查询类型"""
        # 对话式查询
        if IntentTag.REFERENCE in intent_tags or IntentTag.QUESTION in intent_tags:
            return QueryMainType.CONVERSATIONAL

        # 复杂查询
        elif complexity_score > 0.7 or len(intent_tags) > 2:
            return QueryMainType.COMPLEX

        # 简单查询
        else:
            return QueryMainType.SIMPLE

    def _recommend_strategies(self, main_type: QueryMainType, complexity_score: float, word_count: int, intent_tags: List[IntentTag]) -> List[RewriteStrategy]:
        """根据查询特征推荐重写策略"""
        strategies = []

        # 如果包含指代词，优先使用上下文指代处理策略
        if IntentTag.REFERENCE in intent_tags:
            strategies.append(RewriteStrategy.CONTEXT_REFERENCE)

        # 简单查询策略
        if main_type == QueryMainType.SIMPLE or word_count <= 3:
            strategies.extend([RewriteStrategy.RULE_EXPANSION, RewriteStrategy.SYNONYM])

        # 复杂查询策略
        if main_type == QueryMainType.COMPLEX or complexity_score > 0.6:
            strategies.append(RewriteStrategy.LLM_OPTIMIZATION)

        # 对话式查询策略
        if main_type == QueryMainType.CONVERSATIONAL:
            if IntentTag.REFERENCE in intent_tags:
                strategies.append(RewriteStrategy.CONTEXT_REFERENCE)
            else:
                strategies.append(RewriteStrategy.LLM_OPTIMIZATION)

        # 基于意图标签的策略
        if IntentTag.BRANDED in intent_tags or IntentTag.CATEGORY in intent_tags:
            strategies.append(RewriteStrategy.RETRIEVAL_AUGMENTED)

        # 默认总是包含同义词策略
        if RewriteStrategy.SYNONYM not in strategies:
            strategies.append(RewriteStrategy.SYNONYM)

        return strategies

    def _is_recommendation_query(self, query_lower: str) -> bool:
        """精确识别推荐查询，避免知识型查询误判"""
        # 明确的推荐意图词汇
        recommend_keywords = ["推荐", "选什么", "哪个好", "什么好", "买什么", "要什么"]

        # 如果直接包含推荐词汇，认为是推荐查询
        if any(keyword in query_lower for keyword in recommend_keywords):
            return True

        # 处理"什么 + 名词"的模式
        import jieba
        words = list(jieba.cut(query_lower))

        # 如果"什么"后面紧跟商品类别词汇，且查询较短，认为是推荐查询
        if "什么" in words:
            什么_index = words.index("什么")
            if 什么_index < len(words) - 1:
                next_word = words[什么_index + 1]
                # 如果下一个词是商品类别，且总词数小于6，认为是推荐查询
                if next_word in self.category_keywords and len(words) <= 5:
                    return True
                # "什么牌子的..." 明确是推荐查询
                if next_word in ["牌子", "品牌"]:
                    return True

        return False

    def _is_question_query(self, query_lower: str) -> bool:
        """识别疑问查询，区分推荐和纯疑问"""
        question_keywords = ["怎么样", "如何", "是否", "有没有", "为什么", "?", "？"]

        # 基础疑问词汇
        if any(keyword in query_lower for keyword in question_keywords):
            return True

        # "什么是..."、"什么叫..."等知识型疑问
        knowledge_patterns = ["什么是", "什么叫", "什么意思", "怎么用", "怎么做"]
        if any(pattern in query_lower for pattern in knowledge_patterns):
            return True

        return False


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
        import jieba
        original_words = list(jieba.cut(query.strip()))

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
                "characteristics": f"主类型:{characteristics.main_type.value}, 意图标签:{[tag.value for tag in characteristics.intent_tags]}, 复杂度:{characteristics.complexity_score}"
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


class ContextReferenceStrategy(BaseRewriteStrategy):
    """上下文指代处理策略"""

    def __init__(self):
        self.reference_patterns = {
            "这个": "商品",
            "这": "商品",
            "那个": "商品",
            "那": "商品",
            "它": "商品",
            "这款": "",  # 空字符串，避免"款"字重复
            "那款": "",  # 空字符串，避免"款"字重复
            "该": "",  # 空字符串
            "上面的": "商品",
            "刚才的": "商品",
            "之前的": "商品",
            "这种": "类型",
            "那种": "类型",
            "这类": "类型",
            "那类": "类型"
        }

    def rewrite(self, query: str, context: Dict[str, Any], characteristics: QueryCharacteristics) -> List[str]:
        """处理指代词查询改写"""
        rewrites = []

        # 从上下文获取最近的商品信息
        last_product = context.get("last_viewed_product", "")
        last_category = context.get("last_category", "")
        conversation_history = context.get("conversation_history", [])

        # 替换指代词 - 优先使用最具体的上下文信息
        for reference, replacement in self.reference_patterns.items():
            if reference in query:
                # 如果有上下文商品信息，优先使用具体商品名替换
                if last_product:
                    # 对于"这款"、"那款"类的词汇，需要特殊处理避免重复
                    if reference in ["这款", "那款"]:
                        # 将"这款"或"那款"整个替换为商品名
                        specific_rewrite = query.replace(reference, last_product, 1)  # 只替换第一个匹配项
                    else:
                        specific_rewrite = query.replace(reference, last_product)
                    rewrites.append(specific_rewrite)

                # 如果有类别信息，使用类别替换
                elif last_category:
                    if reference in ["这款", "那款"]:
                        category_rewrite = query.replace(reference, last_category, 1)
                    else:
                        category_rewrite = query.replace(reference, last_category)
                    rewrites.append(category_rewrite)

                # 基础替换作为兜底
                else:
                    basic_rewrite = query.replace(reference, replacement)
                    rewrites.append(basic_rewrite)

        # 如果没有匹配的指代词，尝试从对话历史推断
        if not rewrites and conversation_history:
            # 从最近的对话中提取商品关键词
            recent_keywords = self._extract_keywords_from_history(conversation_history)
            if recent_keywords:
                for keyword in recent_keywords[:2]:
                    rewrites.append(f"{keyword} {query}")

        # 如果仍然没有结果，返回通用改写
        if not rewrites:
            rewrites = [
                f"{query} 商品",
                f"推荐 {query}",
                f"热门 {query}"
            ]

        return rewrites[:3]

    def _extract_keywords_from_history(self, history: List[str]) -> List[str]:
        """从对话历史中提取商品关键词"""
        keywords = []
        # 简单的关键词提取逻辑，可以扩展为更复杂的NLP处理
        common_products = ["手机", "电脑", "奶粉", "化妆品", "鞋子", "衣服", "耳机"]

        for msg in history[-3:]:  # 只看最近3条消息
            for product in common_products:
                if product in msg and product not in keywords:
                    keywords.append(product)

        return keywords

    def extract_filters(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """从上下文提取过滤条件"""
        filters = {}

        # 如果有上下文信息，提取相关过滤条件
        if context.get("last_category"):
            filters["category"] = context["last_category"]

        if context.get("last_brand"):
            filters["brand"] = context["last_brand"]

        return filters


class HierarchicalQueryRewriteService:
    """分层查询重写服务"""

    def __init__(self, llm: ChatOpenAI, knowledge_base: Optional[Dict[str, Any]] = None):
        self.analyzer = QueryAnalyzer()
        self.strategies = {
            RewriteStrategy.RULE_EXPANSION: RuleExpansionStrategy(),
            RewriteStrategy.SYNONYM: SynonymStrategy(),
            RewriteStrategy.LLM_OPTIMIZATION: LLMOptimizationStrategy(llm),
            RewriteStrategy.RETRIEVAL_AUGMENTED: RetrievalAugmentedStrategy(knowledge_base),
            RewriteStrategy.CONTEXT_REFERENCE: ContextReferenceStrategy()
        }

    def rewrite(self, input_data: QueryRewriteInput) -> QueryRewriteOutput:
        """执行分层查询重写"""
        query = input_data.raw_query
        context = input_data.user_context

        # 分析查询特征
        characteristics = self.analyzer.analyze(query)

        # 第一阶段：上下文指代处理（如果需要）
        resolved_query = query
        if IntentTag.REFERENCE in characteristics.intent_tags:
            reference_strategy = self.strategies[RewriteStrategy.CONTEXT_REFERENCE]
            reference_rewrites = reference_strategy.rewrite(query, context, characteristics)
            if reference_rewrites:
                resolved_query = reference_rewrites[0]  # 使用第一个解决方案作为基础查询
                print(f"指代解析: '{query}' -> '{resolved_query}'")

        # 重新分析解析后的查询特征（可能会改变）
        if resolved_query != query:
            characteristics = self.analyzer.analyze(resolved_query)

        # 收集所有重写结果
        all_rewrites = []
        all_filters = {}

        # 第二阶段：其他策略执行重写（排除已处理的上下文指代）
        remaining_strategies = [s for s in characteristics.recommended_strategies
                                if s != RewriteStrategy.CONTEXT_REFERENCE]

        for strategy_type in remaining_strategies:
            strategy = self.strategies[strategy_type]
            rewrites = strategy.rewrite(resolved_query, context, characteristics)
            filters = strategy.extract_filters(resolved_query, context)

            all_rewrites.extend(rewrites)
            all_filters.update(filters)

        # 如果原查询包含指代，也添加解析后的查询
        if resolved_query != query:
            all_rewrites.insert(0, resolved_query)

        # 去重并限制数量
        unique_rewrites = list(dict.fromkeys(all_rewrites))  # 保持顺序的去重
        final_rewrites = unique_rewrites[:3] if unique_rewrites else [resolved_query]

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

    def __init__(self, llm: ChatOpenAI, knowledge_base: Optional[Dict[str, Any]] = None):
        """
        初始化Query改写服务

        Args:
            llm: LangChain LLM实例
            knowledge_base: 知识库，用于检索增强重写
        """
        self.llm = llm
        # 使用新的分层重写系统
        self.hierarchical_service = HierarchicalQueryRewriteService(llm, knowledge_base)

    def rewrite(self, input_data: QueryRewriteInput) -> QueryRewriteOutput:
        """
        执行Query改写

        Args:
            input_data: 查询改写输入

        Returns:
            QueryRewriteOutput: 改写结果
        """
        # 使用分层重写系统
        return self.hierarchical_service.rewrite(input_data)

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
        },
        {
            "name": "指代词查询 - 基础指代处理",
            "query": "这个怎么样",
            "context": {"channel": "app", "user_type": "regular"}
        },
        {
            "name": "指代词查询 - 有上下文商品信息",
            "query": "这款有其他颜色吗",
            "context": {
                "channel": "app",
                "last_viewed_product": "iPhone 15 Pro",
                "last_category": "手机",
                "last_brand": "苹果"
            }
        },
        {
            "name": "指代词查询 - 从对话历史推断",
            "query": "它的价格是多少",
            "context": {
                "channel": "web",
                "conversation_history": ["我想买手机", "推荐几款苹果手机", "iPhone 15怎么样"]
            }
        }
    ]

    # 创建服务实例
    service = QueryRewriteService(llm=MockLLM())

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

    print("\n=== 查询特征分析示例 ===")
    analyzer = QueryAnalyzer()

    # analysis_cases = [
    #     "不上火的奶粉",
    #     "苹果iPhone 14",
    #     "什么牌子的护肤品好用又便宜",
    #     "手机"
    # ]
    #
    # for query in analysis_cases:
    #     characteristics = analyzer.analyze(query)
    #     print(f"\n查询: {query}")
    #     print(f"类型: {characteristics.query_type.value}")
    #     print(f"复杂度: {characteristics.complexity_score:.2f}")
    #     print(f"推荐策略: {[s.value for s in characteristics.recommended_strategies]}")
    #     print(
    #         f"特征: 品牌={characteristics.has_brand}, 品类={characteristics.has_category}, 属性={characteristics.has_attributes}")
