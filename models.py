"""
数据传输对象 (DTO) 定义
所有模块间的接口数据结构
"""
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field


# ============ Module 1: Query Rewrite ============
class QueryRewriteInput(BaseModel):
    """Query改写输入"""
    raw_query: str = Field(..., description="原始用户查询")
    user_context: Dict[str, str] = Field(default_factory=dict, description="用户上下文")


class QueryRewriteOutput(BaseModel):
    """Query改写输出"""
    rewritten_queries: List[str] = Field(..., description="改写后的查询列表")
    filters: Dict[str, Any] = Field(default_factory=dict, description="结构化过滤条件")


# ============ Module 2: Hybrid Search ============
class VectorRetrievalInput(BaseModel):
    """向量检索输入"""
    query: str = Field(..., description="查询文本")
    top_k: int = Field(default=50, description="返回结果数量")
    filters: Dict[str, Any] = Field(default_factory=dict, description="过滤条件")


class VectorRetrievalResult(BaseModel):
    """向量检索单条结果"""
    sku_id: str = Field(..., description="商品SKU ID")
    score: float = Field(..., description="相似度分数")
    content: str = Field(..., description="商品描述内容")


class KeywordRetrievalInput(BaseModel):
    """关键词检索输入"""
    keywords: List[str] = Field(..., description="关键词列表")
    filters: Dict[str, Any] = Field(default_factory=dict, description="过滤条件")


class KeywordRetrievalResult(BaseModel):
    """关键词检索单条结果"""
    sku_id: str = Field(..., description="商品SKU ID")
    score: float = Field(..., description="BM25分数")


class MergedCandidate(BaseModel):
    """合并后的候选商品"""
    sku_id: str = Field(..., description="商品SKU ID")
    vector_score: Optional[float] = Field(None, description="向量检索分数")
    keyword_score: Optional[float] = Field(None, description="关键词检索分数")
    sources: List[str] = Field(default_factory=list, description="召回来源")
    content: Optional[str] = Field(None, description="商品内容")


# ============ Module 3: Rerank ============
class RerankInput(BaseModel):
    """重排序输入"""
    query: str = Field(..., description="原始查询")
    candidates: List[MergedCandidate] = Field(..., description="候选商品列表")


class RerankOutput(BaseModel):
    """重排序输出"""
    ranked_skus: List[str] = Field(..., description="排序后的SKU ID列表")


# ============ Module 4: Context Builder ============
class ContextBuilderInput(BaseModel):
    """上下文构建输入"""
    sku_ids: List[str] = Field(..., description="SKU ID列表")


class ProductContext(BaseModel):
    """单个商品上下文"""
    sku_id: str = Field(..., description="商品SKU ID")
    title: str = Field(..., description="商品标题")
    highlights: List[str] = Field(default_factory=list, description="商品亮点")
    instructions: str = Field(default="", description="使用说明")
    description: str = Field(default="", description="商品描述")


class ContextBuilderOutput(BaseModel):
    """上下文构建输出"""
    context: List[ProductContext] = Field(..., description="商品上下文列表")


# ============ Module 5: Real-time Data ============
class RealTimeDataInput(BaseModel):
    """实时数据查询输入"""
    sku_ids: List[str] = Field(..., description="SKU ID列表")


class ProductRealTimeData(BaseModel):
    """单个商品实时数据"""
    price: float = Field(..., description="当前价格")
    stock: int = Field(..., description="库存数量")
    promotion: str = Field(default="", description="促销活动")


class RealTimeDataOutput(BaseModel):
    """实时数据查询输出"""
    data: Dict[str, ProductRealTimeData] = Field(..., description="SKU ID到实时数据的映射")


# ============ Module 6: LLM Generate ============
class LLMGenerateInput(BaseModel):
    """LLM生成输入"""
    query: str = Field(..., description="用户原始查询")
    product_context: List[ProductContext] = Field(..., description="商品上下文")
    real_time_data: Dict[str, ProductRealTimeData] = Field(..., description="实时数据")


class LLMGenerateOutput(BaseModel):
    """LLM生成输出"""
    response: str = Field(..., description="生成的回答")
    referenced_skus: List[str] = Field(default_factory=list, description="引用的SKU列表")


# ============ Module 7: Pipeline State ============
class PipelineState(BaseModel):
    """完整流程状态"""
    # 输入
    raw_query: str = Field(..., description="原始查询")
    user_context: Dict[str, str] = Field(default_factory=dict, description="用户上下文")

    # Query Rewrite
    rewritten_queries: List[str] = Field(default_factory=list)
    filters: Dict[str, Any] = Field(default_factory=dict)

    # Hybrid Search
    candidates: List[MergedCandidate] = Field(default_factory=list)

    # Rerank
    ranked_skus: List[str] = Field(default_factory=list)

    # Context Builder
    product_context: List[ProductContext] = Field(default_factory=list)

    # Real-time Data
    real_time_data: Dict[str, ProductRealTimeData] = Field(default_factory=dict)

    # LLM Generate
    final_response: str = Field(default="")
    referenced_skus: List[str] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True