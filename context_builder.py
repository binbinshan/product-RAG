"""
Module 4: 商品上下文构建服务
职责: 将Top-N商品整理为LLM可消费的上下文
"""
from typing import List, Dict, Any
from models import ContextBuilderInput, ContextBuilderOutput, ProductContext


class ContextBuilderService:
    """上下文构建服务"""

    def __init__(self, max_tokens: int = 2000):
        """
        初始化上下文构建器

        Args:
            max_tokens: 最大token数限制
        """
        self.max_tokens = max_tokens

        # 商品数据库引用
        from hybrid_search import ProductDatabase
        self.db = ProductDatabase()
        self.product_map = {p["sku_id"]: p for p in self.db.products}

    def build_context(self, input_data: ContextBuilderInput) -> ContextBuilderOutput:
        """
        构建商品上下文

        Args:
            input_data: 上下文构建输入

        Returns:
            ContextBuilderOutput: 商品上下文列表
        """
        contexts = []
        total_length = 0

        for sku_id in input_data.sku_ids:
            product = self.product_map.get(sku_id)
            if not product:
                continue

            # 构建单个商品上下文
            context = ProductContext(
                sku_id=sku_id,
                title=product.get("title", ""),
                highlights=product.get("tags", []),
                instructions=f"适合{product.get('age_range', '')}岁使用" if product.get('age_range') else "",
                description=product.get("description", "")
            )

            # 估算token数 (简单估计: 中文1字约1.5token)
            context_text = f"{context.title} {context.description} {' '.join(context.highlights)}"
            estimated_tokens = len(context_text) * 1.5

            # 检查是否超过限制
            if total_length + estimated_tokens > self.max_tokens:
                print(f"警告: 达到token限制,停止添加更多商品上下文")
                break

            contexts.append(context)
            total_length += estimated_tokens

        return ContextBuilderOutput(context=contexts)

    def __call__(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        LangGraph节点调用接口

        Args:
            input_data: 状态字典

        Returns:
            更新后的状态字典
        """
        builder_input = ContextBuilderInput(
            sku_ids=input_data.get("ranked_skus", [])
        )

        output = self.build_context(builder_input)

        return {
            "product_context": output.context
        }


if __name__ == "__main__":
    # 测试代码
    import json

    service = ContextBuilderService(max_tokens=2000)

    input_data = ContextBuilderInput(
        sku_ids=["SKU_1001", "SKU_3003", "SKU_5005"]
    )

    result = service.build_context(input_data)
    print("上下文构建结果:")
    for ctx in result.context:
        print(f"\nSKU: {ctx.sku_id}")
        print(f"标题: {ctx.title}")
        print(f"亮点: {', '.join(ctx.highlights)}")
        print(f"说明: {ctx.instructions}")
        print(f"描述: {ctx.description}")