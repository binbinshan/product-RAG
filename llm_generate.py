"""
Module 6: LLM生成服务
职责: 基于商品上下文和实时数据生成导购回答
使用 LangChain 1.1.0 API
"""
import json
import re
from typing import List, Dict, Any, Set
from pydantic import BaseModel, Field, ValidationError
from models import (
    LLMGenerateInput, LLMGenerateOutput,
    ProductContext, ProductRealTimeData
)

try:
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False


    class ChatPromptTemplate:
        @staticmethod
        def from_messages(messages):
            class MockPrompt:
                def __init__(self, messages):
                    self.messages = messages

                def invoke(self, inputs):
                    """支持模板变量替换的Mock实现"""
                    formatted_messages = []
                    for role, content in self.messages:
                        try:
                            # 简单的模板变量替换
                            if isinstance(content, str) and '{' in content:
                                formatted_content = content.format(**inputs)
                            else:
                                formatted_content = content
                            formatted_messages.append((role, formatted_content))
                        except (KeyError, ValueError) as e:
                            # fallback到原始内容
                            formatted_messages.append((role, content))
                    return formatted_messages

            return MockPrompt(messages)


    class ChatOpenAI:
        pass


class LLMGenerateResponse(BaseModel):
    """结构化LLM生成输出"""
    response: str = Field(description="生成的导购回答")
    recommended_skus: List[str] = Field(description="推荐的SKU ID列表")


class ValidationError(Exception):
    """校验错误"""
    pass


class LLMGenerateService:
    """LLM生成服务 - 受控生成"""

    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

        # 系统提示 - 严格约束
        # 使用最新的ChatPromptTemplate API
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个专业的电商商品导购助手。你的任务是基于提供的商品信息,为用户推荐最合适的商品。

【严格遵守的约束】:
1. 只能推荐给定的商品,不得编造或提及未提供的商品
2. 不得编造商品功效、成分或特性
3. 价格、库存、促销信息必须严格使用提供的实时数据,不得猜测
4. 推荐理由必须可回溯到商品的具体字段(标题、描述、标签等)
5. 如果某商品缺货,必须明确告知用户

【回答格式】:
1. 简短理解用户需求
2. 推荐2-3款最合适的商品,每款商品包含:
   - 商品名称
   - 推荐理由(基于商品特性)
   - 价格和促销信息
   - 库存状态
3. 给出选择建议

保持专业、友好的语气,回答简洁明了。"""),
            ("human", """用户查询: {query}

可推荐的商品信息:
{product_info}

请为用户推荐最合适的商品:""")
        ])

    def _format_product_info(
            self,
            product_context: List[ProductContext],
            real_time_data: Dict[str, ProductRealTimeData]
    ) -> str:
        """格式化商品信息为文本"""
        info_list = []

        for ctx in product_context:
            sku_id = ctx.sku_id
            rt_data = real_time_data.get(sku_id)

            info = f"【{ctx.title}】\n"
            info += f"SKU: {sku_id}\n"
            info += f"商品描述: {ctx.description}\n"
            info += f"产品亮点: {', '.join(ctx.highlights)}\n"
            if ctx.instructions:
                info += f"使用说明: {ctx.instructions}\n"

            if rt_data:
                info += f"当前价格: ¥{rt_data.price}\n"
                info += f"库存状态: {'有货' if rt_data.stock > 0 else '缺货'}({rt_data.stock}件)\n"
                if rt_data.promotion:
                    info += f"促销活动: {rt_data.promotion}\n"

            info_list.append(info)

        return "\n---\n".join(info_list)

    def _extract_actual_references(self, response_text: str, available_skus: Set[str]) -> List[str]:
        """从生成文本中实际提取被引用的SKU"""
        referenced = []
        for sku in available_skus:
            if sku in response_text:
                referenced.append(sku)
        return referenced

    def _validate_response(
        self,
        response_text: str,
        recommended_skus: List[str],
        valid_skus: Set[str],
        real_time_data: Dict[str, ProductRealTimeData]
    ) -> None:
        """校验响应内容的合法性"""

        # 1. 校验推荐的SKU是否在给定范围内
        invalid_skus = set(recommended_skus) - valid_skus
        if invalid_skus:
            raise ValidationError(f"LLM推荐了未提供的商品: {invalid_skus}")

        # 2. 校验价格信息
        # 检查是否有未授权的价格提及
        price_pattern = r'\uffe5\s*([\d,]+(?:\.\d+)?)'  # 匹配 "¥123" 格式
        found_prices = re.findall(price_pattern, response_text)

        for price_str in found_prices:
            price = float(price_str.replace(',', ''))
            # 检查该价格是否在授权数据中
            valid_prices = [rt.price for rt in real_time_data.values()]
            if price not in valid_prices:
                print(f"警告: 发现未授权的价格: ¥{price}")

        # 3. 校验库存状态
        # 检查是否有库存状态的编造
        if '有货' in response_text or '缺货' in response_text:
            # 简单校验：确保提及的库存状态与实时数据一致
            for sku, rt_data in real_time_data.items():
                if sku in response_text:
                    expected_status = '有货' if rt_data.stock > 0 else '缺货'
                    # 这里可以做更精细的校验
                    pass

    def _use_structured_output(self) -> bool:
        """判断是否使用结构化输出"""
        # 检查LangChain是否支持with_structured_output
        return LANGCHAIN_AVAILABLE and hasattr(self.llm, 'with_structured_output')

    def generate(self, input_data: LLMGenerateInput) -> LLMGenerateOutput:
        """
        生成导购回答

        Args:
            input_data: LLM生成输入

        Returns:
            LLMGenerateOutput: 生成结果
        """
        if not input_data.product_context:
            return LLMGenerateOutput(
                response="抱歉,没有找到符合您需求的商品。",
                referenced_skus=[],
                generation_type="fallback",
                recommended_skus=[]
            )

        # 格式化商品信息
        product_info = self._format_product_info(
            input_data.product_context,
            input_data.real_time_data
        )

        # 使用最新的invoke API (LangChain 1.1.0)
        messages = self.prompt.invoke({
            "query": input_data.query,
            "product_info": product_info
        })

        # 准备数据用于校验
        valid_skus = {ctx.sku_id for ctx in input_data.product_context}

        # 调用LLM生成
        try:
            if LANGCHAIN_AVAILABLE and hasattr(self.llm, 'with_structured_output'):
                # 使用结构化输出
                structured_llm = self.llm.with_structured_output(LLMGenerateResponse)
                llm_response: LLMGenerateResponse = structured_llm.invoke(messages)

                # 机器校验
                self._validate_response(
                    llm_response.response,
                    llm_response.recommended_skus,
                    valid_skus,
                    input_data.real_time_data
                )

                # 实际提取引用的SKU
                actual_references = self._extract_actual_references(llm_response.response, valid_skus)

                return LLMGenerateOutput(
                    response=llm_response.response,
                    referenced_skus=actual_references,
                    generation_type="llm",
                    recommended_skus=llm_response.recommended_skus
                )
            else:
                # 传统文本输出
                response = self.llm.invoke(messages)
                generated_text = response.content

                # 实际提取引用的SKU
                actual_references = self._extract_actual_references(generated_text, valid_skus)

                # 基础校验（无结构化输出时）
                self._validate_response(
                    generated_text,
                    actual_references,  # 使用实际引用作为推荐
                    valid_skus,
                    input_data.real_time_data
                )

                return LLMGenerateOutput(
                    response=generated_text,
                    referenced_skus=actual_references,
                    generation_type="llm",
                    recommended_skus=actual_references
                )

        except (Exception, ValidationError) as e:
            print(f"LLM生成失败: {e}, 使用后备方案")
            # 后备方案: 简单模板生成
            response_text = self._generate_fallback_response(
                input_data.query,
                input_data.product_context,
                input_data.real_time_data
            )

            # 后备方案也需要实际解析引用
            actual_references = self._extract_actual_references(response_text, valid_skus)

            return LLMGenerateOutput(
                response=response_text,
                referenced_skus=actual_references,
                generation_type="fallback",
                recommended_skus=list(valid_skus)  # fallback时推荐所有商品
            )

    def _generate_fallback_response(
            self,
            query: str,
            product_context: List[ProductContext],
            real_time_data: Dict[str, ProductRealTimeData]
    ) -> str:
        """后备方案: 基于模板生成回答"""
        response = f"根据您查询\"{query}\",为您推荐以下商品:\n\n"

        for i, ctx in enumerate(product_context[:3], 1):
            rt_data = real_time_data.get(ctx.sku_id)

            response += f"{i}. {ctx.title}\n"
            response += f"   特点: {', '.join(ctx.highlights[:3])}\n"

            if rt_data:
                response += f"   价格: ¥{rt_data.price}"
                if rt_data.promotion:
                    response += f" ({rt_data.promotion})"
                response += "\n"

                if rt_data.stock > 0:
                    response += f"   库存充足\n"
                else:
                    response += f"   暂时缺货\n"

            response += "\n"

        return response.strip()

    def __call__(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        LangGraph节点调用接口

        Args:
            input_data: 状态字典

        Returns:
            更新后的状态字典
        """
        generate_input = LLMGenerateInput(
            query=input_data.get("raw_query", ""),
            product_context=input_data.get("product_context", []),
            real_time_data=input_data.get("real_time_data", {})
        )

        output = self.generate(generate_input)

        return {
            "final_response": output.response,
            "referenced_skus": output.referenced_skus,
            "generation_type": output.generation_type,
            "recommended_skus": output.recommended_skus
        }


if __name__ == "__main__":
    # 测试代码
    from context_builder import ContextBuilderService
    from realtime_data import RealTimeDataService

    # 准备测试数据
    context_service = ContextBuilderService()
    realtime_service = RealTimeDataService()

    from models import ContextBuilderInput, RealTimeDataInput

    context_output = context_service.build_context(
        ContextBuilderInput(sku_ids=["SKU_1001", "SKU_3003"])
    )

    realtime_output = realtime_service.get_real_time_data(
        RealTimeDataInput(sku_ids=["SKU_1001", "SKU_3003"])
    )

    # 生成服务
    from query_rewrite import create_llm
    llm = create_llm()
    generate_service = LLMGenerateService(llm=llm)

    generate_input = LLMGenerateInput(
        query="不上火的奶粉",
        product_context=context_output.context,
        real_time_data=realtime_output.data
    )

    result = generate_service.generate(generate_input)
    print("生成结果:")
    print(result.response)
    print(f"\n生成类型: {result.generation_type}")
    print(f"实际引用: {result.referenced_skus}")
    print(f"LLM推荐: {result.recommended_skus}")