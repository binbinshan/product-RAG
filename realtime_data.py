"""
Module 5: 实时商品数据服务
职责: 提供真实价格、库存、促销信息 (模拟API)
"""
import random
from typing import Dict, Any
from models import RealTimeDataInput, RealTimeDataOutput, ProductRealTimeData


class RealTimeDataService:
    """实时商品数据服务 - 模拟API"""

    def __init__(self):
        """初始化实时数据服务"""
        # 模拟实时数据库
        self._real_time_cache = {
            "SKU_1001": ProductRealTimeData(
                price=299.0,
                stock=120,
                promotion="满299减50"
            ),
            "SKU_2002": ProductRealTimeData(
                price=358.0,
                stock=85,
                promotion=""
            ),
            "SKU_3003": ProductRealTimeData(
                price=279.0,
                stock=200,
                promotion="买2送1"
            ),
            "SKU_4004": ProductRealTimeData(
                price=189.0,
                stock=50,
                promotion=""
            ),
            "SKU_5005": ProductRealTimeData(
                price=329.0,
                stock=0,  # 缺货
                promotion="预售中"
            )
        }

    def _simulate_api_call(self, sku_id: str) -> ProductRealTimeData:
        """
        模拟API调用获取实时数据

        Args:
            sku_id: 商品SKU ID

        Returns:
            实时商品数据
        """
        # 如果缓存中有,直接返回
        if sku_id in self._real_time_cache:
            return self._real_time_cache[sku_id]

        # 否则生成随机数据 (模拟新商品)
        return ProductRealTimeData(
            price=round(random.uniform(100, 500), 2),
            stock=random.randint(0, 300),
            promotion=random.choice(["", "新品上市", "限时折扣", "满减活动"])
        )

    def get_real_time_data(self, input_data: RealTimeDataInput) -> RealTimeDataOutput:
        """
        获取实时商品数据

        Args:
            input_data: 实时数据查询输入

        Returns:
            RealTimeDataOutput: 实时数据
        """
        data = {}

        for sku_id in input_data.sku_ids:
            # 模拟API调用延迟 (实际中可能是HTTP请求)
            real_time_data = self._simulate_api_call(sku_id)
            data[sku_id] = real_time_data

        return RealTimeDataOutput(data=data)

    def __call__(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        LangGraph节点调用接口

        Args:
            input_data: 状态字典

        Returns:
            更新后的状态字典
        """
        rt_input = RealTimeDataInput(
            sku_ids=input_data.get("ranked_skus", [])
        )

        output = self.get_real_time_data(rt_input)

        return {
            "real_time_data": output.data
        }


if __name__ == "__main__":
    # 测试代码
    import json

    service = RealTimeDataService()

    input_data = RealTimeDataInput(
        sku_ids=["SKU_1001", "SKU_3003", "SKU_5005"]
    )

    result = service.get_real_time_data(input_data)
    print("实时数据查询结果:")
    for sku_id, data in result.data.items():
        print(f"\nSKU: {sku_id}")
        print(f"价格: ¥{data.price}")
        print(f"库存: {data.stock}")
        print(f"促销: {data.promotion or '无'}")