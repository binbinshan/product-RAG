"""
RAG测试数据集
包含标准查询、标准答案、评估指标
"""
from typing import List, Dict, Any
from pydantic import BaseModel
import json


class TestQuery(BaseModel):
    """单个测试查询"""
    query_id: str
    query: str
    user_context: Dict[str, Any] = {}
    expected_skus: List[str] = []  # 期望召回的SKU
    relevant_skus: List[str] = []  # 相关的所有SKU
    irrelevant_skus: List[str] = []  # 不相关的SKU
    expected_answer_keywords: List[str] = []  # 期望答案包含的关键词
    difficulty: str = "medium"  # easy, medium, hard


class TestDataset:
    """RAG测试数据集"""

    def __init__(self):
        self.queries = self._create_test_queries()

    def _create_test_queries(self) -> List[TestQuery]:
        """创建测试查询集合"""
        return [
            # 简单查询 - 明确产品类别
            TestQuery(
                query_id="Q001",
                query="不上火的奶粉",
                user_context={"channel": "miniapp", "age_range": "3-6"},
                expected_skus=["SKU_1001", "SKU_3003"],
                relevant_skus=["SKU_1001", "SKU_3003", "SKU_5005", "SKU_2002"],
                irrelevant_skus=["SKU_7001", "SKU_8001"],  # 成人产品
                expected_answer_keywords=["低乳糖", "益生菌", "肠胃", "不上火"],
                difficulty="easy"
            ),

            TestQuery(
                query_id="Q002",
                query="适合3岁宝宝的有机奶粉",
                user_context={"channel": "app", "user_type": "premium"},
                expected_skus=["SKU_2002", "SKU_4004"],
                relevant_skus=["SKU_2002", "SKU_4004", "SKU_1001"],
                irrelevant_skus=["SKU_6001", "SKU_7001"],  # 成人或其他年龄段
                expected_answer_keywords=["有机", "3岁", "天然", "配方"],
                difficulty="easy"
            ),

            # 中等复杂查询 - 多个属性
            TestQuery(
                query_id="Q003",
                query="含益生菌的配方奶粉推荐",
                user_context={"channel": "miniapp"},
                expected_skus=["SKU_3003", "SKU_5005"],
                relevant_skus=["SKU_3003", "SKU_5005", "SKU_1001"],
                irrelevant_skus=["SKU_6001", "SKU_7001"],
                expected_answer_keywords=["益生菌", "配方", "肠道健康"],
                difficulty="medium"
            ),

            TestQuery(
                query_id="Q004",
                query="高端进口奶粉有哪些选择",
                user_context={"channel": "app", "user_type": "premium"},
                expected_skus=["SKU_4004", "SKU_6001"],
                relevant_skus=["SKU_4004", "SKU_6001", "SKU_2002"],
                irrelevant_skus=["SKU_1001", "SKU_8001"],
                expected_answer_keywords=["进口", "高端", "品质", "营养"],
                difficulty="medium"
            ),

            # 复杂查询 - 模糊需求
            TestQuery(
                query_id="Q005",
                query="我家宝宝总是拉肚子，想要个好消化的",
                user_context={"channel": "miniapp", "user_type": "new_user"},
                expected_skus=["SKU_1001", "SKU_3003"],
                relevant_skus=["SKU_1001", "SKU_3003", "SKU_5005"],
                irrelevant_skus=["SKU_7001", "SKU_8001"],
                expected_answer_keywords=["好消化", "益生菌", "肠胃", "低乳糖"],
                difficulty="hard"
            ),

            TestQuery(
                query_id="Q006",
                query="想要性价比高的，但是营养要跟得上",
                user_context={"channel": "app", "user_type": "price_sensitive"},
                expected_skus=["SKU_1001", "SKU_3003"],
                relevant_skus=["SKU_1001", "SKU_3003", "SKU_2002"],
                irrelevant_skus=["SKU_4004", "SKU_6001"],  # 高端产品
                expected_answer_keywords=["性价比", "营养", "价格", "优惠"],
                difficulty="hard"
            ),

            # 特殊场景查询
            TestQuery(
                query_id="Q007",
                query="有没有无糖的奶粉",
                user_context={"channel": "miniapp"},
                expected_skus=["SKU_5005"],
                relevant_skus=["SKU_5005"],
                irrelevant_skus=["SKU_1001", "SKU_2002", "SKU_3003"],
                expected_answer_keywords=["无糖", "0蔗糖", "健康"],
                difficulty="medium"
            ),

            TestQuery(
                query_id="Q008",
                query="1岁宝宝能喝的奶粉",
                user_context={"channel": "app", "age_range": "1-3"},
                expected_skus=["SKU_5005", "SKU_1001"],
                relevant_skus=["SKU_5005", "SKU_1001"],
                irrelevant_skus=["SKU_6001", "SKU_7001"],
                expected_answer_keywords=["1岁", "幼儿", "配方", "营养"],
                difficulty="easy"
            )
        ]

    def get_queries_by_difficulty(self, difficulty: str) -> List[TestQuery]:
        """按难度获取查询"""
        return [q for q in self.queries if q.difficulty == difficulty]

    def get_query_by_id(self, query_id: str) -> TestQuery:
        """按ID获取查询"""
        for query in self.queries:
            if query.query_id == query_id:
                return query
        raise ValueError(f"Query {query_id} not found")

    def export_to_json(self, filepath: str):
        """导出到JSON文件"""
        data = [query.model_dump() for query in self.queries]
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load_from_json(cls, filepath: str) -> 'TestDataset':
        """从JSON文件加载"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        dataset = cls()
        dataset.queries = [TestQuery(**item) for item in data]
        return dataset


if __name__ == "__main__":
    # 创建测试数据集
    dataset = TestDataset()

    print("=== RAG测试数据集 ===")
    print(f"总查询数: {len(dataset.queries)}")

    for difficulty in ["easy", "medium", "hard"]:
        queries = dataset.get_queries_by_difficulty(difficulty)
        print(f"{difficulty.capitalize()}难度: {len(queries)}个查询")

    print("\n=== 查询示例 ===")
    for query in dataset.queries[:3]:
        print(f"\n{query.query_id}: {query.query}")
        print(f"期望SKU: {query.expected_skus}")
        print(f"相关SKU: {query.relevant_skus}")
        print(f"难度: {query.difficulty}")

    # 导出数据集
    dataset.export_to_json("test_dataset.json")
    print(f"\n数据集已导出到 test_dataset.json")