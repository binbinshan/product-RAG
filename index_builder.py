"""
索引构建工具 - 独立的索引加载和构建功能
职责: 构建向量索引和关键词索引，供检索服务使用
"""
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import pymysql
import json
import os
from dotenv import load_dotenv
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility

# 加载环境变量
load_dotenv()


class ProductDatabase:
    """基于MySQL的商品数据库"""

    def __init__(self, db_config: Optional[Dict[str, Any]] = None):
        if db_config is None:
            self.db_config = {
                'host': os.getenv('DB_HOST', 'localhost'),
                'port': int(os.getenv('DB_PORT', '3306')),
                'user': os.getenv('DB_USER', 'root'),
                'password': os.getenv('DB_PASSWORD', 'root'),
                'database': os.getenv('DB_NAME', 'product_rag'),
                'charset': os.getenv('DB_CHARSET', 'utf8mb4')
            }
        else:
            self.db_config = db_config

        # 缓存所有商品数据
        self.products = self.load_all_products()

    def _get_connection(self):
        """获取数据库连接"""
        try:
            return pymysql.connect(**self.db_config)
        except Exception as e:
            print(f"数据库连接失败: {e}")
            return None

    def load_all_products(self) -> List[Dict[str, Any]]:
        """从数据库加载所有商品数据"""
        conn = self._get_connection()
        if not conn:
            return []

        try:
            with conn.cursor(pymysql.cursors.DictCursor) as cursor:
                cursor.execute("""
                               SELECT sku_id, title, category, age_range, tags, description, status
                               FROM products
                               WHERE status = 'ON_SALE'
                               """)
                results = cursor.fetchall()

                products = []
                for row in results:
                    product = dict(row)
                    if product['tags']:
                        product['tags'] = json.loads(product['tags'])
                    else:
                        product['tags'] = []
                    products.append(product)

                return products
        except Exception as e:
            print(f"查询数据失败: {e}")
            return []
        finally:
            conn.close()

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

    def refresh_products(self):
        """刷新商品缓存"""
        self.products = self.load_all_products()


class VectorIndexBuilder:
    """向量索引构建器"""

    def __init__(self, embedding_model: Optional[str] = None, milvus_config: Optional[Dict[str, Any]] = None):
        # 使用环境变量或默认值
        model_name = embedding_model or os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
        self.encoder = SentenceTransformer(model_name)

        self.milvus_config = milvus_config or {
            'host': os.getenv('MILVUS_HOST', 'localhost'),
            'port': os.getenv('MILVUS_PORT', '19530')
        }
        self.collection_name = os.getenv('MILVUS_COLLECTION_NAME', 'product_embeddings')
        self.db = ProductDatabase()

    def setup_milvus(self) -> bool:
        """设置Milvus连接"""
        try:
            connections.connect(
                alias="default",
                host=self.milvus_config['host'],
                port=self.milvus_config['port']
            )
            print("Milvus连接成功")
            return True
        except Exception as e:
            print(f"Milvus连接失败: {e}")
            return False

    def build_vector_index(self) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        构建向量索引

        Returns:
            (success: bool, index_data: Optional[Dict])
            success: 是否成功
            index_data: 索引数据，包含必要的信息
        """
        products = self.db.load_all_products()
        if not products:
            print("没有找到商品数据")
            return False, None

        sku_list = [p["sku_id"] for p in products]
        content_list = [p["description"] for p in products]
        product_map = {p["sku_id"]: p for p in products}

        # 尝试连接Milvus
        milvus_connected = self.setup_milvus()

        if milvus_connected:
            success = self._build_milvus_index(sku_list, content_list)
            if success:
                return True, {
                    "index_type": "milvus",
                    "sku_list": sku_list,
                    "content_list": content_list,
                    "product_map": product_map,
                    "collection_name": self.collection_name
                }

        # 降级到内存索引
        embeddings = self._build_memory_index(content_list)
        return True, {
            "index_type": "memory",
            "sku_list": sku_list,
            "content_list": content_list,
            "product_map": product_map,
            "embeddings": embeddings
        }

    def _build_milvus_index(self, sku_list: List[str], content_list: List[str]) -> bool:
        """构建Milvus向量索引"""
        try:
            # 删除已存在的collection
            if utility.has_collection(self.collection_name):
                utility.drop_collection(self.collection_name)

            # 编码为向量
            embeddings = self.encoder.encode(content_list)
            embeddings = np.array(embeddings).astype('float32')
            dimension = embeddings.shape[1]

            # 定义schema
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="sku_id", dtype=DataType.VARCHAR, max_length=50),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dimension),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=1000)
            ]
            schema = CollectionSchema(fields, "Product embeddings collection")

            # 创建collection
            collection = Collection(self.collection_name, schema)

            # 插入数据
            entities = [
                sku_list,
                embeddings.tolist(),
                content_list
            ]
            collection.insert(entities)

            # 创建索引
            index_params = {
                "metric_type": "L2",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 1024}
            }
            collection.create_index("embedding", index_params)

            # 加载collection到内存
            collection.load()
            print(f"Milvus向量索引构建完成，共{len(sku_list)}个商品")
            return True

        except Exception as e:
            print(f"Milvus索引构建失败: {e}")
            return False

    def _build_memory_index(self, content_list: List[str]) -> np.ndarray:
        """构建内存降级索引"""
        embeddings = self.encoder.encode(content_list)
        embeddings = np.array(embeddings).astype('float32')
        print(f"构建内存向量索引完成，共{len(content_list)}个商品")
        return embeddings


class KeywordIndexBuilder:
    """关键词索引构建器"""

    def __init__(self):
        import jieba
        self.jieba = jieba
        self.db = ProductDatabase()

    def build_keyword_index(self) -> Optional[Dict[str, Any]]:
        """
        构建关键词索引

        Returns:
            index_data: 索引数据字典
        """
        products = self.db.load_all_products()
        if not products:
            print("没有找到商品数据")
            return None

        sku_list = [p["sku_id"] for p in products]
        product_map = {p["sku_id"]: p for p in products}

        # 准备文档 (标题 + 描述 + 标签)
        corpus = []
        for p in products:
            text = f"{p['title']} {p['description']} {' '.join(p['tags'])}"
            # 使用jieba进行中文分词
            tokens = list(self.jieba.cut(text.lower()))
            corpus.append(tokens)

        # 构建BM25索引
        bm25 = BM25Okapi(corpus)
        print(f"BM25关键词索引构建完成，共{len(sku_list)}个商品")

        return {
            "sku_list": sku_list,
            "product_map": product_map,
            "bm25": bm25
        }


class HybridIndexBuilder:
    """混合索引构建器 - 统一构建向量和关键词索引"""

    def __init__(self, embedding_model: Optional[str] = None, milvus_config: Optional[Dict[str, Any]] = None):
        # 使用环境变量或传入的参数
        model_name = embedding_model or os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
        self.vector_builder = VectorIndexBuilder(model_name, milvus_config)
        self.keyword_builder = KeywordIndexBuilder()

    def build_all_indices(self) -> Dict[str, Any]:
        """
        构建所有索引

        Returns:
            完整的索引数据字典
        """
        print("开始构建混合索引...")

        # 构建向量索引
        print("1. 构建向量索引...")
        vector_success, vector_data = self.vector_builder.build_vector_index()

        # 构建关键词索引
        print("2. 构建关键词索引...")
        keyword_data = self.keyword_builder.build_keyword_index()

        if not vector_success or not keyword_data:
            raise RuntimeError("索引构建失败")

        print("混合索引构建完成！")

        return {
            "vector_index": vector_data,
            "keyword_index": keyword_data
        }


if __name__ == "__main__":
    # 测试索引构建
    print("开始构建索引...")

    builder = HybridIndexBuilder()
    indices = builder.build_all_indices()

    print("\n索引构建结果:")
    print(f"向量索引类型: {indices['vector_index']['index_type']}")
    print(f"商品数量: {len(indices['vector_index']['sku_list'])}")
    print(f"关键词索引商品数量: {len(indices['keyword_index']['sku_list'])}")