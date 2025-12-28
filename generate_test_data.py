"""
随机商品数据生成器
基于index_builder.py中的数据库结构生成测试数据
"""
import random
import json
from typing import List, Dict, Any


class RandomProductGenerator:
    """随机商品数据生成器"""

    def __init__(self):
        # 用于自增生成唯一SKU ID
        self.sku_counter = 1000000
        # 商品类别
        self.categories = [
            "玩具", "图书", "文具", "体育用品", "电子产品",
            "服装", "家居用品", "美食", "艺术用品", "户外用品"
        ]

        # 年龄范围
        self.age_ranges = [
            "0-3岁", "3-6岁", "6-12岁", "12-18岁", "18+岁"
        ]

        # 商品状态
        self.statuses = ["ON_SALE", "OFF_SALE", "OUT_OF_STOCK"]

        # 标签库
        self.tag_pools = {
            "玩具": ["益智", "拼装", "电动", "毛绒", "积木", "模型", "遥控", "早教"],
            "图书": ["绘本", "科普", "故事", "教育", "漫画", "文学", "工具书", "百科"],
            "文具": ["书写", "绘画", "收纳", "计算", "装订", "测量", "创意", "环保"],
            "体育用品": ["球类", "健身", "游泳", "跑步", "户外", "保护", "训练", "竞技"],
            "电子产品": ["智能", "教育", "娱乐", "通讯", "音响", "摄影", "游戏", "学习"],
            "服装": ["休闲", "运动", "正装", "户外", "内衣", "配饰", "季节", "时尚"],
            "家居用品": ["收纳", "装饰", "清洁", "厨房", "卧室", "客厅", "浴室", "智能"],
            "美食": ["零食", "饮料", "健康", "有机", "进口", "传统", "新奇", "营养"],
            "艺术用品": ["绘画", "手工", "雕塑", "音乐", "舞蹈", "创作", "装饰", "收藏"],
            "户外用品": ["野营", "徒步", "钓鱼", "登山", "骑行", "防护", "导航", "生存"]
        }

        # 商品名称前缀
        self.name_prefixes = {
            "玩具": ["超级", "迷你", "智能", "经典", "创意", "益智", "趣味", "神奇"],
            "图书": ["精美", "经典", "畅销", "获奖", "原创", "精装", "彩绘", "立体"],
            "文具": ["高级", "环保", "创意", "实用", "便携", "多功能", "专业", "时尚"],
            "体育用品": ["专业", "高级", "轻量", "耐用", "防水", "透气", "舒适", "竞技"],
            "电子产品": ["智能", "高清", "便携", "无线", "多功能", "触控", "语音", "AI"],
            "服装": ["舒适", "时尚", "经典", "休闲", "运动", "透气", "保暖", "防水"],
            "家居用品": ["简约", "实用", "美观", "环保", "多功能", "节省", "智能", "舒适"],
            "美食": ["美味", "健康", "营养", "新鲜", "有机", "无添加", "传统", "创新"],
            "艺术用品": ["专业", "高级", "创意", "精美", "实用", "环保", "多彩", "便携"],
            "户外用品": ["专业", "轻量", "防水", "耐用", "便携", "多功能", "安全", "舒适"]
        }

        # 商品名称主体
        self.name_bodies = {
            "玩具": ["积木", "拼图", "机器人", "汽车", "飞机", "娃娃", "球", "枪", "剑", "城堡"],
            "图书": ["故事书", "绘本", "百科全书", "漫画", "小说", "诗集", "词典", "教材", "杂志", "画册"],
            "文具": ["钢笔", "铅笔", "橡皮", "尺子", "计算器", "文件夹", "笔记本", "胶水", "剪刀", "订书机"],
            "体育用品": ["篮球", "足球", "网球", "跑鞋", "运动服", "护具", "球拍", "哑铃", "瑜伽垫", "游泳镜"],
            "电子产品": ["手机", "平板", "耳机", "音响", "摄像头", "键盘", "鼠标", "充电器", "数据线", "移动电源"],
            "服装": ["T恤", "衬衫", "裤子", "裙子", "外套", "鞋子", "帽子", "围巾", "手套", "袜子"],
            "家居用品": ["台灯", "花瓶", "抱枕", "地毯", "窗帘", "收纳盒", "衣架", "镜子", "时钟", "相框"],
            "美食": ["巧克力", "饼干", "果汁", "坚果", "糖果", "蛋糕", "面包", "酸奶", "水果", "蔬菜"],
            "艺术用品": ["画笔", "颜料", "画布", "雕刻刀", "陶土", "乐器", "舞鞋", "画板", "调色盘", "素描本"],
            "户外用品": ["帐篷", "睡袋", "登山包", "头灯", "指南针", "水壶", "登山鞋", "防晒霜", "急救包", "望远镜"]
        }

    def generate_product_name(self, category: str) -> str:
        """生成商品名称"""
        prefix = random.choice(self.name_prefixes[category])
        body = random.choice(self.name_bodies[category])
        return f"{prefix}{body}"

    def generate_description(self, category: str, name: str) -> str:
        """生成商品描述"""
        descriptions = {
            "玩具": f"这款{name}采用优质材料制作，安全无毒，适合儿童使用。具有教育意义，能够培养孩子的动手能力和想象力。",
            "图书": f"这本{name}内容丰富，插图精美，语言生动有趣。适合亲子阅读，能够拓展知识面，培养阅读兴趣。",
            "文具": f"这款{name}设计精美，功能实用。采用环保材料，使用舒适，是学习和办公的好帮手。",
            "体育用品": f"这款{name}采用专业设计，质量可靠。适合运动训练使用，能够提升运动表现，保护运动安全。",
            "电子产品": f"这款{name}功能强大，操作简便。采用先进技术，性能稳定，为用户提供优质的使用体验。",
            "服装": f"这款{name}面料舒适，版型时尚。做工精细，穿着舒适，适合日常穿搭和多种场合。",
            "家居用品": f"这款{name}设计美观，实用性强。能够提升家居品味，为生活增添便利和舒适。",
            "美食": f"这款{name}口感丰富，营养均衡。选用优质原料，制作工艺精良，是健康美味的选择。",
            "艺术用品": f"这款{name}质量优良，使用方便。适合艺术创作和学习使用，能够激发创意灵感。",
            "户外用品": f"这款{name}设计专业，品质可靠。适合户外活动使用，能够保障安全，提升户外体验。"
        }
        return descriptions.get(category, f"优质的{name}，品质可靠，使用方便。")

    def generate_sku_id(self) -> str:
        """生成自增的唯一SKU ID"""
        self.sku_counter += 1
        return f"SKU{self.sku_counter}"

    def generate_single_product(self) -> Dict[str, Any]:
        """生成单个商品数据"""
        category = random.choice(self.categories)
        name = self.generate_product_name(category)
        description = self.generate_description(category, name)
        age_range = random.choice(self.age_ranges)
        status = random.choice(self.statuses)

        # 生成标签（2-4个）
        available_tags = self.tag_pools.get(category, ["通用", "实用", "优质"])
        num_tags = random.randint(2, min(4, len(available_tags)))
        tags = random.sample(available_tags, num_tags)

        return {
            "sku_id": self.generate_sku_id(),
            "title": name,
            "category": category,
            "age_range": age_range,
            "tags": json.dumps(tags, ensure_ascii=False),
            "description": description,
            "status": status
        }

    def generate_sql_batch(self, count: int = 100) -> str:
        """
        生成批量插入商品的SQL语句

        Args:
            count: 生成商品数量

        Returns:
            完整的SQL INSERT语句
        """
        products = []

        for _ in range(count):
            product = self.generate_single_product()
            products.append(product)

        # 生成SQL
        sql_parts = ["INSERT INTO products (sku_id, title, category, age_range, tags, description, status) VALUES"]

        value_parts = []
        for product in products:
            value_part = f"('{product['sku_id']}', '{product['title']}', '{product['category']}', " \
                        f"'{product['age_range']}', '{product['tags']}', '{product['description']}', '{product['status']}')"
            value_parts.append(value_part)

        sql_parts.append(",\n".join(value_parts))
        sql_parts.append(";")

        return "\n".join(sql_parts)

    def generate_sql_file(self, count: int = 100, filename: str = "test_products.sql") -> str:
        """
        生成SQL文件

        Args:
            count: 生成商品数量
            filename: 文件名

        Returns:
            生成的文件路径
        """
        sql_content = self.generate_sql_batch(count)

        with open(filename, 'w', encoding='utf-8') as f:
            f.write(sql_content)

        print(f"已生成{count}个随机商品的SQL文件: {filename}")
        return filename


def generate_random_products_sql(count: int = 100) -> str:
    """
    快速生成随机商品SQL的函数

    Args:
        count: 生成商品数量，默认100

    Returns:
        SQL语句字符串
    """
    generator = RandomProductGenerator()
    return generator.generate_sql_batch(count)


if __name__ == "__main__":
    # 测试生成
    generator = RandomProductGenerator()

    # 生成10个商品的SQL用于测试
    print("=== 生成10个测试商品SQL ===")
    sql = generator.generate_sql_file(10000, 'my_products.sql')
    print(sql)

    print("\n" + "="*50)
    print("如果需要生成更多商品，可以调用:")
    print("generator.generate_sql_batch(100)  # 生成100个商品")
    print("generator.generate_sql_file(500, 'my_products.sql')  # 生成500个商品到文件")