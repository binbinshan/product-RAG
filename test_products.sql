-- 创建商品数据库表
CREATE DATABASE IF NOT EXISTS product_rag;
USE product_rag;
alter schema product_rag collate utf8mb4_general_ci

-- 创建商品表
DROP TABLE IF EXISTS products;
CREATE TABLE products (
    sku_id VARCHAR(50) PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    category VARCHAR(100) NOT NULL,
    age_range VARCHAR(20),
    tags JSON,
    description TEXT,
    status ENUM('ON_SALE', 'OFF_SALE', 'OUT_OF_STOCK') DEFAULT 'ON_SALE',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

-- 插入测试数据
INSERT INTO products (sku_id, title, category, age_range, tags, description, status) VALUES
('SKU_1001', '儿童低乳糖配方奶粉', '奶粉', '3-6', JSON_ARRAY('低乳糖', '益生菌', '儿童'), '低乳糖配方,适合肠胃敏感儿童,添加益生菌促进消化', 'ON_SALE'),
('SKU_2002', '有机全脂儿童奶粉', '奶粉', '3-6', JSON_ARRAY('有机', '全脂', '儿童'), '100%有机奶源,富含DHA,促进大脑发育', 'ON_SALE'),
('SKU_3003', '肠胃友好型配方奶粉', '奶粉', '3-6', JSON_ARRAY('易消化', '益生元', '儿童'), '特别添加益生元,呵护肠胃健康,减少上火', 'ON_SALE'),
('SKU_4004', '成人高钙奶粉', '奶粉', '18+', JSON_ARRAY('高钙', '成人'), '适合成人补钙,添加维生素D', 'ON_SALE'),
('SKU_5005', '无糖益生菌奶粉', '奶粉', '1-3', JSON_ARRAY('无糖', '益生菌', '幼儿'), '0蔗糖添加,10亿活性益生菌,保护幼儿肠道', 'ON_SALE'),
('SKU_6006', '苹果iPhone 14 Pro', '手机', '18+', JSON_ARRAY('苹果', 'iPhone', '5G', '拍照'), '6.1英寸超视网膜XDR显示屏,A16仿生芯片,专业级摄像头系统', 'ON_SALE'),
('SKU_7007', '华为Mate 50 Pro', '手机', '18+', JSON_ARRAY('华为', '5G', '拍照', '续航'), '6.74英寸曲面屏,骁龙8+处理器,XMAGE影像系统,66W快充', 'ON_SALE'),
('SKU_8008', '小米13 Ultra', '手机', '18+', JSON_ARRAY('小米', '5G', '拍照', '性能'), '6.73英寸2K曲面屏,骁龙8 Gen2,徕卡光学镜头,90W快充', 'ON_SALE'),
('SKU_9009', 'Nike Air Max 90', '鞋子', '18+', JSON_ARRAY('Nike', '运动鞋', '休闲'), '经典气垫设计,透气网眼鞋面,舒适缓震,时尚百搭', 'ON_SALE'),
('SKU_10010', 'Adidas Ultraboost 22', '鞋子', '18+', JSON_ARRAY('Adidas', '跑鞋', '运动'), 'Boost中底科技,Primeknit鞋面,极致能量回弹,专业跑步装备', 'ON_SALE');

-- 创建索引以优化查询性能
CREATE INDEX idx_category ON products(category);
CREATE INDEX idx_status ON products(status);
CREATE INDEX idx_age_range ON products(age_range);
CREATE FULLTEXT INDEX ft_title_desc ON products(title, description);