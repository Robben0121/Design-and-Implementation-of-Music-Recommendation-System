"""
config.py - 完整配置文件（含TempChain-ExRec）
"""

from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """系统配置"""
    
    # ==================== LLM配置 ====================
    DEEPSEEK_API_KEY: str = ""
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com/v1"
    DEEPSEEK_MODEL: str = "deepseek-chat"
    
    # ==================== 数据路径配置 ====================
    MUSIC_DATA_PATH: str = "./data/music_data.json"
    USER_PROFILE_DIR: str = "./data/user_profiles"
    VECTOR_DB_PATH: str = "./data/vector_db"
    
    # ==================== 推荐系统参数 ====================
    TOP_K_RETRIEVE: int = 30
    TOP_K_RECOMMEND: int = 10
    
    # 协同过滤参数
    CF_WEIGHT: float = 0.3
    CF_MIN_FEEDBACK: int = 3
    
    # ==================== TempChain-ExRec配置 ====================
    
    # 核心开关
    ENABLE_TEMPCHAIN_EXREC: bool = True  # 是否启用TempChain-ExRec
    ENABLE_EXPLANATION_VALIDATION: bool = True  # 是否启用解释验证
    ENABLE_MULTI_PATH_EXPLANATION: bool = True  # 是否启用多路径生成
    
    # 解释生成参数
    N_EXPLANATION_PATHS: int = 3  # 生成的解释路径数量 (1-6)
    MAX_VALIDATION_ITERATIONS: int = 2  # 最大验证迭代次数
    VALIDATION_THRESHOLD: float = 0.7  # 验证通过阈值 (0-1)
    
    # 默认场景类型
    # 可选: similarity, mood, collaborative, multi_dimension, contextual, narrative
    DEFAULT_SCENARIO_TYPE: str = "multi_dimension"
    
    # 解释详细程度
    # simple: 简洁版（1-2句话）
    # medium: 中等详细（显示4步骤但简化）
    # full: 完整版（包含备选路径和完整分析）
    EXPLANATION_DETAIL_LEVEL: str = "medium"
    
    # LLM温度参数
    EXPLANATION_GENERATION_TEMPERATURE: float = 0.7  # 解释生成温度
    EXPLANATION_VALIDATION_TEMPERATURE: float = 0.3  # 验证时温度（更保守）
    
    # 模板选择策略
    AUTO_SELECT_TEMPLATE: bool = True  # 是否自动选择模板级别
    FORCE_TEMPLATE_LEVEL: Optional[str] = None  # 强制使用的模板级别 (basic/intermediate/advanced)
    
    # 实验模式
    EXPERIMENT_MODE: bool = False  # 实验模式（记录更多指标）
    COLLECT_EXPLANATION_STATS: bool = True  # 收集解释统计信息
    
    # ==================== 系统配置 ====================
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "./logs/app.log"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# 创建全局设置实例
settings = Settings()


# ==================== 配置预设 ====================

class ConfigPresets:
    """配置预设"""
    
    @staticmethod
    def development():
        """开发环境配置"""
        return {
            'ENABLE_EXPLANATION_VALIDATION': False,  # 跳过验证加快速度
            'N_EXPLANATION_PATHS': 2,  # 减少路径数
            'EXPLANATION_DETAIL_LEVEL': 'simple',
            'LOG_LEVEL': 'DEBUG'
        }
    
    @staticmethod
    def production():
        """生产环境配置"""
        return {
            'ENABLE_EXPLANATION_VALIDATION': True,
            'ENABLE_MULTI_PATH_EXPLANATION': True,
            'N_EXPLANATION_PATHS': 3,
            'EXPLANATION_DETAIL_LEVEL': 'medium',
            'LOG_LEVEL': 'INFO'
        }
    
    @staticmethod
    def experiment():
        """实验环境配置"""
        return {
            'ENABLE_EXPLANATION_VALIDATION': True,
            'ENABLE_MULTI_PATH_EXPLANATION': True,
            'N_EXPLANATION_PATHS': 3,
            'EXPERIMENT_MODE': True,
            'COLLECT_EXPLANATION_STATS': True,
            'MAX_VALIDATION_ITERATIONS': 3,  # 更多迭代
            'EXPLANATION_DETAIL_LEVEL': 'full'
        }
    
    @staticmethod
    def fast():
        """快速模式（牺牲质量换取速度）"""
        return {
            'ENABLE_EXPLANATION_VALIDATION': False,
            'ENABLE_MULTI_PATH_EXPLANATION': False,
            'N_EXPLANATION_PATHS': 1,
            'EXPLANATION_DETAIL_LEVEL': 'simple'
        }


def apply_preset(preset_name: str):
    """应用配置预设"""
    presets = {
        'development': ConfigPresets.development(),
        'production': ConfigPresets.production(),
        'experiment': ConfigPresets.experiment(),
        'fast': ConfigPresets.fast()
    }
    
    if preset_name not in presets:
        raise ValueError(f"未知的预设: {preset_name}")
    
    preset = presets[preset_name]
    for key, value in preset.items():
        setattr(settings, key, value)
    
    print(f"✅ 已应用配置预设: {preset_name}")


# ==================== 配置验证 ====================

def validate_config() -> bool:
    """验证配置"""
    issues = []
    
    # 检查API密钥
    if settings.DEEPSEEK_API_KEY == "your-api-key-here":
        issues.append("⚠️  请设置 DEEPSEEK_API_KEY")
    
    # 检查数据路径
    import os
    if not os.path.exists(settings.MUSIC_DATA_PATH):
        issues.append(f"⚠️  音乐数据文件不存在: {settings.MUSIC_DATA_PATH}")
    
    # 检查TempChain-ExRec参数
    if settings.N_EXPLANATION_PATHS < 1 or settings.N_EXPLANATION_PATHS > 6:
        issues.append("⚠️  N_EXPLANATION_PATHS 应在 1-6 之间")
    
    if settings.VALIDATION_THRESHOLD < 0 or settings.VALIDATION_THRESHOLD > 1:
        issues.append("⚠️  VALIDATION_THRESHOLD 应在 0-1 之间")
    
    if issues:
        print("\n❌ 配置检查发现问题：")
        for issue in issues:
            print(f"   {issue}")
        print("\n请检查配置文件或环境变量\n")
        return False
    else:
        print("✅ 配置检查通过")
        return True


def print_config_summary():
    """打印配置摘要"""
    print("\n" + "="*60)
    print("📋 TempChain-ExRec 配置摘要")
    print("="*60)
    print(f"LLM模型: {settings.DEEPSEEK_MODEL}")
    print(f"TempChain-ExRec: {'✅ 启用' if settings.ENABLE_TEMPCHAIN_EXREC else '❌ 禁用'}")
    
    if settings.ENABLE_TEMPCHAIN_EXREC:
        print(f"  - 多路径生成: {'✅ 启用' if settings.ENABLE_MULTI_PATH_EXPLANATION else '❌ 禁用'}")
        print(f"  - 路径数量: {settings.N_EXPLANATION_PATHS}")
        print(f"  - 解释验证: {'✅ 启用' if settings.ENABLE_EXPLANATION_VALIDATION else '❌ 禁用'}")
        print(f"  - 详细程度: {settings.EXPLANATION_DETAIL_LEVEL}")
        print(f"  - 实验模式: {'✅ 启用' if settings.EXPERIMENT_MODE else '❌ 禁用'}")
    
    print(f"推荐数量: Top-{settings.TOP_K_RECOMMEND}")
    print(f"协同过滤: 权重={settings.CF_WEIGHT}")
    print("="*60 + "\n")


if __name__ == "__main__":
    print("🔧 配置管理工具\n")
    
    # 验证配置
    validate_config()
    
    # 打印摘要
    print_config_summary()
    
    # 示例：应用预设
    print("可用的配置预设：")
    print("  - development: 开发环境（快速、简化）")
    print("  - production: 生产环境（平衡）")
    print("  - experiment: 实验环境（完整功能）")
    print("  - fast: 快速模式（最小化）")
    print("\n使用方法：")
    print("  from config import apply_preset")
    print("  apply_preset('experiment')")
