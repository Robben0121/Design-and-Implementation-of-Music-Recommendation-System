import json
from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from loguru import logger

class MusicItem(BaseModel):
    """音乐数据模型"""
    song_id: str
    song: str
    artist: str
    album: str
    genre: str
    year: int
    release_date: str
    album_type: str
    media: str
    description: str
    tags: List[str]
    semantic_summary: str
    
    def to_text_for_embedding(self) -> str:
        """转换为用于向量化的文本"""
        # 组合多个字段，用于生成更丰富的语义向量
        tags_str = ", ".join(self.tags)
        text = f"""
歌曲：{self.song}
歌手：{self.artist}
专辑：{self.album}
风格：{self.genre}
年代：{self.year}
标签：{tags_str}
语义摘要：{self.semantic_summary}
描述：{self.description}
        """.strip()
        return text
    
    def to_display_format(self) -> Dict:
        """转换为前端展示格式"""
        return {
            "歌曲名": self.song,
            "歌手": self.artist,
            "专辑": self.album,
            "风格": self.genre,
            "年份": self.year,
            "标签": ", ".join(self.tags),
            "推荐理由": self.semantic_summary
        }


class MusicDatabase:
    """音乐数据库管理器"""
    
    def __init__(self, json_path: str):
        self.json_path = json_path
        self.music_items: List[MusicItem] = []
        self.load_data()
    
    def load_data(self):
        """从JSON文件加载数据"""
        logger.info(f"正在加载音乐数据库: {self.json_path}")
        try:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 转换为 MusicItem 对象
            self.music_items = [MusicItem(**item) for item in data]
            logger.success(f"成功加载 {len(self.music_items)} 首歌曲")
            
        except FileNotFoundError:
            logger.error(f"文件不存在: {self.json_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"JSON解析错误: {e}")
            raise
    
    def get_by_id(self, song_id: str) -> Optional[MusicItem]:
        """根据ID获取歌曲"""
        for item in self.music_items:
            if item.song_id == song_id:
                return item
        return None
    
    def filter_by_genre(self, genre: str) -> List[MusicItem]:
        """根据流派筛选"""
        return [item for item in self.music_items if genre.lower() in item.genre.lower()]
    
    def filter_by_year(self, start_year: int, end_year: int) -> List[MusicItem]:
        """根据年代筛选"""
        return [item for item in self.music_items 
                if start_year <= item.year <= end_year]
    
    def filter_by_tags(self, tags: List[str]) -> List[MusicItem]:
        """根据标签筛选（至少匹配一个标签）"""
        return [item for item in self.music_items 
                if any(tag in item.tags for tag in tags)]
    
    def get_all_genres(self) -> List[str]:
        """获取所有流派"""
        genres = set(item.genre for item in self.music_items)
        return sorted(list(genres))
    
    def get_all_artists(self) -> List[str]:
        """获取所有歌手"""
        artists = set(item.artist for item in self.music_items)
        return sorted(list(artists))
    
    def get_statistics(self) -> Dict:
        """获取数据库统计信息"""
        return {
            "total_songs": len(self.music_items),
            "total_artists": len(self.get_all_artists()),
            "total_genres": len(self.get_all_genres()),
            "year_range": (
                min(item.year for item in self.music_items),
                max(item.year for item in self.music_items)
            )
        }
