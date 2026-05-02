import chromadb
from chromadb.config import Settings as ChromaSettings
from typing import List, Dict, Optional
from loguru import logger
import os

from data_loader import MusicDatabase, MusicItem
from config import settings


class VectorSearchEngine:
    """向量检索引擎"""
    
    def __init__(self, music_db: MusicDatabase, db_path: str = None):
        self.music_db = music_db
        self.db_path = db_path or settings.VECTOR_DB_PATH
        
        # 初始化 ChromaDB
        os.makedirs(self.db_path, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=self.db_path,
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        
        # 获取或创建集合
        self.collection = self.client.get_or_create_collection(
            name="music_collection",
            metadata={"description": "音乐推荐系统向量数据库"}
        )
        
        logger.info(f"向量数据库初始化完成，当前包含 {self.collection.count()} 条记录")
    
    def build_index(self, force_rebuild: bool = False):
        """构建向量索引"""
        if self.collection.count() > 0 and not force_rebuild:
            logger.info("向量索引已存在，跳过构建")
            return
        
        logger.info("开始构建向量索引...")
        
        # 准备数据
        ids = []
        documents = []
        metadatas = []
        
        for item in self.music_db.music_items:
            ids.append(item.song_id)
            documents.append(item.to_text_for_embedding())
            metadatas.append({
                "song": item.song,
                "artist": item.artist,
                "album": item.album,
                "genre": item.genre,
                "year": str(item.year),
                "tags": ",".join(item.tags)
            })
        
        # 分批添加（避免一次性添加过多）
        batch_size = 100
        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i:i+batch_size]
            batch_docs = documents[i:i+batch_size]
            batch_metas = metadatas[i:i+batch_size]
            
            self.collection.add(
                ids=batch_ids,
                documents=batch_docs,
                metadatas=batch_metas
            )
            
            if (i + batch_size) % 1000 == 0:
                logger.info(f"已添加 {i+batch_size}/{len(ids)} 条记录")
        
        logger.success(f"向量索引构建完成！共 {self.collection.count()} 条记录")
    
    def search(self, 
               query: str, 
               top_k: int = 20,
               genre_filter: Optional[str] = None,
               year_range: Optional[tuple] = None,
               tags_filter: Optional[List[str]] = None) -> List[Dict]:
        """
        向量检索
        
        Args:
            query: 用户查询
            top_k: 返回top-k个结果
            genre_filter: 流派过滤
            year_range: 年代范围 (start_year, end_year)
            tags_filter: 标签过滤
        """
        logger.info(f"检索查询: {query}")
        
        # 构建where条件
        where_conditions = {}
        if genre_filter:
            where_conditions["genre"] = {"$eq": genre_filter}
        
        # 执行检索
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k * 2,  # 多取一些，后续再过滤
            where=where_conditions if where_conditions else None
        )
        
        # 解析结果
        retrieved_items = []
        if results['ids'] and len(results['ids'][0]) > 0:
            for song_id, distance, metadata in zip(
                results['ids'][0],
                results['distances'][0],
                results['metadatas'][0]
            ):
                # 从数据库获取完整信息
                item = self.music_db.get_by_id(song_id)
                if item:
                    # 应用额外过滤
                    if year_range and not (year_range[0] <= item.year <= year_range[1]):
                        continue
                    if tags_filter and not any(tag in item.tags for tag in tags_filter):
                        continue
                    
                    retrieved_items.append({
                        "item": item,
                        "score": 1 - distance,  # 转换为相似度分数
                        "distance": distance
                    })
        
        # 按相似度排序并返回top_k
        retrieved_items.sort(key=lambda x: x['score'], reverse=True)
        retrieved_items = retrieved_items[:top_k]
        
        logger.info(f"检索到 {len(retrieved_items)} 首歌曲")
        return retrieved_items
    
    def search_similar_songs(self, song_id: str, top_k: int = 10) -> List[Dict]:
        """找相似歌曲"""
        item = self.music_db.get_by_id(song_id)
        if not item:
            return []
        
        # 使用歌曲的语义摘要作为查询
        query = item.semantic_summary
        results = self.search(query, top_k=top_k+1)  # +1因为会包含自己
        
        # 移除自己
        results = [r for r in results if r['item'].song_id != song_id]
        return results[:top_k]
    
    def get_random_songs(self, n: int = 10) -> List[MusicItem]:
        """随机获取歌曲（用于冷启动）"""
        import random
        return random.sample(self.music_db.music_items, min(n, len(self.music_db.music_items)))
