"""
collaborative_engine.py - 协同过滤推荐引擎

实现功能：
1. User-based CF: 基于用户相似度的协同过滤
2. Item-based CF: 基于物品相似度的协同过滤  
3. 混合推荐: 结合内容推荐与协同过滤

核心思路：
- 利用 user_profile.py 中已有的 liked_songs/disliked_songs 构建用户-物品交互矩阵
- 计算用户/物品相似度，生成协同过滤推荐
- 与向量检索结果融合，提供混合推荐
"""

import os
import json
import numpy as np
from typing import List, Dict, Optional, Tuple, Set
from collections import defaultdict
from loguru import logger
from datetime import datetime

from data_loader import MusicDatabase, MusicItem
from user_profile import UserProfile, UserManager


class UserItemMatrix:
    """用户-物品交互矩阵管理器"""
    
    def __init__(self, user_manager: UserManager, music_db: MusicDatabase):
        self.user_manager = user_manager
        self.music_db = music_db
        
        # 用户ID -> 索引映射
        self.user_id_to_idx: Dict[str, int] = {}
        self.idx_to_user_id: Dict[int, str] = {}
        
        # 歌曲ID -> 索引映射
        self.song_id_to_idx: Dict[str, int] = {}
        self.idx_to_song_id: Dict[int, str] = {}
        
        # 交互矩阵: 1=喜欢, -1=不喜欢, 0=未交互
        self.matrix: Optional[np.ndarray] = None
        
        # 缓存路径
        self.cache_dir = "./data/cf_cache"
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def build_matrix(self) -> np.ndarray:
        """构建用户-物品交互矩阵"""
        logger.info("正在构建用户-物品交互矩阵...")
        
        # 获取所有用户
        all_users = self.user_manager.list_users()
        if not all_users:
            logger.warning("没有找到任何用户数据")
            return np.array([])
        
        # 建立映射
        for idx, user_info in enumerate(all_users):
            user_id = user_info['user_id']
            self.user_id_to_idx[user_id] = idx
            self.idx_to_user_id[idx] = user_id
        
        for idx, song in enumerate(self.music_db.music_items):
            self.song_id_to_idx[song.song_id] = idx
            self.idx_to_song_id[idx] = song.song_id
        
        n_users = len(all_users)
        n_songs = len(self.music_db.music_items)
        
        # 初始化矩阵
        self.matrix = np.zeros((n_users, n_songs), dtype=np.float32)
        
        # 填充交互数据
        for user_info in all_users:
            user_id = user_info['user_id']
            profile = self.user_manager.get_user_profile(user_id)
            
            if profile:
                user_idx = self.user_id_to_idx[user_id]
                
                # 喜欢的歌曲
                for song_id in profile.liked_songs:
                    if song_id in self.song_id_to_idx:
                        song_idx = self.song_id_to_idx[song_id]
                        self.matrix[user_idx, song_idx] = 1.0
                
                # 不喜欢的歌曲
                for song_id in profile.disliked_songs:
                    if song_id in self.song_id_to_idx:
                        song_idx = self.song_id_to_idx[song_id]
                        self.matrix[user_idx, song_idx] = -1.0
        
        # 统计信息
        n_interactions = np.count_nonzero(self.matrix)
        sparsity = 1 - (n_interactions / (n_users * n_songs))
        
        logger.success(f"交互矩阵构建完成: {n_users}用户 x {n_songs}歌曲, "
                      f"交互数:{n_interactions}, 稀疏度:{sparsity:.4f}")
        
        return self.matrix
    
    def get_user_vector(self, user_id: str) -> Optional[np.ndarray]:
        """获取用户的交互向量"""
        if self.matrix is None:
            self.build_matrix()
        
        if user_id in self.user_id_to_idx:
            idx = self.user_id_to_idx[user_id]
            return self.matrix[idx]
        
        # 新用户：从 profile 构建临时向量
        profile = self.user_manager.get_user_profile(user_id)
        if profile:
            vec = np.zeros(len(self.song_id_to_idx), dtype=np.float32)
            for song_id in profile.liked_songs:
                if song_id in self.song_id_to_idx:
                    vec[self.song_id_to_idx[song_id]] = 1.0
            for song_id in profile.disliked_songs:
                if song_id in self.song_id_to_idx:
                    vec[self.song_id_to_idx[song_id]] = -1.0
            return vec
        
        return None
    
    def get_song_vector(self, song_id: str) -> Optional[np.ndarray]:
        """获取歌曲的用户交互向量"""
        if self.matrix is None:
            self.build_matrix()
        
        if song_id in self.song_id_to_idx:
            idx = self.song_id_to_idx[song_id]
            return self.matrix[:, idx]
        return None


class CollaborativeFilterEngine:
    """协同过滤推荐引擎"""
    
    def __init__(self, user_manager: UserManager, music_db: MusicDatabase):
        self.user_manager = user_manager
        self.music_db = music_db
        self.matrix_manager = UserItemMatrix(user_manager, music_db)
        
        # 相似度缓存
        self.user_similarity_cache: Dict[str, Dict[str, float]] = {}
        self.item_similarity_cache: Dict[str, Dict[str, float]] = {}
        
        logger.info("协同过滤引擎初始化完成")
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """计算余弦相似度"""
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(np.dot(vec1, vec2) / (norm1 * norm2))
    
    def _pearson_correlation(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """计算皮尔逊相关系数"""
        # 只考虑两者都有交互的项
        mask = (vec1 != 0) & (vec2 != 0)
        if mask.sum() < 2:  # 至少需要2个共同交互
            return 0.0
        
        v1 = vec1[mask]
        v2 = vec2[mask]
        
        mean1, mean2 = v1.mean(), v2.mean()
        v1_centered = v1 - mean1
        v2_centered = v2 - mean2
        
        numerator = np.dot(v1_centered, v2_centered)
        denominator = np.sqrt(np.sum(v1_centered**2) * np.sum(v2_centered**2))
        
        if denominator == 0:
            return 0.0
        
        return float(numerator / denominator)
    
    # ==================== User-based CF ====================
    
    def find_similar_users(self, user_id: str, top_k: int = 20,
                          min_common_items: int = 0) -> List[Tuple[str, float]]:
        """
        找到与目标用户最相似的用户

        Args:
            user_id: 目标用户ID
            top_k: 返回前k个相似用户
            min_common_items: 最少共同交互项数（优化：从1降到0以应对数据稀疏）
        
        Returns:
            [(user_id, similarity_score), ...]
        """
        user_vec = self.matrix_manager.get_user_vector(user_id)
        if user_vec is None or np.count_nonzero(user_vec) == 0:
            return []
        
        similarities = []
        
        for other_id in self.matrix_manager.user_id_to_idx.keys():
            if other_id == user_id:
                continue
            
            other_vec = self.matrix_manager.get_user_vector(other_id)
            if other_vec is None:
                continue
            
            # 检查共同交互项数（两者都有交互的位置）
            common_mask = (user_vec != 0) & (other_vec != 0)
            n_common = common_mask.sum()
            
            if n_common < min_common_items:
                continue
            
            # 使用余弦相似度（更适合隐式反馈数据）
            sim = self._cosine_similarity(user_vec, other_vec)

            if sim > 0.001:  # 优化：降低阈值从0.01到0.001，以应对数据稀疏
                similarities.append((other_id, sim))
        
        # 按相似度排序
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def user_based_recommend(self, user_id: str, top_k: int = 20,
                            n_neighbors: int = 10) -> List[Dict]:
        """
        基于用户的协同过滤推荐
        
        Args:
            user_id: 目标用户ID
            top_k: 推荐歌曲数
            n_neighbors: 使用的相似用户数
        
        Returns:
            [{'item': MusicItem, 'score': float, 'source': 'user_cf'}, ...]
        """
        logger.info(f"[User-CF] 为用户 {user_id} 生成推荐...")
        
        # 获取用户已交互的歌曲
        profile = self.user_manager.get_user_profile(user_id)
        if not profile:
            return []
        
        interacted_songs = profile.liked_songs | profile.disliked_songs
        
        # 找到相似用户
        similar_users = self.find_similar_users(user_id, top_k=n_neighbors)
        if not similar_users:
            logger.info("[User-CF] 未找到相似用户，无法生成推荐")
            return []
        
        # 收集相似用户喜欢的歌曲
        song_scores: Dict[str, float] = defaultdict(float)
        song_weights: Dict[str, float] = defaultdict(float)
        
        for neighbor_id, similarity in similar_users:
            neighbor_profile = self.user_manager.get_user_profile(neighbor_id)
            if not neighbor_profile:
                continue
            
            # 相似用户喜欢的歌曲
            for song_id in neighbor_profile.liked_songs:
                if song_id not in interacted_songs:  # 排除已交互的
                    song_scores[song_id] += similarity * 1.0
                    song_weights[song_id] += similarity
        
        # 计算加权平均分
        recommendations = []
        for song_id, score in song_scores.items():
            if song_weights[song_id] > 0:
                weighted_score = score / song_weights[song_id]
                item = self.music_db.get_by_id(song_id)
                if item:
                    recommendations.append({
                        'item': item,
                        'score': weighted_score,
                        'source': 'user_cf',
                        'reason': f"与您品味相似的用户也喜欢"
                    })
        
        # 按分数排序
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        
        logger.info(f"[User-CF] 生成 {len(recommendations[:top_k])} 首推荐")
        return recommendations[:top_k]
    
    # ==================== Item-based CF ====================
    
    def compute_item_similarity(self, song_id1: str, song_id2: str) -> float:
        """计算两首歌曲的相似度（基于用户行为）"""
        vec1 = self.matrix_manager.get_song_vector(song_id1)
        vec2 = self.matrix_manager.get_song_vector(song_id2)
        
        if vec1 is None or vec2 is None:
            return 0.0
        
        return self._cosine_similarity(vec1, vec2)
    
    def find_similar_songs(self, song_id: str, top_k: int = 20,
                          min_common_users: int = 1) -> List[Tuple[str, float]]:
        """
        找到与目标歌曲最相似的歌曲（基于协同过滤）
        
        Args:
            song_id: 目标歌曲ID
            top_k: 返回前k首相似歌曲
            min_common_users: 最少共同交互用户数
        
        Returns:
            [(song_id, similarity_score), ...]
        """
        song_vec = self.matrix_manager.get_song_vector(song_id)
        if song_vec is None or song_vec.sum() == 0:
            return []
        
        similarities = []
        
        for other_id in self.matrix_manager.song_id_to_idx.keys():
            if other_id == song_id:
                continue
            
            other_vec = self.matrix_manager.get_song_vector(other_id)
            if other_vec is None:
                continue
            
            # 检查共同交互用户数
            common_mask = (song_vec != 0) & (other_vec != 0)
            if common_mask.sum() < min_common_users:
                continue
            
            sim = self._cosine_similarity(song_vec, other_vec)
            if sim > 0:
                similarities.append((other_id, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def item_based_recommend(self, user_id: str, top_k: int = 20) -> List[Dict]:
        """
        基于物品的协同过滤推荐
        
        Args:
            user_id: 目标用户ID
            top_k: 推荐歌曲数
        
        Returns:
            [{'item': MusicItem, 'score': float, 'source': 'item_cf'}, ...]
        """
        logger.info(f"[Item-CF] 为用户 {user_id} 生成推荐...")
        
        profile = self.user_manager.get_user_profile(user_id)
        if not profile or not profile.liked_songs:
            return []
        
        interacted_songs = profile.liked_songs | profile.disliked_songs
        
        # 基于用户喜欢的歌曲，找相似歌曲
        song_scores: Dict[str, float] = defaultdict(float)
        
        for liked_song_id in profile.liked_songs:
            similar_songs = self.find_similar_songs(liked_song_id, top_k=20)
            
            for similar_id, similarity in similar_songs:
                if similar_id not in interacted_songs:
                    song_scores[similar_id] += similarity
        
        # 构建推荐列表
        recommendations = []
        for song_id, score in song_scores.items():
            item = self.music_db.get_by_id(song_id)
            if item:
                recommendations.append({
                    'item': item,
                    'score': score,
                    'source': 'item_cf',
                    'reason': f"与您喜欢的歌曲风格相似"
                })
        
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        
        logger.info(f"[Item-CF] 生成 {len(recommendations[:top_k])} 首推荐")
        return recommendations[:top_k]
    
    # ==================== 混合推荐 ====================
    
    def hybrid_recommend(self, user_id: str, 
                        content_results: List[Dict],
                        top_k: int = 10,
                        cf_weight: float = 0.3) -> List[Dict]:
        """
        混合推荐：结合内容推荐与协同过滤
        
        Args:
            user_id: 用户ID
            content_results: 内容推荐结果 [{'item': MusicItem, 'score': float}, ...]
            top_k: 最终返回数量
            cf_weight: 协同过滤权重 (0-1)
        
        Returns:
            融合后的推荐列表
        """
        logger.info(f"[Hybrid] 生成混合推荐，CF权重={cf_weight}")
        
        # 获取协同过滤推荐
        user_cf_results = self.user_based_recommend(user_id, top_k=top_k)
        item_cf_results = self.item_based_recommend(user_id, top_k=top_k)
        
        # 合并协同过滤结果
        cf_scores: Dict[str, float] = {}
        cf_reasons: Dict[str, str] = {}
        
        for r in user_cf_results:
            song_id = r['item'].song_id
            cf_scores[song_id] = r['score']
            cf_reasons[song_id] = r.get('reason', '')
        
        for r in item_cf_results:
            song_id = r['item'].song_id
            if song_id in cf_scores:
                cf_scores[song_id] = (cf_scores[song_id] + r['score']) / 2
            else:
                cf_scores[song_id] = r['score']
                cf_reasons[song_id] = r.get('reason', '')
        
        # 融合分数
        final_scores: Dict[str, Dict] = {}
        content_weight = 1 - cf_weight
        
        # 处理内容推荐结果
        for r in content_results:
            song_id = r['item'].song_id
            content_score = r.get('score', 0.5)
            cf_score = cf_scores.get(song_id, 0)
            
            # 加权融合
            if cf_score > 0:
                final_score = content_weight * content_score + cf_weight * cf_score
                source = 'hybrid'
                reason = cf_reasons.get(song_id, '')
            else:
                final_score = content_score
                source = 'content'
                reason = ''
            
            final_scores[song_id] = {
                'item': r['item'],
                'score': final_score,
                'source': source,
                'cf_boost': cf_score > 0,
                'reason': reason
            }
        
        # 添加纯协同过滤的结果（内容推荐未覆盖的）
        for song_id, cf_score in cf_scores.items():
            if song_id not in final_scores:
                item = self.music_db.get_by_id(song_id)
                if item:
                    final_scores[song_id] = {
                        'item': item,
                        'score': cf_weight * cf_score,
                        'source': 'cf_only',
                        'cf_boost': True,
                        'reason': cf_reasons.get(song_id, '')
                    }
        
        # 排序并返回
        results = list(final_scores.values())
        results.sort(key=lambda x: x['score'], reverse=True)
        
        logger.info(f"[Hybrid] 最终推荐 {len(results[:top_k])} 首歌曲")
        return results[:top_k]
    
    def get_cf_statistics(self) -> Dict:
        """获取协同过滤统计信息"""
        if self.matrix_manager.matrix is None:
            self.matrix_manager.build_matrix()
        
        matrix = self.matrix_manager.matrix
        n_users = len(self.matrix_manager.user_id_to_idx)
        n_songs = len(self.matrix_manager.song_id_to_idx)
        
        if matrix is None or matrix.size == 0:
            return {
                'n_users': 0,
                'n_songs': 0,
                'n_interactions': 0,
                'sparsity': 1.0,
                'avg_ratings_per_user': 0,
                'avg_ratings_per_song': 0
            }
        
        n_interactions = np.count_nonzero(matrix)
        
        return {
            'n_users': n_users,
            'n_songs': n_songs,
            'n_interactions': n_interactions,
            'n_likes': int((matrix == 1).sum()),
            'n_dislikes': int((matrix == -1).sum()),
            'sparsity': 1 - (n_interactions / (n_users * n_songs)) if n_users * n_songs > 0 else 1.0,
            'avg_ratings_per_user': n_interactions / n_users if n_users > 0 else 0,
            'avg_ratings_per_song': n_interactions / n_songs if n_songs > 0 else 0
        }
    
    def refresh_matrix(self):
        """刷新交互矩阵（当有新的用户反馈时调用）"""
        self.matrix_manager.build_matrix()
        self.user_similarity_cache.clear()
        self.item_similarity_cache.clear()
        logger.info("协同过滤矩阵已刷新")


# ==================== 测试代码 ====================

if __name__ == "__main__":
    from config import settings
    
    # 初始化
    music_db = MusicDatabase(settings.MUSIC_DATA_PATH)
    user_manager = UserManager(settings.USER_PROFILE_DIR if hasattr(settings, 'USER_PROFILE_DIR') else "./data/user_profiles")
    
    cf_engine = CollaborativeFilterEngine(user_manager, music_db)
    
    # 统计信息
    stats = cf_engine.get_cf_statistics()
    print(f"协同过滤统计: {stats}")
    
    # 测试推荐（需要有用户数据）
    users = user_manager.list_users()
    if users:
        test_user_id = users[0]['user_id']
        print(f"\n为用户 {test_user_id} 生成推荐:")
        
        # User-based CF
        user_cf_results = cf_engine.user_based_recommend(test_user_id, top_k=5)
        print(f"\nUser-CF 推荐:")
        for r in user_cf_results:
            print(f"  - {r['item'].song} ({r['item'].artist}), score={r['score']:.3f}")
        
        # Item-based CF
        item_cf_results = cf_engine.item_based_recommend(test_user_id, top_k=5)
        print(f"\nItem-CF 推荐:")
        for r in item_cf_results:
            print(f"  - {r['item'].song} ({r['item'].artist}), score={r['score']:.3f}")