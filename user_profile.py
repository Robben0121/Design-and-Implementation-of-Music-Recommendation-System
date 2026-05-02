"""
user_profile.py - 多用户画像管理模块（改进版）

新增功能：
1. 多用户隔离 - 每个用户独立的画像存储
2. 用户认证 - 登录/注册/游客模式
3. 会话管理 - 区分会话偏好和持久偏好
4. 用户管理器 - 统一管理所有用户
"""

import json
import os
import hashlib
import uuid
from typing import List, Dict, Optional, Set
from datetime import datetime
from pydantic import BaseModel, Field
from loguru import logger


# ==================== 数据模型（保持原有结构） ====================

class UserPreferences(BaseModel):
    """用户偏好模型"""
    favorite_genres: List[str] = Field(default_factory=list)
    favorite_artists: List[str] = Field(default_factory=list)
    favorite_tags: List[str] = Field(default_factory=list)
    dislike_genres: List[str] = Field(default_factory=list)
    dislike_artists: List[str] = Field(default_factory=list)
    
    preferred_era: Optional[str] = None
    year_range: Optional[tuple] = None
    
    moods: List[str] = Field(default_factory=list)
    listening_contexts: List[str] = Field(default_factory=list)
    
    tempo_preference: Optional[str] = None
    vocal_preference: Optional[str] = None
    language_preferences: List[str] = Field(default_factory=list)


class ConversationHistory(BaseModel):
    """对话历史"""
    timestamp: str
    user_message: str
    system_response: str
    recommended_songs: List[str] = Field(default_factory=list)
    feedback: Optional[str] = None


class SessionContext(BaseModel):
    """会话上下文（临时偏好）"""
    current_mood: Optional[str] = None
    current_context: Optional[str] = None
    session_start: str = Field(default_factory=lambda: datetime.now().isoformat())
    recent_queries: List[str] = Field(default_factory=list)
    recent_recommendations: List[str] = Field(default_factory=list)


# ==================== 用户画像类（扩展原有功能） ====================

class UserProfile:
    """用户画像管理器"""
    
    def __init__(self, user_id: str, username: str = None, profile_dir: str = "./data/user_profiles"):
        self.user_id = user_id
        self.username = username or user_id
        self.profile_dir = profile_dir
        self.profile_path = os.path.join(profile_dir, f"{user_id}.json")
        
        # 初始化数据
        self.preferences = UserPreferences()
        self.conversation_history: List[ConversationHistory] = []
        self.liked_songs: Set[str] = set()
        self.disliked_songs: Set[str] = set()
        self.created_at: str = datetime.now().isoformat()
        self.updated_at: str = datetime.now().isoformat()
        
        # 新增：会话上下文
        self.session: Optional[SessionContext] = None
        
        # 新增：统计信息
        self.total_recommendations: int = 0
        self.positive_feedback_count: int = 0
        self.negative_feedback_count: int = 0
        
        # 加载已有画像
        self.load_profile()
    
    def load_profile(self):
        """加载用户画像"""
        if os.path.exists(self.profile_path):
            try:
                with open(self.profile_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                self.username = data.get('username', self.user_id)
                self.preferences = UserPreferences(**data.get('preferences', {}))
                self.conversation_history = [
                    ConversationHistory(**h) for h in data.get('conversation_history', [])
                ]
                self.liked_songs = set(data.get('liked_songs', []))
                self.disliked_songs = set(data.get('disliked_songs', []))
                self.created_at = data.get('created_at', self.created_at)
                self.updated_at = data.get('updated_at', self.updated_at)
                self.total_recommendations = data.get('total_recommendations', 0)
                self.positive_feedback_count = data.get('positive_feedback_count', 0)
                self.negative_feedback_count = data.get('negative_feedback_count', 0)
                
                logger.info(f"成功加载用户画像: {self.username} ({self.user_id})")
            except Exception as e:
                logger.error(f"加载用户画像失败: {e}")
    
    def save_profile(self):
        """保存用户画像"""
        os.makedirs(self.profile_dir, exist_ok=True)
        
        data = {
            'user_id': self.user_id,
            'username': self.username,
            'preferences': self.preferences.model_dump(),
            'conversation_history': [h.model_dump() for h in self.conversation_history[-50:]],
            'liked_songs': list(self.liked_songs),
            'disliked_songs': list(self.disliked_songs),
            'created_at': self.created_at,
            'updated_at': datetime.now().isoformat(),
            'total_recommendations': self.total_recommendations,
            'positive_feedback_count': self.positive_feedback_count,
            'negative_feedback_count': self.negative_feedback_count,
        }
        
        with open(self.profile_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.debug(f"用户画像已保存: {self.username}")
    
    # ==================== 会话管理（新增） ====================
    
    def start_session(self):
        """开始新会话"""
        self.session = SessionContext()
        logger.debug(f"用户 {self.username} 开始新会话")
    
    def end_session(self):
        """结束会话"""
        self.session = None
        logger.debug(f"用户 {self.username} 结束会话")
    
    def update_session_context(self, mood: str = None, context: str = None, query: str = None):
        """更新会话上下文"""
        if not self.session:
            self.start_session()
        
        if mood:
            self.session.current_mood = mood
        if context:
            self.session.current_context = context
        if query:
            self.session.recent_queries.append(query)
            self.session.recent_queries = self.session.recent_queries[-10:]
    
    # ==================== 原有方法（保持兼容） ====================
    
    def add_conversation(self, user_message: str, system_response: str, 
                        recommended_songs: List[str] = None):
        """添加对话历史"""
        conv = ConversationHistory(
            timestamp=datetime.now().isoformat(),
            user_message=user_message,
            system_response=system_response,
            recommended_songs=recommended_songs or []
        )
        self.conversation_history.append(conv)
        
        if len(self.conversation_history) > 50:
            self.conversation_history = self.conversation_history[-50:]
        
        # 更新推荐统计
        if recommended_songs:
            self.total_recommendations += len(recommended_songs)
            # 更新会话中的最近推荐
            if self.session:
                self.session.recent_recommendations.extend(recommended_songs)
                self.session.recent_recommendations = self.session.recent_recommendations[-20:]
        
        self.save_profile()
    
    def update_preferences_from_text(self, extracted_prefs: Dict):
        """从LLM提取的偏好更新用户画像"""
        # 获取置信度（默认0.5）
        confidence = extracted_prefs.get('confidence', 0.5)
        
        # 高置信度时更新持久偏好
        if confidence >= 0.6:
            if 'genres' in extracted_prefs:
                for genre in extracted_prefs['genres']:
                    if genre and genre not in self.preferences.favorite_genres:
                        self.preferences.favorite_genres.append(genre)
            
            if 'artists' in extracted_prefs:
                for artist in extracted_prefs['artists']:
                    if artist and artist not in self.preferences.favorite_artists:
                        self.preferences.favorite_artists.append(artist)
            
            if 'tags' in extracted_prefs:
                for tag in extracted_prefs['tags']:
                    if tag and tag not in self.preferences.favorite_tags:
                        self.preferences.favorite_tags.append(tag)
            
            if 'moods' in extracted_prefs:
                for mood in extracted_prefs['moods']:
                    if mood and mood not in self.preferences.moods:
                        self.preferences.moods.append(mood)
            
            if 'context' in extracted_prefs:
                context = extracted_prefs['context']
                if context and context not in self.preferences.listening_contexts:
                    self.preferences.listening_contexts.append(context)
            
            if 'tempo' in extracted_prefs:
                self.preferences.tempo_preference = extracted_prefs['tempo']
            
            if 'vocal' in extracted_prefs:
                self.preferences.vocal_preference = extracted_prefs['vocal']
            
            if 'language' in extracted_prefs:
                for lang in extracted_prefs.get('language', []):
                    if lang and lang not in self.preferences.language_preferences:
                        self.preferences.language_preferences.append(lang)
        
        # 低置信度时只更新会话上下文
        if self.session:
            if 'moods' in extracted_prefs and extracted_prefs['moods']:
                self.session.current_mood = extracted_prefs['moods'][0]
            if 'context' in extracted_prefs:
                self.session.current_context = extracted_prefs['context']
        
        logger.info(f"用户偏好已更新: {extracted_prefs}")
        self.save_profile()
    
    def add_liked_song(self, song_id: str):
        """添加喜欢的歌曲"""
        self.liked_songs.add(song_id)
        self.disliked_songs.discard(song_id)
        self.positive_feedback_count += 1
        self.save_profile()
    
    def add_disliked_song(self, song_id: str):
        """添加不喜欢的歌曲"""
        self.disliked_songs.add(song_id)
        self.liked_songs.discard(song_id)
        self.negative_feedback_count += 1
        self.save_profile()
    
    def get_preference_summary(self) -> str:
        """获取用户偏好摘要"""
        summary_parts = []
        
        if self.preferences.favorite_genres:
            summary_parts.append(f"喜欢的流派：{', '.join(self.preferences.favorite_genres)}")
        
        if self.preferences.favorite_artists:
            summary_parts.append(f"喜欢的歌手：{', '.join(self.preferences.favorite_artists[:10])}")
        
        if self.preferences.favorite_tags:
            summary_parts.append(f"喜欢的标签：{', '.join(self.preferences.favorite_tags)}")
        
        if self.preferences.moods:
            summary_parts.append(f"情绪偏好：{', '.join(self.preferences.moods)}")
        
        if self.preferences.listening_contexts:
            summary_parts.append(f"听歌场景：{', '.join(self.preferences.listening_contexts)}")
        
        if self.preferences.tempo_preference:
            summary_parts.append(f"节奏偏好：{self.preferences.tempo_preference}")
        
        if self.preferences.vocal_preference:
            summary_parts.append(f"人声偏好：{self.preferences.vocal_preference}")
        
        # 新增：会话上下文信息
        if self.session:
            if self.session.current_mood:
                summary_parts.append(f"当前心情：{self.session.current_mood}")
            if self.session.current_context:
                summary_parts.append(f"当前场景：{self.session.current_context}")
        
        if self.liked_songs:
            summary_parts.append(f"已收藏歌曲数：{len(self.liked_songs)}")
        
        if not summary_parts:
            return "暂无用户偏好信息"
        
        return "\n".join(summary_parts)
    
    def get_filter_conditions(self) -> Dict:
        """根据用户画像生成过滤条件"""
        conditions = {}
        
        if self.preferences.favorite_genres:
            conditions['preferred_genres'] = self.preferences.favorite_genres
        
        if self.preferences.dislike_genres:
            conditions['exclude_genres'] = self.preferences.dislike_genres
        
        if self.preferences.year_range:
            conditions['year_range'] = self.preferences.year_range
        
        if self.disliked_songs:
            conditions['exclude_song_ids'] = list(self.disliked_songs)
        
        return conditions
    
    def get_statistics(self) -> Dict:
        """获取用户统计信息"""
        total_feedback = self.positive_feedback_count + self.negative_feedback_count
        return {
            'total_conversations': len(self.conversation_history),
            'total_recommendations': self.total_recommendations,
            'liked_songs': len(self.liked_songs),
            'disliked_songs': len(self.disliked_songs),
            'positive_feedback': self.positive_feedback_count,
            'negative_feedback': self.negative_feedback_count,
            'satisfaction_rate': self.positive_feedback_count / max(1, total_feedback),
        }


# ==================== 多用户管理器（新增） ====================

class UserManager:
    """
    多用户管理器
    
    功能：
    - 用户注册/登录/游客模式
    - 多用户画像隔离
    - 统一的用户管理接口
    """
    
    def __init__(self, profile_dir: str = "./data/user_profiles"):
        self.profile_dir = profile_dir
        os.makedirs(profile_dir, exist_ok=True)
        
        # 用户索引文件
        self.index_path = os.path.join(profile_dir, "_user_index.json")
        self.user_index = self._load_index()  # {username: user_id}
        
        # 当前活跃的用户画像缓存 {user_id: UserProfile}
        self.active_users: Dict[str, UserProfile] = {}
        
        logger.info(f"用户管理器初始化完成，已注册用户数：{len(self.user_index)}")
    
    def _load_index(self) -> Dict[str, str]:
        """加载用户索引"""
        if os.path.exists(self.index_path):
            try:
                with open(self.index_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        return {}
    
    def _save_index(self):
        """保存用户索引"""
        with open(self.index_path, 'w', encoding='utf-8') as f:
            json.dump(self.user_index, f, ensure_ascii=False, indent=2)
    
    def _hash_password(self, password: str) -> str:
        """密码哈希"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def _get_password_path(self, user_id: str) -> str:
        """获取密码文件路径"""
        return os.path.join(self.profile_dir, f"{user_id}.pwd")
    
    # ==================== 用户认证 ====================
    
    def register(self, username: str, password: str) -> tuple[bool, str, Optional[str]]:
        """
        用户注册
        
        Returns:
            (success, message, user_id)
        """
        # 验证用户名
        if not username or len(username) < 2:
            return False, "用户名至少2个字符", None
        if len(username) > 20:
            return False, "用户名最多20个字符", None
        if username in self.user_index:
            return False, "用户名已存在", None
        
        # 验证密码
        if not password or len(password) < 4:
            return False, "密码至少4个字符", None
        
        # 生成用户ID
        user_id = str(uuid.uuid4())[:8]
        
        # 创建用户画像
        profile = UserProfile(
            user_id=user_id,
            username=username,
            profile_dir=self.profile_dir
        )
        profile.save_profile()
        
        # 保存密码
        with open(self._get_password_path(user_id), 'w') as f:
            f.write(self._hash_password(password))
        
        # 更新索引
        self.user_index[username] = user_id
        self._save_index()
        
        logger.info(f"新用户注册: {username} (ID: {user_id})")
        return True, "注册成功", user_id
    
    def login(self, username: str, password: str) -> tuple[bool, str, Optional[UserProfile]]:
        """
        用户登录
        
        Returns:
            (success, message, user_profile)
        """
        # 检查用户是否存在
        if username not in self.user_index:
            return False, "用户不存在", None
        
        user_id = self.user_index[username]
        
        # 验证密码
        pwd_path = self._get_password_path(user_id)
        if not os.path.exists(pwd_path):
            return False, "账户数据异常", None
        
        with open(pwd_path, 'r') as f:
            stored_hash = f.read().strip()
        
        if self._hash_password(password) != stored_hash:
            return False, "密码错误", None
        
        # 加载用户画像
        profile = self.get_user_profile(user_id)
        profile.start_session()
        
        logger.info(f"用户登录: {username} (ID: {user_id})")
        return True, "登录成功", profile
    
    def guest_login(self) -> UserProfile:
        """
        游客登录
        
        Returns:
            UserProfile
        """
        guest_id = f"guest_{str(uuid.uuid4())[:6]}"
        guest_name = f"游客_{guest_id[-4:]}"
        
        profile = UserProfile(
            user_id=guest_id,
            username=guest_name,
            profile_dir=self.profile_dir
        )
        profile.start_session()
        
        # 缓存但不加入索引（游客不持久化索引）
        self.active_users[guest_id] = profile
        
        logger.info(f"游客登录: {guest_name} (ID: {guest_id})")
        return profile
    
    def logout(self, user_id: str):
        """用户登出"""
        if user_id in self.active_users:
            self.active_users[user_id].end_session()
            del self.active_users[user_id]
            logger.info(f"用户登出: {user_id}")
    
    # ==================== 用户画像管理 ====================
    
    def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """获取用户画像"""
        # 先检查缓存
        if user_id in self.active_users:
            return self.active_users[user_id]
        
        # 从文件加载
        profile_path = os.path.join(self.profile_dir, f"{user_id}.json")
        if os.path.exists(profile_path):
            profile = UserProfile(user_id=user_id, profile_dir=self.profile_dir)
            self.active_users[user_id] = profile
            return profile
        
        return None
    
    def get_or_create_default_user(self) -> UserProfile:
        """获取或创建默认用户（兼容旧代码）"""
        default_id = "default_user"
        profile = self.get_user_profile(default_id)
        
        if not profile:
            profile = UserProfile(
                user_id=default_id,
                username="默认用户",
                profile_dir=self.profile_dir
            )
            self.active_users[default_id] = profile
        
        return profile
    
    def delete_user(self, user_id: str) -> bool:
        """删除用户"""
        # 从缓存移除
        if user_id in self.active_users:
            del self.active_users[user_id]
        
        # 删除文件
        profile_path = os.path.join(self.profile_dir, f"{user_id}.json")
        pwd_path = self._get_password_path(user_id)
        
        if os.path.exists(profile_path):
            os.remove(profile_path)
        if os.path.exists(pwd_path):
            os.remove(pwd_path)
        
        # 从索引移除
        username_to_remove = None
        for username, uid in self.user_index.items():
            if uid == user_id:
                username_to_remove = username
                break
        
        if username_to_remove:
            del self.user_index[username_to_remove]
            self._save_index()
        
        logger.info(f"用户已删除: {user_id}")
        return True
    
    def list_users(self) -> List[Dict]:
        """列出所有用户"""
        users = []
        for username, user_id in self.user_index.items():
            profile = self.get_user_profile(user_id)
            if profile:
                stats = profile.get_statistics()
                users.append({
                    'user_id': user_id,
                    'username': username,
                    'created_at': profile.created_at,
                    'total_conversations': stats['total_conversations'],
                    'liked_songs': stats['liked_songs'],
                })
        return users
    
    def list_user_choices(self) -> list:
        """
        给前端下拉框用的用户列表。

        这里为了简单，直接返回 user_id 列表：
        - 下拉框会显示 user_id
        - 切换用户时直接把这个值当成 user_id 用
        """
        users = self.list_users()
        return [u["user_id"] for u in users]

    
    def user_exists(self, username: str) -> bool:
        """检查用户是否存在"""
        return username in self.user_index


# ==================== 测试代码 ====================

if __name__ == "__main__":
    # 测试多用户功能
    manager = UserManager("./data/test_profiles")
    
    # 注册用户
    success, msg, user_id = manager.register("alice", "password123")
    print(f"注册: {msg}, user_id: {user_id}")
    
    # 登录
    success, msg, profile = manager.login("alice", "password123")
    print(f"登录: {msg}")
    
    if profile:
        # 更新偏好
        profile.update_preferences_from_text({
            'genres': ['流行', '摇滚'],
            'artists': ['周杰伦'],
            'moods': ['轻松'],
            'confidence': 0.9
        })
        
        # 添加对话
        profile.add_conversation(
            "推荐一些周杰伦的歌",
            "为您推荐...",
            ['song_001', 'song_002']
        )
        
        # 查看画像
        print(f"\n用户画像摘要:\n{profile.get_preference_summary()}")
        print(f"\n统计信息: {profile.get_statistics()}")
    
    # 游客登录
    guest = manager.guest_login()
    print(f"\n游客: {guest.username}")
    
    # 列出所有用户
    print(f"\n所有用户: {manager.list_users()}")