"""
app.py - 音乐推荐系统主应用（多用户版 + 协同过滤）

功能：
1. 用户登录/注册界面
2. 多用户画像隔离
3. 游客模式支持
4. 列表推荐 + 单首反馈
5. 【新增】协同过滤推荐（User-CF + Item-CF 混合）
"""

import gradio as gr
from typing import List, Tuple, Optional, Dict
from loguru import logger
import sys
from datetime import datetime

from config import settings
from data_loader import MusicDatabase
from vector_engine import VectorSearchEngine
from user_profile import UserProfile, UserManager
from llm_engine import LLMRecommendationEngine
from collaborative_engine import CollaborativeFilterEngine  


class MusicRecommendationApp:
    """Main music recommendation application (multi-user + collaborative filtering + explainability)."""
    
    def __init__(self):
        logger.info("Initializing music recommendation system...")
        
        # 加载音乐数据库
        self.music_db = MusicDatabase(settings.MUSIC_DATA_PATH)
        
        # 初始化向量检索引擎
        self.vector_engine = VectorSearchEngine(self.music_db)
        self.vector_engine.build_index()
        
        # 初始化LLM推荐引擎（含可解释性模块）
        self.llm_engine = LLMRecommendationEngine(self.vector_engine)
        
        # 初始化用户管理器
        self.user_manager = UserManager(
            settings.USER_PROFILE_DIR if hasattr(settings, 'USER_PROFILE_DIR') 
            else "./data/user_profiles"
        )
        
        # 【新增】初始化协同过滤引擎
        self.cf_engine = CollaborativeFilterEngine(self.user_manager, self.music_db)
        self._init_collaborative_filter()
        
        # 【新增】用于缓存最近的推荐解释
        self.explanation_cache = {}
        
        logger.success("Music recommendation system initialized successfully! (with collaborative filtering + explainability)")
    
    def _init_collaborative_filter(self):
        """Initialize collaborative filtering matrix."""
        try:
            self.cf_engine.matrix_manager.build_matrix()
            stats = self.cf_engine.get_cf_statistics()
            logger.info(f"Collaborative filtering ready: {stats['n_users']} users, {stats['n_interactions']} interactions")
        except Exception as e:
            logger.warning(f"Collaborative filtering initialization: {e} (normal on first startup or without user data)")
    
    def process_message(self, user_message: str, chat_history: List, 
                       user_id: str, username: str) -> Tuple[str, List]:
        """Process user messages."""
        if not user_message.strip():
            return "", chat_history
        
        # Check whether the user is logged in
        if not user_id:
            chat_history.append({"role": "user", "content": user_message})
            chat_history.append({"role": "assistant", "content": "⚠️ Please log in before using recommendation features."})
            return "", chat_history
        
        logger.info(f"[{username}] user message: {user_message}")
        
        # Get current user profile
        current_user = self.user_manager.get_user_profile(user_id)
        if not current_user:
            chat_history.append({"role": "user", "content": user_message})
            chat_history.append({"role": "assistant", "content": "⚠️ Session expired, please log in again."})
            return "", chat_history
        
        try:
            # Check whether this is an explain request
            if ("explain" in user_message.lower() or "detail" in user_message.lower() or "details" in user_message.lower()) and any(term in user_message.lower() for term in ["song", "track", "#", "no", "number"]):
                response = self._handle_explain_request(user_message, user_id)
                if response:
                    chat_history.append({"role": "user", "content": user_message})
                    chat_history.append({"role": "assistant", "content": response})
                    return "", chat_history
            
            # 1. 提取用户偏好
            extracted_prefs = self.llm_engine.extract_user_preferences(user_message)
            if extracted_prefs:
                current_user.update_preferences_from_text(extracted_prefs)
            
            # 2. 判断用户意图
            needs_recommendation = self._check_if_needs_recommendation(user_message)
            
            if needs_recommendation:
                # 【修改】生成混合推荐（内容 + 协同过滤）
                response, recommended_ids = self._generate_hybrid_recommendation(
                    user_message, current_user, user_id
                )
            else:
                # 直接传递对话历史给LLM
                response = self.llm_engine.chat_with_context(
                    user_message=user_message,
                    conversation_history=chat_history[-10:]
                )
                recommended_ids = []
            
            # 保存对话历史
            current_user.add_conversation(
                user_message=user_message,
                system_response=response,
                recommended_songs=recommended_ids
            )
            
            # 更新聊天历史
            chat_history.append({"role": "user", "content": user_message})
            chat_history.append({"role": "assistant", "content": response})
            
            return "", chat_history
            
        except Exception as e:
            logger.error(f"Error while processing message: {e}")
            error_msg = "Sorry, something went wrong. Please try again later."
            chat_history.append({"role": "user", "content": user_message})
            chat_history.append({"role": "assistant", "content": error_msg})
            return "", chat_history
    
    def _generate_hybrid_recommendation(self, user_message: str, 
                                        current_user: UserProfile,
                                        user_id: str) -> Tuple[str, List[str]]:
        """
        【新增】生成混合推荐（内容推荐 + 协同过滤）
        
        Returns:
            (response_text, recommended_song_ids)
        """
        # 获取配置
        top_k = getattr(settings, 'TOP_K_RECOMMEND', 10)
        cf_weight = getattr(settings, 'CF_WEIGHT', 0.3)
        cf_min_feedback = getattr(settings, 'CF_MIN_FEEDBACK', 3)
        explanation_level = getattr(settings, 'EXPLANATION_LEVEL', 'simple')
        
        # 1. 内容推荐（多取一些用于融合）
        content_result = self.llm_engine.recommend_songs(
            user_query=user_message,
            user_profile=current_user,
            top_k=top_k * 2
        )
        
        content_songs = content_result['recommendations']
        reasoning = content_result['reasoning']
        
        # 2. 判断是否启用协同过滤
        user_feedback_count = len(current_user.liked_songs) + len(current_user.disliked_songs)
        enable_cf = user_feedback_count >= cf_min_feedback
        
        # 【新增】收集 RAG 分数（从向量检索）
        enhanced_query = self.llm_engine._build_enhanced_query(user_message, current_user)
        rag_results = self.vector_engine.search(
            query=enhanced_query,
            top_k=top_k * 2
        )
        rag_score_map = {r['item'].song_id: r['score'] for r in rag_results}
        
        if enable_cf:
            logger.info(f"Collaborative filtering recommendation enabled (feedback count: {user_feedback_count})")
            
            # 转换为CF需要的格式
            content_for_cf = [
                {'item': song, 'score': 0.8 - i * 0.02}  # 按排名递减分数
                for i, song in enumerate(content_songs)
            ]
            
            # 混合推荐
            hybrid_results = self.cf_engine.hybrid_recommend(
                user_id=user_id,
                content_results=content_for_cf,
                top_k=top_k,
                cf_weight=cf_weight
            )
            
            final_songs = [r['item'] for r in hybrid_results]
            cf_sources = [r.get('source', 'content') for r in hybrid_results]
            cf_reasons = [r.get('reason', '') for r in hybrid_results]
        else:
            logger.info(f"Collaborative filtering not enabled (requires at least {cf_min_feedback} feedbacks, current {user_feedback_count})")
            final_songs = content_songs[:top_k]
            cf_sources = ['content'] * len(final_songs)
            cf_reasons = [''] * len(final_songs)
        
        # 【新增】获取每首歌的 RAG 分数
        rag_scores = [rag_score_map.get(song.song_id, 0.8) for song in final_songs]
        
        # 【优化】缓存推荐上下文，用于后续按需生成详细解释
        if user_id:
            self.explanation_cache[user_id] = {
                'songs': final_songs,
                'user_query': user_message,
                'user_profile': current_user,
                'rag_scores': rag_scores,
                'cf_sources': cf_sources,
                'cf_reasons': cf_reasons,
                'timestamp': datetime.now().isoformat()
            }
        
        # 3. 格式化响应（不生成详细TempChain解释，节省资源）
        response = self._format_recommendation_response_with_cf(
            songs=final_songs,
            reasoning=reasoning,
            cf_sources=cf_sources,
            cf_reasons=cf_reasons,
            user_id=user_id,
            explanation_level=explanation_level
        )
        
        recommended_ids = [song.song_id for song in final_songs]
        return response, recommended_ids
    
    def _generate_single_explanation_on_demand(self, cached_context: Dict, song_index: int) -> str:
        """
        【新增】按需生成单首歌曲的详细TempChain-ExRec解释
        
        Args:
            cached_context: 缓存的推荐上下文
            song_index: 歌曲在列表中的索引（0-based）
        
        Returns:
            详细的Markdown格式解释
        """
        song = cached_context['songs'][song_index]
        user_query = cached_context['user_query']
        user_profile = cached_context['user_profile']
        rag_score = cached_context['rag_scores'][song_index]
        cf_source = cached_context['cf_sources'][song_index]
        cf_reason = cached_context['cf_reasons'][song_index]
        
        # 构建CF信息
        cf_info = None
        if cf_source in ['cf_only', 'hybrid']:
            cf_info = {
                'source': cf_source,
                'reason': cf_reason
            }
        
        # 调用TempChain-ExRec生成详细解释
        detailed_explanation = self.llm_engine.explain_single_song_tempchain(
            song=song,
            user_query=user_query,
            user_profile=user_profile,
            rag_score=rag_score,
            cf_info=cf_info
        )
        
        return detailed_explanation
    
    def _check_if_needs_recommendation(self, message: str) -> bool:
        """Determine whether a recommendation should be generated."""
        keywords = [
            'recommend', 'song', 'music', 'listen', 'find', 'want', 'give me',
            'play', 'any', 'suggest', 'track', 'playlist'
        ]
        return any(keyword in message.lower() for keyword in keywords)
    
    def _handle_explain_request(self, message: str, user_id: str) -> Optional[str]:
        """
        Handle explain requests like "Explain song 1" on demand.
        
        Args:
            message: user message
            user_id: user ID
        
        Returns:
            Explanation text, or None if this is not an explanation request.
        """
        # Check whether there is a cached recommendation for this user
        if user_id not in self.explanation_cache:
            return None
        
        cached = self.explanation_cache[user_id]
        
        # Try to extract the requested song number
        import re
        numbers = re.findall(r'(?:song|track|#|no\.?|number)?\s*(\d+)', message, flags=re.IGNORECASE)
        extracted = [num for group in numbers for num in group if num]
        
        if not extracted:
            # If no explicit number is provided, maybe the user wants full explanation
            if "all" in message.lower() or "every" in message.lower():
                logger.info(f"User requested detailed explanation for all recommendations ({len(cached['songs'])} songs)")
                explanation_text = "## 📖 Detailed Recommendation Explanations\n\n"
                
                # Generate detailed explanations for the first 5 songs
                for i, song in enumerate(cached['songs'][:5], 1):
                    logger.info(f"Generating detailed explanation for song {i}...")
                    expl = self._generate_single_explanation_on_demand(cached, i-1)
                    explanation_text += f"### {i}. {expl}\n\n"
                
                return explanation_text
            return None
        
        # Use the first extracted number
        song_num = int(extracted[0])
        
        # Validate the requested index
        if not (1 <= song_num <= len(cached['songs'])):
            return f"Sorry, there is no song number {song_num} in the recommendation list. There are {len(cached['songs'])} songs available."
        
        logger.info(f"User requested detailed explanation for song {song_num}")
        
        try:
            explanation = self._generate_single_explanation_on_demand(cached, song_num - 1)
            return explanation
        except Exception as e:
            logger.error(f"Failed to generate explanation: {e}")
            return "Sorry, there was a problem generating the detailed explanation. Please try again later."
    
    def _format_recommendation_response(self, songs: List, reasoning: str, **kwargs) -> str:
        """Format recommendation response (legacy compatibility)."""
        return self._format_recommendation_response_with_cf(
            songs, reasoning, ['content'] * len(songs), [''] * len(songs), **kwargs
        )
    
    def _format_recommendation_response_with_cf(self, songs: List, reasoning: str,
                                                cf_sources: List[str],
                                                cf_reasons: List[str],
                                                user_id: str = None,
                                                explanation_level: str = "simple") -> str:
        """
        Format recommendations with concise reasoning.
        
        Source markers:
        - no marker: content-based recommendation
        - ⚡: hybrid recommendation with collaborative filtering boost
        - 🎯: collaborative filtering only recommendation
        
        Note: detailed TempChain explanations are generated on demand.
        """
        if not songs:
            return "Sorry, no matching songs were found. Please try a different description."
        
        response = f"{reasoning}\n\nHere are the songs recommended for you:\n\n"
        
        has_cf_only = False
        has_hybrid = False
        
        for i, (song, source, cf_reason) in enumerate(zip(songs, cf_sources, cf_reasons), 1):
            if source == 'cf_only':
                mark = " 🎯"
                has_cf_only = True
            elif source == 'hybrid':
                mark = " ⚡"
                has_hybrid = True
            else:
                mark = ""
            
            response += f"{i}. **{song.song}** - {song.artist}{mark}\n"
            response += f"   📀 Album: {song.album}\n"
            response += f"   🎵 Genre: {song.genre} | Year: {song.year}\n"
            response += f"   🏷️ Tags: {', '.join(song.tags[:4])}\n"
            response += f"   💭 {song.semantic_summary}\n"
            
            if source in ('cf_only', 'hybrid') and cf_reason:
                response += f"   ✨ _{cf_reason}_\n"
            
            response += "\n"
        
        if has_cf_only or has_hybrid:
            response += "\n> "
            if has_cf_only:
                response += "🎯 = Recommended by users with similar tastes  "
            if has_hybrid:
                response += "⚡ = Collaborative filtering boost"
            response += "\n"
        
        response += '\n💡 **Want detailed reasoning?** Type "Explain song X" to view a TempChain-ExRec deep analysis.\n'
        
        return response
    
    def get_user_profile_summary(self, user_id: str) -> str:
        """Get user profile summary."""
        if not user_id:
            return "Please log in first."
        
        profile = self.user_manager.get_user_profile(user_id)
        if not profile:
            return "User profile does not exist."
        
        summary = f"## 👤 User Profile\n\n"
        summary += profile.get_preference_summary()
        
        stats = profile.get_statistics()
        summary += f"\n\n---\n"
        summary += f"💖 Liked songs: {stats['liked_songs']}\n"
        summary += f"👎 Disliked songs: {stats['disliked_songs']}\n"
        summary += f"💬 Conversations: {stats['total_conversations']}\n"
        summary += f"🎵 Total recommendations: {stats['total_recommendations']} songs\n"
        if stats['positive_feedback'] + stats['negative_feedback'] > 0:
            summary += f"😊 Satisfaction rate: {stats['satisfaction_rate']:.0%}\n"
        
        cf_min = getattr(settings, 'CF_MIN_FEEDBACK', 3)
        total_feedback = stats['liked_songs'] + stats['disliked_songs']
        if total_feedback >= cf_min:
            summary += f"\n🎯 Collaborative filtering: **Activated**"
        else:
            summary += f"\n🎯 Collaborative filtering: {cf_min - total_feedback} more feedbacks needed to activate"
        
        return summary
    
    def get_database_stats(self) -> str:
        """Get database statistics, including collaborative filtering."""
        stats = self.music_db.get_statistics()
        cf_stats = self.cf_engine.get_cf_statistics()
        
        result = f"""## 📊 Music Database Statistics

- Total songs: {stats['total_songs']:,}
- Artists: {stats['total_artists']:,}
- Genres: {stats['total_genres']}
- Year range: {stats['year_range'][0]} - {stats['year_range'][1]}
- Vector index count: {self.vector_engine.collection.count():,}

## 🤝 Collaborative Filtering Statistics

- Active users: {cf_stats['n_users']}
- Total interactions: {cf_stats['n_interactions']}
"""
        if cf_stats['n_interactions'] > 0:
            result += f"  - 👍 Likes: {cf_stats['n_likes']}\n"
            result += f"  - 👎 Dislikes: {cf_stats['n_dislikes']}\n"
            result += f"- Sparsity: {cf_stats['sparsity']:.2%}"
        
        return result
    
    # ==================== 反馈相关 ====================
    
    def provide_feedback(self, user_id: str, feedback_type: str, target_song: str = None):
        """Provide feedback for recommendations (batch or single song)."""
        if not user_id:
            return "Please log in first."
        
        profile = self.user_manager.get_user_profile(user_id)
        if not profile or not profile.conversation_history:
            return "No recommendations available for feedback."
        
        last_conv = profile.conversation_history[-1]
        
        # 确定反馈的歌曲列表
        if target_song:
            # 单首反馈：从下拉框选项中提取 song_id
            parts = target_song.split("|")
            if len(parts) >= 2:
                song_ids = [parts[-1].strip()]
            else:
                return "Unable to recognize the selected song."
        else:
            # 整批反馈
            song_ids = last_conv.recommended_songs
        
        if not song_ids:
            return "No recommendations available for feedback."
        
        # 记录反馈（直接操作 Set）
        feedback_count = 0
        for song_id in song_ids:
            if feedback_type == 'like':
                # 添加到喜欢列表，从不喜欢列表移除
                profile.liked_songs.add(song_id)
                profile.disliked_songs.discard(song_id)
                profile.positive_feedback_count += 1
                feedback_count += 1
            elif feedback_type == 'dislike':
                # 添加到不喜欢列表，从喜欢列表移除
                profile.disliked_songs.add(song_id)
                profile.liked_songs.discard(song_id)
                profile.negative_feedback_count += 1
                feedback_count += 1
        
        # 更新对话反馈标记
        last_conv.feedback = feedback_type
        profile.save_profile()
        
        # 【新增】刷新协同过滤矩阵
        try:
            self.cf_engine.refresh_matrix()
            logger.info("Collaborative filtering matrix updated")
        except Exception as e:
            logger.warning(f"Failed to refresh collaborative filtering: {e}")
        
        emoji = "👍" if feedback_type == 'like' else "👎"
        action = "liked" if feedback_type == 'like' else "disliked"
        
        if target_song:
            return f"{emoji} Recorded that you {action} this song!"
        else:
            return f"{emoji} Recorded that you {action} these {feedback_count} songs!"
    
    def _last_reco_dropdown_options(self, user_id: str):
        """Convert the most recent recommendation songs into dropdown options."""
        profile = self.user_manager.get_user_profile(user_id) if user_id else None
        if not profile or not profile.conversation_history:
            return gr.update(choices=[], value=None)
        last_ids = profile.conversation_history[-1].recommended_songs or []
        opts = []
        for i, sid in enumerate(last_ids, 1):
            item = self.music_db.get_by_id(sid)
            if item:
                opts.append(f"{i}. {item.song} - {item.artist} | {sid}")
            else:
                opts.append(f"{i}. (Unknown) | {sid}")
        return gr.update(choices=opts, value=None)
    
    # ==================== 用户认证方法 ====================
    
    def register_user(self, username: str, password: str, confirm_password: str) -> Tuple[str, str]:
        """User registration."""
        if not username or not password:
            return "❌ Please enter both username and password.", ""
        
        if password != confirm_password:
            return "❌ Passwords do not match.", ""
        
        success, message, user_id = self.user_manager.register(username, password)
        
        if success:
            return f"✅ {message}! Please log in.", ""
        else:
            return f"❌ {message}", ""
    
    def login_user(self, username: str, password: str):
        """User login."""
        if not username or not password:
            return "", "", "❌ Please enter username and password.", gr.update(visible=True), gr.update(visible=False), ""
        
        success, message, profile = self.user_manager.login(username, password)
        
        if success:
            # 【新增】登录后刷新协同过滤
            try:
                self.cf_engine.refresh_matrix()
            except:
                pass
            
            return (
                profile.user_id, 
                profile.username, 
                "", 
                gr.update(visible=False),  # 隐藏登录区
                gr.update(visible=True),   # 显示主界面
                self.get_user_profile_summary(profile.user_id)
            )
        else:
            return "", "", f"❌ {message}", gr.update(visible=True), gr.update(visible=False), ""
    
    def guest_login(self):
        """Guest login."""
        profile = self.user_manager.guest_login()
        return (
            profile.user_id,
            profile.username,
            "",
            gr.update(visible=False),
            gr.update(visible=True),
            self.get_user_profile_summary(profile.user_id)
        )
    
    def logout_user(self, user_id: str):
        """User logout."""
        if user_id:
            self.user_manager.logout(user_id)
        return (
            "",  # user_id
            "",  # username
            [],  # chatbot
            gr.update(visible=True),   # show login section
            gr.update(visible=False),  # hide main section
            "Please log in first."
        )
    
    # ==================== Gradio 界面 ====================
    def create_gradio_interface(self):
        """Create the Gradio interface."""

        # 自定义 CSS - 为聊天窗口添加可调整大小的功能
        self.custom_css = """
        /* 为聊天窗口容器添加可调整大小的样式 */
        .chatbot-container {
            resize: both !important;
            overflow: auto !important;
            min-width: 400px !important;
            min-height: 300px !important;
            max-width: 100% !important;
            max-height: 800px !important;
        }

        /* 添加调整手柄的视觉提示 */
        .chatbot-container::after {
            content: "";
            position: absolute;
            bottom: 2px;
            right: 2px;
            width: 10px;
            height: 10px;
            background: linear-gradient(135deg, transparent 50%, #666 50%);
            cursor: se-resize;
            pointer-events: none;
        }

        /* 用户画像区域也可调整大小 */
        .profile-container {
            resize: vertical !important;
            overflow: auto !important;
            min-height: 200px !important;
            max-height: 600px !important;
        }

        /* 添加调整大小的视觉提示 */
        .profile-container::after {
            content: "";
            position: absolute;
            bottom: 2px;
            right: 2px;
            width: 10px;
            height: 10px;
            background: linear-gradient(135deg, transparent 50%, #666 50%);
            cursor: s-resize;
            pointer-events: none;
        }
        """

        with gr.Blocks(title="🎵 Smart Music Recommendation System") as demo:

            # 状态变量
            user_id_state = gr.State("")
            username_state = gr.State("")
            
            gr.Markdown("""
            # 🎵 Smart Music Recommendation System
            > Personalized recommendations powered by LLM + collaborative filtering + explainability
            """)
            
            # ========== Login / Register Section ==========
            with gr.Group(visible=True) as login_section:
                gr.Markdown("### 👤 User Login")
                
                with gr.Tabs():
                    with gr.TabItem("Login"):
                        login_username = gr.Textbox(label="Username", placeholder="Enter username")
                        login_password = gr.Textbox(label="Password", type="password", placeholder="Enter password")
                        login_btn = gr.Button("Login", variant="primary")
                        login_msg = gr.Markdown("")
                    
                    with gr.TabItem("Register"):
                        reg_username = gr.Textbox(label="Username", placeholder="2-20 characters")
                        reg_password = gr.Textbox(label="Password", type="password", placeholder="At least 4 characters")
                        reg_confirm = gr.Textbox(label="Confirm Password", type="password")
                        reg_btn = gr.Button("Register", variant="primary")
                        reg_msg = gr.Markdown("")
                
                guest_btn = gr.Button("🎭 Guest Mode (No registration, no persistence)", variant="secondary")
            
            # ========== 主界面区域 ==========
            with gr.Group(visible=False) as main_section:
                
                with gr.Row():
                    user_display = gr.Markdown("", elem_id="user_display")
                    logout_btn = gr.Button("Logout", scale=0, size="sm")
                
                with gr.Row():
                    # Left: chat area
                    with gr.Column(scale=2):
                        chatbot = gr.Chatbot(
                            label="💬 Chat Recommendations",
                            height=450,
                            show_label=True,
                            elem_classes=["chatbot-container"]
                        )
                        
                        with gr.Row():
                            msg_input = gr.Textbox(
                                label="Input Message",
                                placeholder="Tell me what kind of music you want...",
                                lines=2,
                                scale=4
                            )
                            send_btn = gr.Button("Send", variant="primary", scale=1)

                        
                        with gr.Row():
                            like_btn = gr.Button("👍 Like these recommendations", scale=1)
                            dislike_btn = gr.Button("👎 Not satisfied", scale=1)
                            clear_btn = gr.Button("🗑️ Clear conversation", scale=1)
                        
                        # Single-song feedback controls
                        with gr.Row():
                            selected_song = gr.Dropdown(label="Feedback for a specific song", choices=[], value=None, scale=3)
                            like_one_btn = gr.Button("👍 Like this song", scale=1)
                            dislike_one_btn = gr.Button("👎 Dislike this song", scale=1)

                        feedback_msg = gr.Markdown("")
                        
                        # Example queries
                        gr.Examples(
                            examples=[
                                ["Recommend some relaxing music for evening study"],
                                ["I want upbeat pop songs from after 2020"],
                                ["Recommend songs by Chappell Roan"],
                                ["Any rock music good for working out?"],
                                ["I'm feeling down, recommend some healing songs"],
                                ["Explain song 1"],  # Example for explanation feature
                            ],
                            inputs=msg_input
                        )
                    
                    # Right: user profile
                    with gr.Column(scale=1):
                        profile_display = gr.Markdown(
                            value="Please log in first.",
                            label="User Profile",
                            elem_classes=["profile-container"]
                        )
                        refresh_profile_btn = gr.Button("🔄 Refresh Profile")
                        
                        gr.Markdown("---")
                        stats_display = gr.Markdown(
                            value=self.get_database_stats(),
                            label="Database Info"
                        )
                        
                        gr.Markdown("---")
                        gr.Markdown("""### 💡 Usage Tips
- Describe the music type, mood, or scene you want.
- For example: "Recommend some relaxing music for studying"
- The system will remember your preferences and improve over time.
- Click 👍/👎 to provide feedback and help the system learn.
- 🎯 indicates collaborative filtering recommendations.
- After 3 feedback entries, collaborative filtering will activate.

### 🔬 Explainable Recommendations
- Type "Explain song X" to view the detailed reasoning.
- The system uses a 4-step analysis chain:
  - User preference analysis
  - Evidence matching
  - Expected benefit
  - Deep interpretation
""")
            
            # ========== 事件绑定 ==========
            
            # 登录
            login_btn.click(
                fn=self.login_user,
                inputs=[login_username, login_password],
                outputs=[user_id_state, username_state, login_msg, login_section, main_section, profile_display]
            ).then(
                fn=lambda uid, uname: f"### 👤 Welcome, **{uname}**!" if uid else "",
                inputs=[user_id_state, username_state],
                outputs=[user_display]
            )
            
            # 注册
            reg_btn.click(
                fn=self.register_user,
                inputs=[reg_username, reg_password, reg_confirm],
                outputs=[reg_msg, login_msg]
            )
            
            # 游客登录
            guest_btn.click(
                fn=self.guest_login,
                inputs=[],
                outputs=[user_id_state, username_state, login_msg, login_section, main_section, profile_display]
            ).then(
                fn=lambda uid, uname: f"### 👤 **{uname}** (Guest Mode)" if uid else "",
                inputs=[user_id_state, username_state],
                outputs=[user_display]
            )
            
            # 登出
            logout_btn.click(
                fn=self.logout_user,
                inputs=[user_id_state],
                outputs=[user_id_state, username_state, chatbot, login_section, main_section, profile_display]
            )
            
            # 发送消息
            send_btn.click(
                fn=self.process_message,
                inputs=[msg_input, chatbot, user_id_state, username_state],
                outputs=[msg_input, chatbot]
            ).then(
                fn=lambda uid: self._last_reco_dropdown_options(uid),
                inputs=[user_id_state],
                outputs=[selected_song]
            ).then(
                fn=self.get_user_profile_summary,
                inputs=[user_id_state],
                outputs=[profile_display]
            )
            
            # 回车发送
            msg_input.submit(
                fn=self.process_message,
                inputs=[msg_input, chatbot, user_id_state, username_state],
                outputs=[msg_input, chatbot]
            ).then(
                fn=lambda uid: self._last_reco_dropdown_options(uid),
                inputs=[user_id_state],
                outputs=[selected_song]
            ).then(
                fn=self.get_user_profile_summary,
                inputs=[user_id_state],
                outputs=[profile_display]
            )
            
            # 反馈（整批）
            like_btn.click(
                fn=lambda uid: self.provide_feedback(uid, 'like'),
                inputs=[user_id_state],
                outputs=[feedback_msg]
            ).then(
                fn=self.get_user_profile_summary,
                inputs=[user_id_state],
                outputs=[profile_display]
            )
            
            dislike_btn.click(
                fn=lambda uid: self.provide_feedback(uid, 'dislike'),
                inputs=[user_id_state],
                outputs=[feedback_msg]
            ).then(
                fn=self.get_user_profile_summary,
                inputs=[user_id_state],
                outputs=[profile_display]
            )

            # 反馈（单首）
            like_one_btn.click(
                fn=lambda uid, target: self.provide_feedback(uid, 'like', target),
                inputs=[user_id_state, selected_song],
                outputs=[feedback_msg]
            ).then(
                fn=self.get_user_profile_summary,
                inputs=[user_id_state],
                outputs=[profile_display]
            )

            dislike_one_btn.click(
                fn=lambda uid, target: self.provide_feedback(uid, 'dislike', target),
                inputs=[user_id_state, selected_song],
                outputs=[feedback_msg]
            ).then(
                fn=self.get_user_profile_summary,
                inputs=[user_id_state],
                outputs=[profile_display]
            )
            
            # 清空对话
            clear_btn.click(
                fn=lambda: [],
                inputs=[],
                outputs=[chatbot]
            )
            
            # 刷新画像
            refresh_profile_btn.click(
                fn=self.get_user_profile_summary,
                inputs=[user_id_state],
                outputs=[profile_display]
            )
        
        return demo


def main():
    """Main function."""
    # 配置日志
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add("logs/app.log", rotation="100 MB", level="DEBUG")
    
    try:
        app = MusicRecommendationApp()
        demo = app.create_gradio_interface()
        demo.launch(share=True, inbrowser=True, css=app.custom_css)
        
    except Exception as e:
        logger.error(f"Application failed to start: {e}")
        raise


if __name__ == "__main__":
    main()