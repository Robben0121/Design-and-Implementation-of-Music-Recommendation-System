from typing import List, Dict, Optional, Tuple
from openai import OpenAI
from loguru import logger
import json

from config import settings
from data_loader import MusicItem
from vector_engine import VectorSearchEngine
from user_profile import UserProfile

# 【新增】导入TempChain-ExRec模块
from tempchain_exrec import (
    TempChainExRec,
    ScenarioType,
    ExplanationPath,
    ExplanationResult
)


class LLMRecommendationEngine:
    """基于LLM的音乐推荐引擎（集成TempChain-ExRec）"""
    
    def __init__(self, vector_engine: VectorSearchEngine):
        self.vector_engine = vector_engine
        
        # 初始化 DeepSeek 客户端（兼容 OpenAI API）
        self.client = OpenAI(
            api_key=settings.DEEPSEEK_API_KEY,
            base_url=settings.DEEPSEEK_BASE_URL
        )
        self.model = settings.DEEPSEEK_MODEL
        
        # 【新增】初始化TempChain-ExRec引擎
        enable_validation = getattr(settings, 'ENABLE_EXPLANATION_VALIDATION', True)
        enable_multi_path = getattr(settings, 'ENABLE_MULTI_PATH_EXPLANATION', True)
        
        self.explainer = TempChainExRec(
            llm_client=self.client,
            model=self.model,
            enable_validation=enable_validation,
            enable_multi_path=enable_multi_path
        )
        
        logger.info("LLM推荐引擎初始化完成（集成TempChain-ExRec）")
    
    def extract_user_preferences(self, user_message: str,
                                conversation_history: List[Dict] = None) -> Dict:
        """
        从用户消息中提取偏好信息（支持场景推断 + 对话上下文）

        Returns:
            提取的偏好字典
        """
        system_prompt = """You are a music preference analysis expert. Extract music preferences from the user's message.

Rules:
1. Only extract preferences the user explicitly stated or that can be reasonably inferred; leave undetermined fields empty.
2. Infer mood, tempo and genre from the scenario, e.g.:
   - exercise/running/gym -> moods:["energetic","upbeat"], tempo:"fast", genres:["Electronic","Rock","Pop"]
   - studying/working/reading -> moods:["calm","focused"], tempo:"slow", genres:["Classical","Easy Listening","Soundtrack"]
   - sleep/bedtime -> moods:["peaceful","relaxed"], tempo:"slow", genres:["Easy Listening","Classical","Jazz"]
   - sad/heartbreak -> moods:["melancholy","emotional"], tempo:"slow", genres:["Folk","Pop"]
   - driving/travel -> moods:["easygoing","happy"], tempo:"medium", genres:["Pop","Rock"]
3. Use English for genre/mood/tag/context values, and stick to common labels the music database uses (e.g. Pop, Rock, Electronic, Folk, Jazz, Classical, Soundtrack, Easy Listening, Melodic, Catchy, Emotional, Rhythmic).
4. "confidence" reflects certainty: 0.8-1.0 for explicitly stated preferences, 0.4-0.6 for scenario-inferred ones.

Return JSON only (no other text):
{
    "genres": ["genre1"],
    "artists": ["artist1"],
    "tags": ["tag1"],
    "moods": ["mood1"],
    "context": "scenario",
    "tempo": "tempo",
    "vocal": "vocal",
    "year_preference": "year preference",
    "language": ["language preference"],
    "confidence": 0.8
}"""

        try:
            messages = [{"role": "system", "content": system_prompt}]
            if conversation_history:
                history_text = "\n".join(
                    f"{'用户' if m.get('role') == 'user' else '助手'}: {m.get('content', '')}"
                    for m in conversation_history[-6:]
                )
                messages.append({"role": "user", "content": f"对话上下文：\n{history_text}\n\n当前用户消息：{user_message}"})
            else:
                messages.append({"role": "user", "content": user_message})

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.3
            )

            content = response.choices[0].message.content.strip()
            if content.startswith("```json"):
                content = content.split("```json")[1].split("```")[0].strip()
            elif content.startswith("```"):
                content = content.split("```")[1].split("```")[0].strip()

            preferences = json.loads(content)
            logger.info(f"提取的用户偏好: {preferences}")
            return preferences

        except Exception as e:
            logger.error(f"提取用户偏好失败: {e}")
            return {}
    
    def recommend_songs(self, 
                       user_query: str,
                       user_profile: Optional[UserProfile] = None,
                       top_k: int = 10) -> Dict:
        """
        基于用户查询和画像推荐歌曲
        
        Returns:
            {
                'recommendations': [MusicItem列表],
                'reasoning': '推荐理由',
                'retrieved_count': 检索到的歌曲数
            }
        """
        logger.info(f"处理推荐请求: {user_query}")
        
        # 1. 构建增强查询
        enhanced_query = self._build_enhanced_query(user_query, user_profile)
        
        # 2. 向量检索
        retrieved = self.vector_engine.search(
            query=enhanced_query,
            top_k=settings.TOP_K_RETRIEVE
        )
        
        if not retrieved:
            logger.warning("未检索到相关歌曲")
            return {
                'recommendations': [],
                'reasoning': 'Sorry, no songs matching your request were found.',
                'retrieved_count': 0
            }
        
        # 3. 过滤不喜欢的歌曲
        if user_profile:
            retrieved = [
                r for r in retrieved 
                if r['item'].song_id not in user_profile.disliked_songs
            ]
        
        # 4. 使用LLM重排序
        recommendations = self._llm_rerank_and_explain(
            user_query=user_query,
            retrieved_items=[r['item'] for r in retrieved],
            user_profile=user_profile,
            top_k=top_k
        )
        
        return {
            'recommendations': recommendations['songs'],
            'reasoning': recommendations['explanation'],
            'retrieved_count': len(retrieved)
        }

    # ==================== Baseline 2: pure LLM recommendation (no RAG) ====================

    def _get_db_title_index(self) -> Dict:
        """Lazily build a normalized title/artist index over the music DB (used to map
        LLM-generated song titles back to database items)."""
        if getattr(self, '_db_index', None) is not None:
            return self._db_index

        import re

        def norm(s: str) -> str:
            s = re.sub(r'[^a-z0-9 ]', '', (s or '').lower())
            return re.sub(r'\s+', ' ', s).strip()

        db = self.vector_engine.music_db
        by_title: Dict[str, List[MusicItem]] = {}
        norm_titles = []
        norm_artists = []
        for item in db.music_items:
            nt = norm(item.song)
            by_title.setdefault(nt, []).append(item)
            norm_titles.append(nt)
            norm_artists.append(norm(item.artist))

        self._db_index = {
            'items': db.music_items,
            'by_title': by_title,
            'norm_titles': norm_titles,
            'norm_artists': norm_artists,
        }
        return self._db_index

    def _match_song_in_db(self, title: str, artist: str) -> Optional[MusicItem]:
        """Map an LLM-generated (title, artist) back to a DB item (exact -> fuzzy)."""
        import re
        from difflib import SequenceMatcher

        def norm(s: str) -> str:
            s = re.sub(r'[^a-z0-9 ]', '', (s or '').lower())
            return re.sub(r'\s+', ' ', s).strip()

        idx = self._get_db_title_index()
        nt, na = norm(title), norm(artist)

        # 1) exact title match (prefer matching artist when both present)
        exact = idx['by_title'].get(nt, [])
        if exact:
            for item in exact:
                if not na or norm(item.artist) == na:
                    return item
            return exact[0]

        # 2) fuzzy title (+artist) match, cheap first-char prefilter
        best, best_score = None, 0.0
        for item, it, ia in zip(idx['items'], idx['norm_titles'], idx['norm_artists']):
            if not nt or not it or nt[0] != it[0]:
                continue
            ts = SequenceMatcher(None, it, nt).ratio()
            score = ts
            if na and ia:
                score = 0.7 * ts + 0.3 * SequenceMatcher(None, ia, na).ratio()
            if score > best_score:
                best_score, best = score, item
        if best is not None and best_score >= 0.85:
            return best
        return None

    def recommend_songs_llm_only(self, user_query: str, top_k: int = 10) -> Dict:
        """
        Baseline 2: pure LLM recommendation with NO retrieval (RAG disabled).

        The model generates song titles/artists from its own knowledge; we map them
        back to database items where possible and count unmatched entries as
        hallucinations (returned as None slots in `songs`).

        Returns:
            {
                'songs': [MusicItem or None, ...],  # len == top_k (None = hallucinated slot)
                'hallucinated': int,
                'raw': [{'title': str, 'artist': str}, ...]
            }
        """
        system_prompt = """You are a music recommendation assistant. Recommend specific real songs based ONLY on the user's request, using your own knowledge of real songs. Do NOT search any database.

Return JSON only:
{
    "songs": [
        {"title": "Song Name", "artist": "Artist Name"}
    ]
}"""
        user_prompt = f"Recommend {top_k} specific real songs for the request: {user_query}"

        raw_songs = []
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.7,
            )
            content = response.choices[0].message.content.strip()
            if content.startswith("```json"):
                content = content.split("```json")[1].split("```")[0].strip()
            elif content.startswith("```"):
                content = content.split("```")[1].split("```")[0].strip()
            data = json.loads(content)
            raw_songs = data.get("songs", [])
        except Exception as e:
            logger.error(f"Baseline2 (LLM-only) failed: {e}")

        songs: List[Optional[MusicItem]] = []
        raw = []
        hallucinated = 0
        for entry in raw_songs[:top_k]:
            title = str(entry.get("title", "")).strip()
            artist = str(entry.get("artist", "")).strip()
            raw.append({"title": title, "artist": artist})
            item = self._match_song_in_db(title, artist) if title else None
            if item is not None:
                songs.append(item)
            else:
                songs.append(None)
                hallucinated += 1

        while len(songs) < top_k:
            songs.append(None)

        return {"songs": songs, "hallucinated": hallucinated, "raw": raw}

    def _build_enhanced_query(self, user_query: str,
                             user_profile: Optional[UserProfile]) -> str:
        """构建增强查询"""
        if not user_profile or not user_profile.preferences.favorite_genres:
            return user_query
        
        preferences = []
        if user_profile.preferences.favorite_genres:
            preferences.append(f"Favorite genres: {', '.join(user_profile.preferences.favorite_genres[:3])}")
        if user_profile.preferences.moods:
            preferences.append(f"Mood preferences: {', '.join(user_profile.preferences.moods[:2])}")
        
        if preferences:
            enhanced = f"{user_query}。{' '.join(preferences)}"
            logger.info(f"增强查询: {enhanced}")
            return enhanced
        
        return user_query
    
    def _llm_rerank_and_explain(self,
                               user_query: str,
                               retrieved_items: List[MusicItem],
                               user_profile: Optional[UserProfile],
                               top_k: int) -> Dict:
        """使用LLM重新排序"""
        
        # 构建候选歌曲信息
        candidates_text = ""
        for i, item in enumerate(retrieved_items[:20], 1):
            candidates_text += f"""
{i}. 《{item.song}》 - {item.artist}
   Album: {item.album}
   Genre: {item.genre} | Year: {item.year}
   Tags: {', '.join(item.tags[:5])}
   Description: {item.semantic_summary}
"""
        
        # 构建用户画像信息
        profile_info = ""
        if user_profile:
            profile_info = f"\nUser preferences:\n{user_profile.get_preference_summary()}\n"
        
        system_prompt = """You are a senior music recommendation expert.

Select the most suitable songs from the candidates based on the user's needs and preferences.

Requirements:
1. Return JSON with "recommended_songs" and "explanation" fields
2. recommended_songs is a list of candidate song numbers
3. explanation is a concise recommendation reason (50-100 words)
4. Consider diversity of music

Example output:
{
    "recommended_songs": [3, 7, 1, 10, 15],
    "explanation": "Here are some songs recommended for you..."
}"""
        
        user_prompt = f"""User request: {user_query}
{profile_info}
Candidate songs:
{candidates_text}

Select the {top_k} most suitable songs from the candidates above."""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7
            )
            
            content = response.choices[0].message.content.strip()
            
            if content.startswith("```json"):
                content = content.split("```json")[1].split("```")[0].strip()
            elif content.startswith("```"):
                content = content.split("```")[1].split("```")[0].strip()
            
            result = json.loads(content)
            
            # 提取推荐歌曲
            recommended_indices = result.get('recommended_songs', [])
            recommended_songs = []
            for idx in recommended_indices[:top_k]:
                if 1 <= idx <= len(retrieved_items):
                    recommended_songs.append(retrieved_items[idx - 1])
            
            if not recommended_songs:
                recommended_songs = retrieved_items[:top_k]
                explanation = "Here are some retrieved songs that may match your request."
            else:
                explanation = result.get('explanation', 'Here are some songs recommended for you.')
            
            return {
                'songs': recommended_songs,
                'explanation': explanation
            }
            
        except Exception as e:
            logger.error(f"LLM重排序失败: {e}")
            return {
                'songs': retrieved_items[:top_k],
                'explanation': 'Here are some related songs recommended for you.'
            }
    
    # ==================== 【新增】TempChain-ExRec可解释性方法 ====================
    
    def generate_tempchain_explanations(self,
                                       songs: List[MusicItem],
                                       user_query: str,
                                       user_profile: Optional[UserProfile],
                                       rag_scores: List[float],
                                       cf_infos: List[Optional[Dict]] = None,
                                       scenario_type: ScenarioType = ScenarioType.MULTI_DIMENSION
                                       ) -> List[ExplanationResult]:
        """
        使用TempChain-ExRec生成完整解释
        
        Args:
            songs: 推荐歌曲列表
            user_query: 用户查询
            user_profile: 用户画像
            rag_scores: RAG相似度分数
            cf_infos: 协同过滤信息列表
            scenario_type: 推荐场景类型
        
        Returns:
            ExplanationResult列表
        """
        if not cf_infos:
            cf_infos = [None] * len(songs)
        
        results = []
        for song, rag_score, cf_info in zip(songs, rag_scores, cf_infos):
            try:
                # 构建证据字典
                evidence = {
                    'rag_score': rag_score,
                }
                
                if cf_info:
                    evidence['cf_info'] = cf_info
                
                if user_profile:
                    evidence['user_interactions'] = len(user_profile.conversation_history)
                
                # 生成解释
                result = self.explainer.explain_recommendation(
                    song=song,
                    user_query=user_query,
                    user_profile=user_profile,
                    evidence=evidence,
                    scenario_type=scenario_type
                )
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"生成解释失败 for {song.song}: {e}")
        
        return results
    
    def format_explanation(self, 
                          result: ExplanationResult,
                          detail_level: str = "medium") -> str:
        """
        格式化解释结果
        
        Args:
            result: ExplanationResult
            detail_level: "simple" | "medium" | "full"
        
        Returns:
            格式化的Markdown文本
        """
        return self.explainer.format_explanation_markdown(result, detail_level)
    
    def explain_single_song_tempchain(self,
                                     song: MusicItem,
                                     user_query: str,
                                     user_profile: Optional[UserProfile],
                                     rag_score: float = 0.8,
                                     cf_info: Optional[Dict] = None,
                                     scenario_type: ScenarioType = ScenarioType.MULTI_DIMENSION
                                     ) -> str:
        """
        为单首歌曲生成详细TempChain解释
        
        Returns:
            详细的Markdown格式解释
        """
        # 构建证据
        evidence = {'rag_score': rag_score}
        if cf_info:
            evidence['cf_info'] = cf_info
        if user_profile:
            evidence['user_interactions'] = len(user_profile.conversation_history)
        
        # 生成解释
        result = self.explainer.explain_recommendation(
            song=song,
            user_query=user_query,
            user_profile=user_profile,
            evidence=evidence,
            scenario_type=scenario_type
        )
        
        # 格式化为详细版本
        return self.format_explanation(result, detail_level="full")
    
    def get_explanation_statistics(self, 
                                   results: List[ExplanationResult]) -> Dict:
        """
        获取解释统计信息（用于论文实验）
        
        Returns:
            {
                'avg_confidence': float,
                'validation_pass_rate': float,
                'path_distribution': Dict,
                'avg_iterations': float
            }
        """
        if not results:
            return {}
        
        confidences = [r.overall_confidence for r in results]
        validations = [r.validation_passed for r in results]
        iterations = [r.iteration_count for r in results]
        paths = [r.primary_chain.path_type.value for r in results]
        
        # 路径分布统计
        from collections import Counter
        path_dist = dict(Counter(paths))
        
        stats = {
            'avg_confidence': sum(confidences) / len(confidences),
            'validation_pass_rate': sum(validations) / len(validations),
            'path_distribution': path_dist,
            'avg_iterations': sum(iterations) / len(iterations),
            'n_samples': len(results)
        }
        
        logger.info(f"解释统计: {stats}")
        return stats
    
    # ==================== 原有对话方法 ====================
    
    def is_recommendation_request(self, user_message: str,
                                  conversation_history: List[Dict] = None) -> bool:
        """判断用户消息是否为推荐请求（识别无显式关键词的追问/改写，如「跑步时」「安静一点的」）。"""
        system_prompt = """You are an intent detector. Decide whether the user's message is a request for music recommendation, including:
- Direct requests (recommend / give me some / do you have any songs for ...)
- Refining, narrowing or modifying a previous request (e.g. "for running", "something calmer", "another batch", "Chinese songs")

Return JSON only: {"recommend": true} or {"recommend": false}. No other text."""
        messages = [{"role": "system", "content": system_prompt}]
        if conversation_history:
            messages.extend(conversation_history[-4:])
        messages.append({"role": "user", "content": user_message})

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.0
            )
            content = response.choices[0].message.content.strip()
            if content.startswith("```"):
                content = content.split("```")[1].strip()
                if content.startswith("json"):
                    content = content[4:].strip()
            data = json.loads(content)
            return bool(data.get("recommend", False))
        except Exception as e:
            logger.error(f"意图识别失败: {e}")
            return False

    def chat_with_context(self,
                         user_message: str,
                         conversation_history: List[Dict] = None) -> str:
        """与用户对话"""
        system_prompt = """You are a friendly music recommendation assistant. Your tasks:
1. Understand the user's music needs and preferences
2. Gather information through natural conversation (but don't ask too many questions)
3. Provide personalized music recommendations
4. Use a light, friendly tone

Notes:
- Ask at most 1-2 questions at a time
- Recommend directly when the user's need is already clear
- Keep replies concise and natural
"""
        
        messages = [{"role": "system", "content": system_prompt}]
        
        if conversation_history:
            messages.extend(conversation_history[-10:])
        
        messages.append({"role": "user", "content": user_message})
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.8
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"对话生成失败: {e}")
            return "Sorry, I ran into a problem. Could you repeat that?"
