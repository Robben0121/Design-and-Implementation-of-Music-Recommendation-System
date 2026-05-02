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
                                conversation_history: List[str] = None) -> Dict:
        """
        从用户消息中提取偏好信息
        
        Returns:
            提取的偏好字典
        """
        system_prompt = """你是一个音乐偏好分析专家。从用户的对话中提取他们的音乐偏好信息。

请以JSON格式返回提取结果，包含以下字段（如果用户没提到就留空）：
{
    "genres": ["流派1", "流派2"],
    "artists": ["歌手1", "歌手2"],
    "tags": ["标签1", "标签2"],
    "moods": ["情绪1", "情绪2"],
    "context": "场景",
    "tempo": "节奏",
    "vocal": "人声",
    "year_preference": "年代偏好",
    "language": ["语言偏好"]
}

只返回JSON，不要其他文字。"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
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
                'reasoning': '抱歉，没有找到符合您需求的歌曲。',
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
    
    def _build_enhanced_query(self, user_query: str, 
                             user_profile: Optional[UserProfile]) -> str:
        """构建增强查询"""
        if not user_profile or not user_profile.preferences.favorite_genres:
            return user_query
        
        preferences = []
        if user_profile.preferences.favorite_genres:
            preferences.append(f"偏好流派: {', '.join(user_profile.preferences.favorite_genres[:3])}")
        if user_profile.preferences.moods:
            preferences.append(f"情绪偏好: {', '.join(user_profile.preferences.moods[:2])}")
        
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
   专辑：{item.album}
   风格：{item.genre} | 年份：{item.year}
   标签：{', '.join(item.tags[:5])}
   特点：{item.semantic_summary}
"""
        
        # 构建用户画像信息
        profile_info = ""
        if user_profile:
            profile_info = f"\n用户偏好：\n{user_profile.get_preference_summary()}\n"
        
        system_prompt = """你是一位资深音乐推荐专家。

请根据用户的需求和偏好，从候选歌曲中选出最合适的歌曲。

要求：
1. 返回JSON格式，包含 "recommended_songs" 和 "explanation" 两个字段
2. recommended_songs 是一个列表，每个元素是候选歌曲的编号
3. explanation 是一段简洁的推荐理由（50-100字）
4. 考虑音乐的多样性

示例输出：
{
    "recommended_songs": [3, 7, 1, 10, 15],
    "explanation": "根据您的需求为您推荐了这些歌曲..."
}"""
        
        user_prompt = f"""用户需求：{user_query}
{profile_info}
候选歌曲：
{candidates_text}

请从以上候选歌曲中选出最合适的 {top_k} 首歌曲进行推荐。"""
        
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
                explanation = "为您推荐了检索到的相关歌曲。"
            else:
                explanation = result.get('explanation', '根据您的需求为您推荐了这些歌曲。')
            
            return {
                'songs': recommended_songs,
                'explanation': explanation
            }
            
        except Exception as e:
            logger.error(f"LLM重排序失败: {e}")
            return {
                'songs': retrieved_items[:top_k],
                'explanation': '为您推荐了以下相关歌曲。'
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
    
    def chat_with_context(self, 
                         user_message: str,
                         conversation_history: List[Dict] = None) -> str:
        """与用户对话"""
        system_prompt = """你是一个友好的音乐推荐助手。你的任务是：
1. 理解用户的音乐需求和偏好
2. 通过自然对话收集用户信息（但不要问太多问题）
3. 提供个性化的音乐推荐
4. 用轻松愉快的语气与用户交流

注意：
- 不要一次问太多问题，最多问1-2个
- 如果用户的需求已经明确，直接推荐即可
- 回复简洁自然，不要太正式
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
            return "抱歉，我遇到了一些问题。能再说一遍吗？"
