"""
TempChain-ExRec: Template-Chain Explainable Recommendation Algorithm

完整实现包含：
1. 三层模板库（Basic/Intermediate/Advanced）
2. 动态模板选择机制
3. 多路径Chain-of-Explanation生成（6条路径）
4. 自动质量验证与迭代优化
5. 模板使用统计与优化
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from loguru import logger
import json
import numpy as np
from openai import OpenAI

from data_loader import MusicItem
from user_profile import UserProfile


# ==================== 枚举类型定义 ====================

class TemplateLevel(Enum):
    """模板复杂度级别"""
    BASIC = "basic"           # 基础级：简单直接
    INTERMEDIATE = "intermediate"  # 中级：结构化分析
    ADVANCED = "advanced"     # 高级：深度推理


class ScenarioType(Enum):
    """推荐场景类型"""
    SIMILARITY = "similarity"     # 相似度推荐
    MOOD = "mood"                 # 情绪/氛围推荐
    COLLABORATIVE = "collaborative"  # 协同过滤推荐
    MULTI_DIMENSION = "multi_dimension"  # 多维度综合推荐
    CONTEXTUAL = "contextual"     # 场景化推荐
    NARRATIVE = "narrative"       # 叙事性推荐


class ExplanationPath(Enum):
    """解释路径类型"""
    EMOTION = "emotion"       # 情感路径
    RHYTHM = "rhythm"         # 节奏路径
    LYRIC = "lyric"          # 歌词路径
    SIMILARITY = "similarity"  # 相似度路径
    CULTURAL = "cultural"     # 文化路径
    CONTEXT = "context"       # 场景路径


# ==================== 数据类定义 ====================

@dataclass
class ExplanationChain:
    """4步解释链条"""
    step_1_user_analysis: str      # 用户偏好分析
    step_2_evidence_matching: str  # 证据匹配
    step_3_benefit_prediction: str # 收益预测
    step_4_cultural_expansion: str # 文化/情感扩展
    path_type: ExplanationPath     # 解释路径类型
    confidence: float              # 置信度 (0-1)
    evidence_sources: List[str]    # 证据来源


@dataclass
class ExplanationResult:
    """完整的解释结果"""
    song: MusicItem
    primary_chain: ExplanationChain      # 主解释链
    alternative_chains: List[ExplanationChain]  # 备选解释链
    template_used: str                   # 使用的模板
    template_level: TemplateLevel        # 模板级别
    scenario_type: ScenarioType          # 场景类型
    overall_confidence: float            # 总体置信度
    validation_passed: bool              # 是否通过验证
    iteration_count: int                 # 迭代次数


# ==================== 模板管理器 ====================

class TemplateManager:
    """
    模板管理器
    
    负责：
    1. 存储和管理三层模板库
    2. 根据用户画像和场景动态选择模板
    3. 统计模板使用情况
    4. 基于反馈优化模板选择策略
    """
    
    def __init__(self):
        self.templates = self._init_templates()
        self.usage_stats = {}  # 模板使用统计
        logger.info("模板管理器初始化完成")
    
    def _init_templates(self) -> Dict:
        """初始化三层模板库"""
        templates = {
            # ========== 基础级模板 ==========
            TemplateLevel.BASIC: {
                ScenarioType.SIMILARITY: {
                    "template": """Similarity-based recommendation for 《{song_name}》:
1. User preference: you like {user_preference} music
2. Match reason: this song's {similarity_dimension} is similar to 《{reference_song}》 you like
3. Expected effect: it delivers a similar listening experience
4. Note: one of {artist}'s representative works""",
                    "required_vars": ["song_name", "user_preference", "similarity_dimension", "reference_song", "artist"]
                },
                
                ScenarioType.MOOD: {
                    "template": """Mood-based recommendation for 《{song_name}》:
1. User state: you currently want {mood} music
2. Match reason: this song's atmosphere is {song_mood} with a {tempo} tempo
3. Expected effect: it can adjust your {mood_effect}
4. Note: suitable for {context}""",
                    "required_vars": ["song_name", "mood", "song_mood", "tempo", "mood_effect", "context"]
                },
                
                ScenarioType.COLLABORATIVE: {
                    "template": """Collaborative filtering recommendation for 《{song_name}》:
1. User group: {n_users} users with similar tastes
2. Match reason: they all like this song, score {cf_score}
3. Expected effect: it likely fits your taste
4. Note: {genre}, released in {year}""",
                    "required_vars": ["song_name", "n_users", "cf_score", "genre", "year"]
                }
            },
            
            # ========== 中级模板 ==========
            TemplateLevel.INTERMEDIATE: {
                ScenarioType.MULTI_DIMENSION: {
                    "template": """Multi-dimensional analysis recommendation for 《{song_name}》:
1. User profile analysis:
   - Preferred styles: {favorite_genres}
   - Mood tendencies: {mood_preferences}
   - Listening contexts: {typical_context}

2. Multi-dimensional match evidence:
   - Style similarity: {style_similarity_score} (similar to {ref_songs})
   - Mood fit: {mood_match_score}
   - Collaborative filtering: recommended by {cf_users} similar users

3. Expected benefits:
   - Emotional: {emotional_benefit}
   - Cognitive: {cognitive_benefit}

4. Deep interpretation:
   - Musical characteristics: {musical_characteristics}
   - Cultural background: {cultural_context}""",
                    "required_vars": [
                        "song_name", "favorite_genres", "mood_preferences", "typical_context",
                        "style_similarity_score", "ref_songs", "mood_match_score",
                        "cf_users", "emotional_benefit", "cognitive_benefit",
                        "musical_characteristics", "cultural_context"
                    ]
                },
                
                ScenarioType.CONTEXTUAL: {
                    "template": """Contextual recommendation for 《{song_name}》:
1. Scenario analysis:
   - Current context: {current_context}
   - User need: {user_need}
   - Time characteristics: {time_feature}

2. Scenario match:
   - Song features: {song_features} fits the {current_context} scenario
   - Tempo fit: {tempo_analysis}
   - Atmosphere: {atmosphere_creation}

3. Effects:
   - Short-term: {immediate_effect}
   - Long-term: {long_term_effect}

4. Personalized tips: {personalized_tips}""",
                    "required_vars": [
                        "song_name", "current_context", "user_need", "time_feature",
                        "song_features", "tempo_analysis", "atmosphere_creation",
                        "immediate_effect", "long_term_effect", "personalized_tips"
                    ]
                },
                
                ScenarioType.NARRATIVE: {
                    "template": """Narrative recommendation for 《{song_name}》:
1. The music story:
   Created in {creation_background}, {artist} expressed {theme} through {creative_technique}

2. Why it suits you:
   - Theme resonance: {theme} relates to your interest in {user_interest}
   - Emotional connection: {emotional_connection}
   - Taste match: {taste_match}

3. Listening experience:
   - First impression: {first_impression}
   - Deeper experience: {deep_experience}

4. Related: if you like this, also try {related_songs}""",
                    "required_vars": [
                        "song_name", "creation_background", "artist", "creative_technique", "theme",
                        "user_interest", "emotional_connection", "taste_match",
                        "first_impression", "deep_experience", "related_songs"
                    ]
                }
            },
            
            # ========== 高级模板 ==========
            TemplateLevel.ADVANCED: {
                ScenarioType.MULTI_DIMENSION: {
                    "template": """Deep analysis recommendation for 《{song_name}》 - {artist}:

[Step 1: Deep analysis of your music profile]
Based on your {n_interactions} interactions, we built your music preference model:
- Core preference dimensions: {core_preferences}
- Emotional patterns: {emotion_pattern}
- Exploration boundaries: {exploration_boundary}
- Music maturity: {music_maturity}

[Step 2: Multi-layer evidence chain]
Layer 1 - Semantic matching (RAG):
  Similarity: {rag_score}, main matched dimensions: {rag_dimensions}
  Reference songs: {rag_references}

Layer 2 - Collaborative signals (CF):
  Recommended by {cf_n_users} similar users (similarity {cf_similarity})
  Their common characteristics: {cf_user_characteristics}

Layer 3 - Deep features:
  - Music theory: {music_theory_analysis}
  - Emotion computing: {emotion_computation}
  - Cultural symbols: {cultural_symbols}

[Step 3: Multi-dimensional benefit prediction]
Immediate benefits:
  - Mood regulation: {immediate_mood_effect}
  - Cognitive experience: {immediate_cognitive_effect}

Medium-term benefits:
  - Taste expansion: {medium_term_taste_expansion}
  - Emotional resonance: {medium_term_emotional_resonance}

Long-term value:
  - Music literacy: {long_term_music_literacy}
  - Cultural understanding: {long_term_cultural_understanding}

[Step 4: Deep cultural & emotional interpretation]
Musicology perspective: {musicology_perspective}
Cultural studies perspective: {cultural_studies_perspective}
Emotional psychology perspective: {emotional_psychology_perspective}

Personalized insights: {personalized_insights}""",
                    "required_vars": [
                        "song_name", "artist", "n_interactions", "core_preferences",
                        "emotion_pattern", "exploration_boundary", "music_maturity",
                        "rag_score", "rag_dimensions", "rag_references",
                        "cf_n_users", "cf_similarity", "cf_user_characteristics",
                        "music_theory_analysis", "emotion_computation", "cultural_symbols",
                        "immediate_mood_effect", "immediate_cognitive_effect",
                        "medium_term_taste_expansion", "medium_term_emotional_resonance",
                        "long_term_music_literacy", "long_term_cultural_understanding",
                        "musicology_perspective", "cultural_studies_perspective",
                        "emotional_psychology_perspective", "personalized_insights"
                    ]
                }
            }
        }
        
        return templates
    
    def select_template(self,
                       user_profile: Optional[UserProfile],
                       scenario_type: ScenarioType,
                       query_complexity: str = "medium") -> Tuple[str, TemplateLevel]:
        """
        动态选择模板
        
        Args:
            user_profile: 用户画像
            scenario_type: 推荐场景类型
            query_complexity: 查询复杂度 (simple/medium/complex)
        
        Returns:
            (template_string, template_level)
        """
        # 计算用户音乐成熟度
        maturity_score = self._calculate_user_maturity(user_profile)
        
        # 根据成熟度和查询复杂度选择模板级别
        template_level = self._determine_template_level(
            maturity_score, query_complexity
        )
        
        # 获取模板
        if template_level in self.templates and scenario_type in self.templates[template_level]:
            template_data = self.templates[template_level][scenario_type]
            template = template_data["template"]
            
            # 记录使用统计
            self._record_usage(template_level, scenario_type)
            
            logger.info(f"选择模板: {template_level.value} - {scenario_type.value}")
            return template, template_level
        else:
            # 降级到基础模板
            logger.warning(f"未找到匹配模板，使用基础模板")
            return self._get_fallback_template(scenario_type)
    
    def _calculate_user_maturity(self, user_profile: Optional[UserProfile]) -> float:
        """
        计算用户音乐成熟度 (0-1)
        
        考虑因素：
        - 交互历史长度
        - 音乐品味多样性
        - 反馈质量
        """
        if not user_profile:
            return 0.3  # 新用户默认成熟度
        
        # 交互次数因子 (0-0.4)
        n_conversations = len(user_profile.conversation_history)
        interaction_factor = min(n_conversations / 50, 0.4)
        
        # 反馈数量因子 (0-0.3)
        n_feedback = len(user_profile.liked_songs) + len(user_profile.disliked_songs)
        feedback_factor = min(n_feedback / 30, 0.3)
        
        # 品味多样性因子 (0-0.3)
        n_genres = len(user_profile.preferences.favorite_genres)
        diversity_factor = min(n_genres / 5, 0.3)
        
        maturity = interaction_factor + feedback_factor + diversity_factor
        logger.debug(f"用户成熟度: {maturity:.2f} (交互:{interaction_factor:.2f}, 反馈:{feedback_factor:.2f}, 多样性:{diversity_factor:.2f})")
        
        return maturity
    
    def _determine_template_level(self, maturity_score: float, 
                                  query_complexity: str) -> TemplateLevel:
        """确定模板级别"""
        complexity_scores = {
            "simple": 0.0,
            "medium": 0.3,
            "complex": 0.6
        }
        
        complexity_score = complexity_scores.get(query_complexity, 0.3)
        combined_score = (maturity_score * 0.7 + complexity_score * 0.3)
        
        if combined_score >= 0.7:
            return TemplateLevel.ADVANCED
        elif combined_score >= 0.4:
            return TemplateLevel.INTERMEDIATE
        else:
            return TemplateLevel.BASIC
    
    def _get_fallback_template(self, scenario_type: ScenarioType) -> Tuple[str, TemplateLevel]:
        """获取降级模板"""
        basic_templates = self.templates[TemplateLevel.BASIC]
        
        if scenario_type in basic_templates:
            return basic_templates[scenario_type]["template"], TemplateLevel.BASIC
        
        # 使用通用相似度模板
        return basic_templates[ScenarioType.SIMILARITY]["template"], TemplateLevel.BASIC
    
    def _record_usage(self, template_level: TemplateLevel, scenario_type: ScenarioType):
        """记录模板使用统计"""
        key = f"{template_level.value}_{scenario_type.value}"
        if key not in self.usage_stats:
            self.usage_stats[key] = {"count": 0, "total_score": 0.0}
        self.usage_stats[key]["count"] += 1
    
    def update_template_from_feedback(self, template_level: TemplateLevel,
                                     scenario_type: ScenarioType,
                                     feedback_score: float):
        """
        根据用户反馈更新模板统计
        
        Args:
            template_level: 模板级别
            scenario_type: 场景类型
            feedback_score: 反馈分数 (0-1)
        """
        key = f"{template_level.value}_{scenario_type.value}"
        if key in self.usage_stats:
            self.usage_stats[key]["total_score"] += feedback_score
            avg_score = self.usage_stats[key]["total_score"] / self.usage_stats[key]["count"]
            logger.info(f"模板 {key} 平均评分更新为: {avg_score:.3f}")


# ==================== Chain-of-Explanation生成器 ====================

class ChainOfExplanationGenerator:
    """
    CoE生成器
    
    负责：
    1. 生成6条不同路径的解释链
    2. 使用LLM填充模板
    3. 计算各路径置信度
    4. 选择最优路径组合
    """
    
    def __init__(self, llm_client: OpenAI, model: str):
        self.client = llm_client
        self.model = model
        self.path_definitions = self._init_path_definitions()
        logger.info("CoE生成器初始化完成")
    
    def _init_path_definitions(self) -> Dict:
        """初始化6条解释路径的定义"""
        return {
            ExplanationPath.EMOTION: {
                "focus": "Emotional resonance and mood regulation",
                "keywords": ["emotion", "mood", "feeling", "resonance"],
                "weight": 0.25
            },
            ExplanationPath.RHYTHM: {
                "focus": "Rhythm, melody and listening experience",
                "keywords": ["rhythm", "melody", "arrangement", "timbre"],
                "weight": 0.20
            },
            ExplanationPath.LYRIC: {
                "focus": "Lyric theme and narrative",
                "keywords": ["lyrics", "theme", "story", "expression"],
                "weight": 0.15
            },
            ExplanationPath.SIMILARITY: {
                "focus": "Similarity and stylistic continuity",
                "keywords": ["similar", "style", "genre", "taste"],
                "weight": 0.20
            },
            ExplanationPath.CULTURAL: {
                "focus": "Cultural symbols and era background",
                "keywords": ["culture", "era", "background", "influence"],
                "weight": 0.10
            },
            ExplanationPath.CONTEXT: {
                "focus": "Use case and functional value",
                "keywords": ["scenario", "use", "suitable", "effect"],
                "weight": 0.10
            }
        }
    
    def generate_multi_path_chains(self,
                                   song: MusicItem,
                                   user_query: str,
                                   user_profile: Optional[UserProfile],
                                   evidence: Dict,
                                   n_paths: int = 3) -> List[ExplanationChain]:
        """
        生成多条路径的解释链
        
        Args:
            song: 推荐歌曲
            user_query: 用户查询
            user_profile: 用户画像
            evidence: 证据字典 {rag_score, cf_info, etc.}
            n_paths: 生成路径数量
        
        Returns:
            解释链列表，按置信度排序
        """
        logger.info(f"生成 {n_paths} 条解释路径")
        
        # 分析查询，选择最相关的路径
        relevant_paths = self._select_relevant_paths(user_query, n_paths)
        
        chains = []
        for path_type in relevant_paths:
            try:
                chain = self._generate_single_path_chain(
                    song, user_query, user_profile, evidence, path_type
                )
                chains.append(chain)
            except Exception as e:
                logger.error(f"生成路径 {path_type.value} 失败: {e}")
        
        # 按置信度排序
        chains.sort(key=lambda x: x.confidence, reverse=True)
        
        return chains
    
    def _select_relevant_paths(self, user_query: str, n_paths: int) -> List[ExplanationPath]:
        """根据查询选择最相关的路径"""
        scores = {}
        query_lower = user_query.lower()
        
        for path_type, definition in self.path_definitions.items():
            # 关键词匹配分数
            keyword_score = sum(1 for kw in definition["keywords"] if kw in query_lower)
            # 结合权重
            scores[path_type] = keyword_score * 10 + definition["weight"] * 5
        
        # 选择分数最高的n条路径
        sorted_paths = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        selected = [path for path, _ in sorted_paths[:n_paths]]
        
        # 确保至少包含EMOTION和SIMILARITY
        if ExplanationPath.EMOTION not in selected:
            selected[0] = ExplanationPath.EMOTION
        if ExplanationPath.SIMILARITY not in selected and len(selected) > 1:
            selected[1] = ExplanationPath.SIMILARITY
        
        logger.info(f"选择路径: {[p.value for p in selected]}")
        return selected
    
    def _generate_single_path_chain(self,
                                    song: MusicItem,
                                    user_query: str,
                                    user_profile: Optional[UserProfile],
                                    evidence: Dict,
                                    path_type: ExplanationPath) -> ExplanationChain:
        """生成单条路径的解释链"""
        
        path_def = self.path_definitions[path_type]
        
        system_prompt = f"""You are a music recommendation explanation expert, generating an explanation using the [{path_type.value}] path.

Path focus: {path_def['focus']}

Generate a 4-step Chain-of-Explanation, each step 30-40 words:
1. User preference analysis - analyze the user from the {path_def['focus']} angle
2. Evidence matching - explain how the song matches on the {path_def['focus']} dimension
3. Benefit prediction - predict benefits in terms of {path_def['focus']}
4. Deep interpretation - the deeper value of {path_def['focus']}

Return JSON:
{{
    "step_1": "...",
    "step_2": "...",
    "step_3": "...",
    "step_4": "...",
    "confidence": 0.85
}}

confidence is the explanation confidence (0-1)."""
        
        # 构建证据文本
        evidence_text = self._format_evidence(evidence)
        
        # 构建用户信息
        user_info = "New user"
        if user_profile:
            user_info = user_profile.get_preference_summary()
        
        user_prompt = f"""User query: {user_query}

User information:
{user_info}

Recommended song:
- Title: 《{song.song}》
- Artist: {song.artist}
- Genre: {song.genre}
- Tags: {', '.join(song.tags[:5])}
- Description: {song.semantic_summary}

Evidence:
{evidence_text}

Generate the explanation chain from the [{path_type.value}] path."""
        
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
            
            # 解析JSON
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            result = json.loads(content)
            
            chain = ExplanationChain(
                step_1_user_analysis=result.get("step_1", ""),
                step_2_evidence_matching=result.get("step_2", ""),
                step_3_benefit_prediction=result.get("step_3", ""),
                step_4_cultural_expansion=result.get("step_4", ""),
                path_type=path_type,
                confidence=result.get("confidence", 0.7),
                evidence_sources=list(evidence.keys())
            )
            
            return chain
            
        except Exception as e:
            logger.error(f"生成路径失败: {e}")
            # 降级方案
            return self._create_fallback_chain(song, path_type, evidence)
    
    def _format_evidence(self, evidence: Dict) -> str:
        """格式化证据"""
        lines = []
        
        if "rag_score" in evidence:
            score = max(0.0, min(1.0, float(evidence['rag_score'])))
            lines.append(f"- RAG similarity: {score:.2%}")

        if "cf_info" in evidence and evidence["cf_info"]:
            cf = evidence["cf_info"]
            reason = cf.get('reason', '')
            if reason:
                lines.append(f"- Collaborative filtering: {reason}")

        if "user_interactions" in evidence:
            lines.append(f"- User interactions: {evidence['user_interactions']}")

        return "\n".join(lines) if lines else "No additional evidence"
    
    def _create_fallback_chain(self, song: MusicItem, 
                               path_type: ExplanationPath,
                               evidence: Dict) -> ExplanationChain:
        """创建降级解释链"""
        return ExplanationChain(
            step_1_user_analysis=f"Based on your music preferences",
            step_2_evidence_matching=f"《{song.song}》 matches your needs on the {path_type.value} dimension",
            step_3_benefit_prediction=f"It delivers a good experience on the {path_type.value} dimension",
            step_4_cultural_expansion=f"A representative work by {song.artist}",
            path_type=path_type,
            confidence=0.6,
            evidence_sources=list(evidence.keys())
        )


# ==================== 解释验证器 ====================

class ExplanationValidator:
    """
    解释验证器
    
    负责：
    1. 评估解释质量（一致性、完整性、证据支持度）
    2. 迭代优化解释
    3. 生成质量报告
    """
    
    def __init__(self, llm_client: OpenAI, model: str):
        self.client = llm_client
        self.model = model
        logger.info("解释验证器初始化完成")
    
    def validate_and_improve(self,
                            chain: ExplanationChain,
                            song: MusicItem,
                            evidence: Dict,
                            max_iterations: int = 2) -> Tuple[ExplanationChain, bool, Dict]:
        """
        验证并改进解释
        
        Returns:
            (improved_chain, validation_passed, quality_report)
        """
        current_chain = chain
        iteration = 0
        
        while iteration < max_iterations:
            # 评估质量
            quality_scores = self._evaluate_quality(current_chain, song, evidence)
            
            # 判断是否通过
            overall_score = np.mean(list(quality_scores.values()))
            
            if overall_score >= 0.7:  # 通过阈值
                logger.info(f"验证通过 (分数: {overall_score:.2f})")
                return current_chain, True, quality_scores
            
            # 尝试改进
            logger.info(f"第{iteration+1}次改进 (当前分数: {overall_score:.2f})")
            try:
                current_chain = self._improve_chain(
                    current_chain, song, evidence, quality_scores
                )
                iteration += 1
            except Exception as e:
                logger.error(f"改进失败: {e}")
                break
        
        # 达到最大迭代次数
        final_scores = self._evaluate_quality(current_chain, song, evidence)
        final_score = np.mean(list(final_scores.values()))
        passed = final_score >= 0.6  # 降低阈值
        
        logger.info(f"最终验证{'通过' if passed else '未通过'} (分数: {final_score:.2f})")
        return current_chain, passed, final_scores
    
    def _evaluate_quality(self, chain: ExplanationChain, 
                         song: MusicItem, evidence: Dict) -> Dict[str, float]:
        """评估解释质量"""
        scores = {}
        
        # 1. 完整性检查 (0-1)
        completeness = self._check_completeness(chain)
        scores["completeness"] = completeness
        
        # 2. 一致性检查 (0-1)
        consistency = self._check_consistency(chain, song)
        scores["consistency"] = consistency
        
        # 3. 证据支持度 (0-1)
        evidence_support = self._check_evidence_support(chain, evidence)
        scores["evidence_support"] = evidence_support
        
        return scores
    
    def _check_completeness(self, chain: ExplanationChain) -> float:
        """检查完整性"""
        steps = [
            chain.step_1_user_analysis,
            chain.step_2_evidence_matching,
            chain.step_3_benefit_prediction,
            chain.step_4_cultural_expansion
        ]
        
        # 检查每一步是否非空且有足够长度
        valid_steps = sum(1 for step in steps if step and len(step) >= 10)
        return valid_steps / 4.0
    
    def _check_consistency(self, chain: ExplanationChain, song: MusicItem) -> float:
        """检查一致性（简化版，实际可用LLM评估）"""
        # 检查歌曲名是否在解释中
        all_text = (chain.step_1_user_analysis + chain.step_2_evidence_matching +
                   chain.step_3_benefit_prediction + chain.step_4_cultural_expansion).lower()
        
        mentions_song = song.song.lower() in all_text or song.artist.lower() in all_text
        return 0.8 if mentions_song else 0.5
    
    def _check_evidence_support(self, chain: ExplanationChain, evidence: Dict) -> float:
        """检查证据支持度"""
        # 如果有高置信度证据，分数更高
        if "rag_score" in evidence and evidence["rag_score"] > 0.7:
            return 0.9
        elif "cf_info" in evidence and evidence["cf_info"]:
            return 0.85
        else:
            return 0.7
    
    def _improve_chain(self, chain: ExplanationChain, song: MusicItem,
                      evidence: Dict, quality_scores: Dict) -> ExplanationChain:
        """改进解释链（简化版）"""
        # 找出最弱的维度
        weakest = min(quality_scores.items(), key=lambda x: x[1])
        weakness = weakest[0]
        
        # 针对性改进
        if weakness == "completeness":
            # 补充空白步骤
            if not chain.step_4_cultural_expansion or len(chain.step_4_cultural_expansion) < 10:
                chain.step_4_cultural_expansion = f"{song.artist} is a representative artist in the {song.genre} genre, and this song showcases their distinctive style"
        
        return chain


# ==================== TempChain-ExRec主引擎 ====================

class TempChainExRec:
    """
    TempChain-ExRec主引擎
    
    整合所有模块，提供统一接口
    """
    
    def __init__(self, llm_client: OpenAI, model: str, 
                 enable_validation: bool = True,
                 enable_multi_path: bool = True):
        self.template_manager = TemplateManager()
        self.coe_generator = ChainOfExplanationGenerator(llm_client, model)
        self.validator = ExplanationValidator(llm_client, model)
        
        self.enable_validation = enable_validation
        self.enable_multi_path = enable_multi_path
        
        logger.success("TempChain-ExRec引擎初始化完成")
    
    def explain_recommendation(self,
                              song: MusicItem,
                              user_query: str,
                              user_profile: Optional[UserProfile],
                              evidence: Dict,
                              scenario_type: ScenarioType = ScenarioType.MULTI_DIMENSION
                              ) -> ExplanationResult:
        """
        生成完整的推荐解释
        
        Args:
            song: 推荐歌曲
            user_query: 用户查询
            user_profile: 用户画像
            evidence: 证据字典 {rag_score, cf_info, etc.}
            scenario_type: 场景类型
        
        Returns:
            ExplanationResult
        """
        logger.info(f"为《{song.song}》生成解释")
        
        # 1. 选择模板
        template, template_level = self.template_manager.select_template(
            user_profile, scenario_type
        )
        
        # 2. 生成解释链
        if self.enable_multi_path:
            chains = self.coe_generator.generate_multi_path_chains(
                song, user_query, user_profile, evidence, n_paths=3
            )
            primary_chain = chains[0] if chains else None
            alternative_chains = chains[1:] if len(chains) > 1 else []
        else:
            primary_chain = self.coe_generator._generate_single_path_chain(
                song, user_query, user_profile, evidence, ExplanationPath.EMOTION
            )
            alternative_chains = []
        
        if not primary_chain:
            raise ValueError("解释生成失败")
        
        # 3. 验证与改进
        validation_passed = True
        quality_report = {}
        iteration_count = 0
        
        if self.enable_validation:
            primary_chain, validation_passed, quality_report = \
                self.validator.validate_and_improve(
                    primary_chain, song, evidence, max_iterations=2
                )
            iteration_count = 1 if not validation_passed else 0
        
        # 4. 计算总体置信度
        overall_confidence = self._calculate_overall_confidence(
            primary_chain, alternative_chains, quality_report
        )
        
        # 5. 构建结果
        result = ExplanationResult(
            song=song,
            primary_chain=primary_chain,
            alternative_chains=alternative_chains,
            template_used=template,
            template_level=template_level,
            scenario_type=scenario_type,
            overall_confidence=overall_confidence,
            validation_passed=validation_passed,
            iteration_count=iteration_count
        )
        
        return result
    
    def _calculate_overall_confidence(self, primary: ExplanationChain,
                                     alternatives: List[ExplanationChain],
                                     quality: Dict) -> float:
        """计算总体置信度"""
        # 主解释置信度权重60%
        conf = primary.confidence * 0.6
        
        # 备选解释数量和质量权重20%
        if alternatives:
            avg_alt_conf = np.mean([c.confidence for c in alternatives])
            conf += avg_alt_conf * 0.2
        
        # 验证质量权重20%
        if quality:
            avg_quality = np.mean(list(quality.values()))
            conf += avg_quality * 0.2
        
        return min(conf, 1.0)
    
    def format_explanation_markdown(self, result: ExplanationResult,
                                   detail_level: str = "full") -> str:
        """Format as Markdown."""
        chain = result.primary_chain

        if detail_level == "simple":
            return f"""**《{result.song.song}》** - {result.song.artist}
{chain.step_2_evidence_matching} {chain.step_3_benefit_prediction}"""

        elif detail_level == "medium":
            return f"""## Explanation for 《{result.song.song}》

**Path**: {chain.path_type.value} | **Confidence**: {result.overall_confidence:.0%}

1️⃣ {chain.step_1_user_analysis}
2️⃣ {chain.step_2_evidence_matching}
3️⃣ {chain.step_3_benefit_prediction}
4️⃣ {chain.step_4_cultural_expansion}"""

        else:  # full
            md = f"""## 🎵 Detailed explanation for 《{result.song.song}》

**Artist**: {result.song.artist} | **Album**: {result.song.album}
**Genre**: {result.song.genre} | **Year**: {result.song.year}

---

### 📖 Main explanation chain ({chain.path_type.value} path)

**1️⃣ User preference analysis**
{chain.step_1_user_analysis}

**2️⃣ Evidence matching**
{chain.step_2_evidence_matching}

**3️⃣ Expected benefit**
{chain.step_3_benefit_prediction}

**4️⃣ Deep interpretation**
{chain.step_4_cultural_expansion}

---

### 📊 Recommendation basis

- **Confidence**: {result.overall_confidence:.0%}
- **Validation**: {'✅ Passed' if result.validation_passed else '⚠️ Needs improvement'}
- **Evidence sources**: {', '.join(chain.evidence_sources)}
"""

            # Add alternative paths
            if result.alternative_chains:
                md += "\n### 🔀 Alternative explanation paths\n\n"
                for i, alt in enumerate(result.alternative_chains[:2], 1):
                    md += f"""**Path {i}: {alt.path_type.value}** (Confidence: {alt.confidence:.0%})
- {alt.step_2_evidence_matching}
- {alt.step_3_benefit_prediction}

"""

            return md
