"""
个人档案解析器

功能：
- 加载和验证 JSON 档案文件
- 提取研究兴趣关键词（带权重）
- 提取技术技能列表
- 生成档案摘要（用于 LLM Prompt 注入）
- 使用 pydantic 进行数据验证
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field, ValidationError, field_validator

logger = logging.getLogger(__name__)


# ============================================================
# Pydantic 数据模型
# ============================================================

class PersonalInfo(BaseModel):
    """个人信息"""
    name: str
    email: str
    current_institution: str
    current_position: str
    website: str = ""
    citizenship: str = ""
    google_scholar: str = ""


class ResearchInterest(BaseModel):
    """研究兴趣"""
    topic: str
    keywords: List[str]
    priority: int = Field(default=3, ge=1, le=5)

    @field_validator("keywords")
    @classmethod
    def keywords_not_empty(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("keywords 不能为空")
        return [kw.strip().lower() for kw in v]


class TechnicalSkill(BaseModel):
    """技术技能"""
    category: str
    skills: List[str]
    proficiency: int = Field(default=3, ge=1, le=5)


class Publication(BaseModel):
    """出版物"""
    title: str
    authors: List[str]
    journal: str
    year: int
    doi: str = ""
    contribution_summary: str = ""
    url: str = ""


class Education(BaseModel):
    """教育背景"""
    institution: str
    degree: str
    year_start: int
    year_end: Optional[int] = None
    advisor: str = ""
    gpa: str = ""


class TargetPreferences(BaseModel):
    """目标偏好"""
    locations: List[str] = Field(default_factory=list)
    school_ranks: List[str] = Field(default_factory=list)
    professor_ranks: List[str] = Field(default_factory=list)
    min_quality_score: int = Field(default=70, ge=0, le=100)
    max_applications: int = Field(default=50, ge=1)


class AdditionalInfo(BaseModel):
    """附加信息"""
    awards: List[str] = Field(default_factory=list)
    languages: Dict[str, str] = Field(default_factory=dict)
    research_statement: str = ""
    teaching_experience: str = ""
    references: List[Dict[str, str]] = Field(default_factory=list)


class Profile(BaseModel):
    """完整的个人档案"""
    personal_info: PersonalInfo
    research_interests: List[ResearchInterest]
    technical_skills: List[TechnicalSkill]
    publications: List[Publication]
    education: List[Education]
    target_preferences: TargetPreferences
    additional_info: AdditionalInfo = Field(default_factory=AdditionalInfo)


# ============================================================
# 档案解析器
# ============================================================

class ProfileParser:
    """
    个人档案解析器。

    使用示例:
        parser = ProfileParser()
        keywords = parser.get_research_keywords()
        summary = parser.get_profile_summary()
    """

    def __init__(self, profile_path: str = "profiles/my_profile_template.json"):
        """
        初始化解析器。

        Args:
            profile_path: 档案 JSON 文件路径
        """
        self.profile_path = Path(profile_path)
        self.profile: Profile = self.load()
        logger.info(f"档案已加载: {self.profile.personal_info.name}")

    # --------------------------------------------------------
    # 加载与验证
    # --------------------------------------------------------

    def load(self) -> Profile:
        """
        加载并验证档案 JSON 文件。

        Returns:
            验证后的 Profile 对象

        Raises:
            FileNotFoundError: 档案文件不存在
            ValueError: JSON 格式错误或缺少必填字段
        """
        if not self.profile_path.exists():
            raise FileNotFoundError(
                f"档案文件不存在: {self.profile_path.absolute()}\n"
                f"请参考 profiles/my_profile_template.json 创建你的档案文件。"
            )

        try:
            with open(self.profile_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"档案文件 JSON 格式错误: {e}") from e

        try:
            profile = Profile(**raw_data)
        except ValidationError as e:
            error_details = []
            for err in e.errors():
                field = " → ".join(str(loc) for loc in err["loc"])
                error_details.append(f"  - {field}: {err['msg']}")
            raise ValueError(
                f"档案验证失败:\n" + "\n".join(error_details)
            ) from e

        return profile

    def validate_profile(self) -> bool:
        """
        验证档案完整性。

        检查所有必填字段是否填写（非空）。

        Returns:
            True 表示通过验证
        """
        issues = []

        # 检查个人信息
        pi = self.profile.personal_info
        if not pi.name:
            issues.append("姓名未填写")
        if not pi.email:
            issues.append("邮箱未填写")

        # 检查研究兴趣
        if not self.profile.research_interests:
            issues.append("研究兴趣为空")

        # 检查出版物
        if not self.profile.publications:
            issues.append("出版物列表为空")

        # 检查教育背景
        if not self.profile.education:
            issues.append("教育背景为空")

        if issues:
            logger.warning(f"档案验证发现问题:\n" + "\n".join(f"  - {i}" for i in issues))
            return False

        logger.info("档案验证通过")
        return True

    # --------------------------------------------------------
    # 数据提取
    # --------------------------------------------------------

    def get_research_keywords(self) -> List[str]:
        """
        返回所有研究关键词（去重、小写）。

        Returns:
            关键词列表
        """
        keywords_set = set()
        for interest in self.profile.research_interests:
            for kw in interest.keywords:
                keywords_set.add(kw.lower())
        return sorted(keywords_set)

    def get_weighted_keywords(self) -> Dict[str, int]:
        """
        返回带权重的关键词字典。

        权重 = 研究兴趣优先级 (1-5)，同一关键词取最高优先级。

        Returns:
            {keyword: priority} 字典
        """
        weighted: Dict[str, int] = {}
        for interest in self.profile.research_interests:
            for kw in interest.keywords:
                kw_lower = kw.lower()
                current = weighted.get(kw_lower, 0)
                weighted[kw_lower] = max(current, interest.priority)
        return dict(sorted(weighted.items(), key=lambda x: x[1], reverse=True))

    def get_skills(self) -> List[str]:
        """
        返回所有技术技能（去重）。

        Returns:
            技能列表
        """
        skills_set = set()
        for skill_group in self.profile.technical_skills:
            for skill in skill_group.skills:
                skills_set.add(skill)
        return sorted(skills_set)

    def get_skills_by_category(self) -> Dict[str, List[str]]:
        """
        按类别返回技能。

        Returns:
            {category: [skill1, skill2, ...]}
        """
        return {
            skill_group.category: list(skill_group.skills)
            for skill_group in self.profile.technical_skills
        }

    # --------------------------------------------------------
    # 摘要生成
    # --------------------------------------------------------

    def get_profile_summary(self) -> str:
        """
        生成完整的个人档案摘要文本。

        用于注入 LLM Prompt，帮助模型了解申请人背景。

        Returns:
            摘要文本
        """
        pi = self.profile.personal_info
        pref = self.profile.target_preferences

        lines = [
            f"## 申请人档案摘要",
            f"",
            f"### 基本信息",
            f"- 姓名: {pi.name}",
            f"- 邮箱: {pi.email}",
            f"- 当前机构: {pi.current_institution}",
            f"- 当前职位: {pi.current_position}",
            f"- 国籍: {pi.citizenship}",
        ]

        if pi.website:
            lines.append(f"- 个人网站: {pi.website}")

        # 教育背景
        lines.append(f"\n### 教育背景")
        for edu in self.profile.education:
            end_year = str(edu.year_end) if edu.year_end else "至今"
            lines.append(f"- {edu.degree}, {edu.institution} ({edu.year_start}-{end_year})")
            if edu.advisor:
                lines.append(f"  导师: {edu.advisor}")

        # 研究兴趣
        lines.append(f"\n### 研究兴趣")
        for interest in sorted(
            self.profile.research_interests,
            key=lambda x: x.priority,
            reverse=True,
        ):
            lines.append(
                f"- [{interest.priority}/5] {interest.topic}: "
                f"{', '.join(interest.keywords)}"
            )

        # 技术技能
        lines.append(f"\n### 技术技能")
        for skill_group in sorted(
            self.profile.technical_skills,
            key=lambda x: x.proficiency,
            reverse=True,
        ):
            lines.append(
                f"- [{skill_group.proficiency}/5] {skill_group.category}: "
                f"{', '.join(skill_group.skills)}"
            )

        # 目标偏好
        lines.append(f"\n### 申请偏好")
        lines.append(f"- 目标地区: {', '.join(pref.locations)}")
        lines.append(f"- 学校排名: {', '.join(pref.school_ranks)}")
        lines.append(f"- 教授职称: {', '.join(pref.professor_ranks)}")
        lines.append(f"- 最低质量分: {pref.min_quality_score}")

        # 附加信息
        add = self.profile.additional_info
        if add.research_statement:
            lines.append(f"\n### 研究陈述")
            lines.append(add.research_statement)

        if add.awards:
            lines.append(f"\n### 获奖")
            for award in add.awards:
                lines.append(f"- {award}")

        if add.languages:
            lines.append(f"\n### 语言能力")
            for lang, level in add.languages.items():
                lines.append(f"- {lang}: {level}")

        return "\n".join(lines)

    def get_publication_summary(self) -> str:
        """
        生成代表作摘要。

        Returns:
            摘要文本
        """
        if not self.profile.publications:
            return "（暂无出版物记录）"

        lines = [f"## 代表作 ({len(self.profile.publications)}篇)\n"]
        for i, pub in enumerate(self.profile.publications, 1):
            lines.append(f"### {i}. {pub.title}")
            lines.append(f"- 作者: {', '.join(pub.authors)}")
            lines.append(f"- 发表: {pub.journal} ({pub.year})")
            if pub.doi:
                lines.append(f"- DOI: {pub.doi}")
            if pub.contribution_summary:
                lines.append(f"- 贡献: {pub.contribution_summary}")
            lines.append("")

        return "\n".join(lines)

    def get_short_summary(self, max_length: int = 300) -> str:
        """
        生成简短摘要（用于邮件等场景）。

        Args:
            max_length: 最大字符数

        Returns:
            简短摘要
        """
        pi = self.profile.personal_info
        edu = self.profile.education[0] if self.profile.education else None

        parts = [
            f"{pi.name}, {pi.current_position} at {pi.current_institution}.",
        ]

        if edu:
            parts.append(f"PhD candidate in {edu.degree} (advisor: {edu.advisor})."
                         if edu.advisor else
                         f"PhD candidate in {edu.degree}.")

        topics = [ri.topic for ri in sorted(
            self.profile.research_interests,
            key=lambda x: x.priority,
            reverse=True,
        )]
        if topics:
            parts.append(f"Research interests: {'; '.join(topics)}.")

        summary = " ".join(parts)
        if len(summary) > max_length:
            summary = summary[: max_length - 3] + "..."

        return summary

    # --------------------------------------------------------
    # 打印
    # --------------------------------------------------------

    def print_profile_info(self) -> None:
        """友好地打印档案信息到控制台"""
        print(self.get_profile_summary())
        print()
        print(self.get_publication_summary())


# ============================================================
# 自测入口
# ============================================================

if __name__ == "__main__":
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
    )

    print("=" * 60)
    print("  个人档案解析器 - 自测")
    print("=" * 60)

    try:
        parser = ProfileParser()

        print(f"\n✅ 档案加载成功: {parser.profile.personal_info.name}")
        print(f"   邮箱: {parser.profile.personal_info.email}")

        print(f"\n📚 研究关键词 ({len(parser.get_research_keywords())}个):")
        for kw in parser.get_research_keywords():
            print(f"   - {kw}")

        print(f"\n⭐ 加权关键词 (Top 5):")
        for kw, weight in list(parser.get_weighted_keywords().items())[:5]:
            print(f"   - [{weight}] {kw}")

        print(f"\n🛠️ 技术技能 ({len(parser.get_skills())}项):")
        for cat, skills in parser.get_skills_by_category().items():
            print(f"   - {cat}: {', '.join(skills)}")

        print(f"\n📝 简短摘要:")
        print(f"   {parser.get_short_summary()}")

        print(f"\n🔍 验证结果: {'✅ 通过' if parser.validate_profile() else '⚠️ 存在问题'}")

    except FileNotFoundError as e:
        print(f"⚠️ {e}")
        print("请将 profiles/my_profile_template.json 复制为你的档案文件。")
    except Exception as e:
        print(f"❌ 错误: {e}")
