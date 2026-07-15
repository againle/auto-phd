"""
套磁信模板管理器（Email Templates）

提供多种风格的套磁信模板，支持变量注入和自定义模板。

模板风格：
- academic:    学术严谨型（默认，适合顶级院校）
- concise:     简洁直接型（适合欧美教授）
- enthusiastic: 热情积极型（适合年轻AP）
- chinese:     中文版（适合华人教授）
"""

import logging
import re
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class _SafeDict(dict):
    """安全字典：缺失的键返回原占位符而非抛 KeyError"""

    def __missing__(self, key):
        return f"{{{key}}}"

# ============================================================
# 内置模板
# ============================================================

TEMPLATES: Dict[str, Dict[str, str]] = {
    # ── 学术严谨型 ──
    "academic": {
        "name": "学术严谨型",
        "description": "正式、专业，强调学术匹配和研究深度。适合顶级院校和资深教授。",
        "subject": "Prospective PhD Applicant - Interest in {paper_title}",
        "body": """Dear Professor {professor_name},

I am writing to express my keen interest in joining your research group at {professor_institution} as a PhD student. I have been closely following your work, and your paper "{paper_title}" particularly resonated with my research direction.

What impressed me most was {core_contribution}. This is a significant contribution that advances our understanding of this area. While reading your paper, I noticed that one of the unresolved questions you raised is {unresolved_question}.

This challenge aligns directly with my research interests. {my_connection}

In my previous research, I {my_publication}. Through this work and related projects, I have developed strong expertise in {my_skills}, which I believe would allow me to make meaningful contributions to addressing the questions you have identified.

I have attached my CV for your reference, and I would be honored to discuss how my background might complement your ongoing research. I understand you receive many inquiries, and I genuinely appreciate your time.

Looking forward to hearing from you.

Sincerely,
{my_name}
{my_position}
{my_institution}
Email: {my_email}
Website: {my_website}""",
    },

    # ── 简洁直接型 ──
    "concise": {
        "name": "简洁直接型",
        "description": "简短、高效，直奔主题。适合北美和欧洲教授，提高阅读率。",
        "subject": "PhD Application Inquiry - {paper_title}",
        "body": """Dear Professor {professor_name},

I'm {my_name}, a PhD candidate at {my_institution}. I read your recent work on {paper_title} and found {core_contribution} particularly compelling.

Your paper mentions {unresolved_question} as an open problem. This maps directly to my research — {my_connection}. I've also worked on {my_publication}, giving me hands-on experience with {my_skills}.

I'd love to explore joining your lab at {professor_institution}. My CV is attached. Would you have 15 minutes for a brief chat in the coming weeks?

Best,
{my_name}
{my_position}, {my_institution}
{my_email} | {my_website}""",
    },

    # ── 热情积极型 ──
    "enthusiastic": {
        "name": "热情积极型",
        "description": "充满热情，展现对新方向的渴望和主动性。适合年轻Assistant Professor。",
        "subject": "Excited to Join Your Lab! — PhD Applicant Inspired by {paper_title}",
        "body": """Dear Professor {professor_name},

I hope this email finds you well! I recently came across your paper "{paper_title}" and couldn't stop thinking about it — {core_contribution} is genuinely exciting and exactly the kind of research I want to pursue!

I was especially intrigued by the open question you raised: {unresolved_question}. This is something I've been actively thinking about, and I already have some initial ideas for how to approach it. {my_connection}

I have a solid foundation to build on — {my_publication}, and I'm proficient in {my_skills}. I'm eager to bring this experience to your group at {professor_institution} and tackle these exciting challenges together.

I would absolutely love to chat about potential opportunities! I've attached my CV, and I'm happy to prepare a brief research proposal if that would be helpful.

Thank you so much for your time — looking forward to connecting!

Warm regards,
{my_name}
{my_position}, {my_institution}
{my_email}
{my_website}""",
    },

    # ── 中文版（华人教授） ──
    "chinese": {
        "name": "中文版",
        "description": "适合发给华人教授，用中文表达更自然亲切。",
        "subject": "博士申请咨询 — 关于{paper_title}的研究",
        "body": """尊敬的{professor_name}老师：

您好！我是{my_name}，目前在{my_institution}攻读博士学位（预计{my_position}）。我一直密切关注您课题组的研究，尤其是您最近发表的《{paper_title}》一文，让我受益匪浅。

您在文中提出的{core_contribution}给我留下了深刻的印象。在阅读过程中，我注意到您提到的一个开放性问题——{unresolved_question}，这与我的研究兴趣高度契合。{my_connection}

在之前的研究中，我{my_publication}，积累了{my_skills}方面的扎实经验。我相信这些背景可以为解决您提出的问题提供帮助。

我真诚希望有机会加入您的课题组，继续深入这一领域的研究。随信附上我的简历，期待能得到您的指导和建议。

祝您工作顺利！

{my_name}
{my_position}，{my_institution}
邮箱：{my_email}
个人主页：{my_website}""",
    },
}


# ============================================================
# 模板管理器
# ============================================================

class EmailTemplateManager:
    """
    套磁信模板管理器。

    使用示例:
        mgr = EmailTemplateManager()
        email = mgr.render("academic", {
            "professor_name": "Dr. Smith",
            "professor_institution": "Stanford",
            ...
        })
        print(email["subject"])
        print(email["body"])
    """

    def __init__(self):
        self._templates = dict(TEMPLATES)  # 复制内置模板
        self._custom_templates: Dict[str, Dict[str, str]] = {}
        logger.info(f"模板管理器初始化: {len(self._templates)} 个内置模板")

    # --------------------------------------------------------
    # 模板管理
    # --------------------------------------------------------

    def get_template_names(self) -> List[str]:
        """返回所有可用模板名称"""
        return list(self._templates.keys())

    def get_template_info(self, name: str) -> Optional[Dict[str, str]]:
        """
        获取模板信息。

        Args:
            name: 模板名称

        Returns:
            {name, description, subject, body} 或 None
        """
        tmpl = self._templates.get(name)
        if not tmpl:
            logger.warning(f"模板不存在: {name}")
            return None
        return {
            "name": name,
            "label": tmpl.get("name", name),
            "description": tmpl.get("description", ""),
            "subject": tmpl.get("subject", ""),
            "body": tmpl.get("body", ""),
        }

    def list_templates(self) -> List[Dict[str, str]]:
        """列出所有模板及其描述"""
        return [
            {
                "name": name,
                "label": info["name"],
                "description": info["description"],
            }
            for name, info in self._templates.items()
        ]

    def add_custom_template(
        self,
        name: str,
        label: str,
        subject: str,
        body: str,
        description: str = "",
        overwrite: bool = False,
    ) -> bool:
        """
        添加自定义模板。

        Args:
            name: 模板唯一名称（用于 get_template）
            label: 显示名称
            subject: 邮件标题模板
            body: 邮件正文模板
            description: 模板描述
            overwrite: 是否覆盖同名模板

        Returns:
            True 成功，False 失败
        """
        if name in self._templates and not overwrite:
            logger.warning(f"模板 '{name}' 已存在，使用 overwrite=True 强制覆盖")
            return False

        self._templates[name] = {
            "name": label,
            "description": description,
            "subject": subject,
            "body": body,
        }
        self._custom_templates[name] = self._templates[name]

        logger.info(f"自定义模板已添加: {name} ({label})")
        return True

    def remove_template(self, name: str) -> bool:
        """删除模板（仅允许删除自定义模板）"""
        if name not in self._templates:
            return False
        if name in TEMPLATES:
            logger.warning(f"不能删除内置模板: {name}")
            return False

        del self._templates[name]
        self._custom_templates.pop(name, None)
        return True

    def reset_to_defaults(self) -> None:
        """重置为内置模板"""
        self._templates = dict(TEMPLATES)
        self._custom_templates.clear()
        logger.info("模板已重置为默认值")

    # --------------------------------------------------------
    # 渲染
    # --------------------------------------------------------

    def render(
        self,
        style: str = "academic",
        variables: Optional[Dict[str, str]] = None,
        validate: bool = True,
    ) -> Dict[str, str]:
        """
        渲染邮件模板。

        Args:
            style: 模板名称 ("academic", "concise", "enthusiastic", "chinese")
            variables: 变量字典
            validate: 是否验证必填变量

        Returns:
            {"subject": "...", "body": "..."}

        Raises:
            ValueError: 模板不存在
        """
        tmpl = self._templates.get(style)
        if not tmpl:
            available = ", ".join(self._templates.keys())
            raise ValueError(
                f"模板 '{style}' 不存在。可用: {available}"
            )

        variables = variables or {}

        if validate:
            self._validate_variables(tmpl, variables)

        # 使用 SafeDict 避免缺失变量时报 KeyError
        safe_vars = _SafeDict(variables)

        subject = tmpl["subject"].format_map(safe_vars)
        body = tmpl["body"].format_map(safe_vars)

        return {
            "subject": subject,
            "body": body,
            "style": style,
            "style_label": tmpl.get("name", style),
        }

    def render_all_styles(
        self,
        variables: Dict[str, str],
    ) -> Dict[str, Dict[str, str]]:
        """
        用所有模板渲染同一组变量，方便对比选择。

        Returns:
            {style_name: {subject, body, style_label}}
        """
        results = {}
        for style in self._templates:
            results[style] = self.render(style, variables, validate=False)
        return results

    # --------------------------------------------------------
    # 预览
    # --------------------------------------------------------

    def preview(
        self,
        style: str = "academic",
        variables: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        在终端中预览邮件。

        Args:
            style: 模板名称
            variables: 变量字典

        Returns:
            格式化的预览文本
        """
        if variables is None:
            variables = self._get_placeholder_variables()

        result = self.render(style, variables, validate=False)
        tmpl_info = self._templates.get(style, {})

        lines = [
            f"{'═' * 65}",
            f"  套磁信预览 — {tmpl_info.get('name', style)}",
            f"{'─' * 65}",
            f"  Subject: {result['subject']}",
            f"{'─' * 65}",
            result["body"],
            f"{'═' * 65}",
        ]
        return "\n".join(lines)

    def preview_all(self) -> str:
        """预览所有模板（使用占位符变量）"""
        variables = self._get_placeholder_variables()
        parts = []
        for style in self._templates:
            parts.append(self.preview(style, variables))
            parts.append("")
        return "\n".join(parts)

    # --------------------------------------------------------
    # 辅助
    # --------------------------------------------------------

    def _validate_variables(
        self, tmpl: Dict[str, str], variables: Dict[str, str]
    ) -> None:
        """验证必填变量是否都已提供"""
        combined = tmpl["subject"] + " " + tmpl["body"]
        # 匹配 {variable_name} 格式
        required = set(re.findall(r'\{(\w+)\}', combined))

        missing = required - set(variables.keys())
        if missing:
            logger.warning(
                f"模板变量未填充: {', '.join(sorted(missing))}"
            )

    @staticmethod
    def _get_placeholder_variables() -> Dict[str, str]:
        """返回占位符变量（用于预览）"""
        return {
            "professor_name": "Dr. Alice Smith",
            "professor_institution": "Stanford University",
            "paper_title": "Deep Learning for Protein Structure Prediction",
            "core_contribution": "the novel Transformer-based architecture that achieved SOTA accuracy",
            "unresolved_question": "scaling the method to multi-chain protein complexes",
            "my_connection": "I have been exploring graph neural networks for modeling protein-protein interactions, which could directly address the multi-chain challenge",
            "my_publication": "published a first-author paper at NeurIPS on graph-based molecular modeling",
            "my_skills": "PyTorch, molecular dynamics simulation, and protein structure analysis",
            "my_name": "San Zhang (张三)",
            "my_position": "PhD Candidate (expected 2027)",
            "my_institution": "Tsinghua University",
            "my_email": "san.zhang@example.edu",
            "my_website": "https://example.com/sanzhang",
        }


# ============================================================
# 便捷函数
# ============================================================

def get_template(style: str = "academic") -> Dict[str, str]:
    """
    便捷函数：获取指定风格的模板（无需实例化）。

    Args:
        style: 模板名称

    Returns:
        {name, description, subject, body}
    """
    mgr = EmailTemplateManager()
    return mgr.get_template_info(style) or {}


# ============================================================
# 自测入口
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )

    print(f"\n{'═' * 65}")
    print("  套磁信模板管理器 - 自测")
    print(f"{'═' * 65}")

    mgr = EmailTemplateManager()

    # 测试1: 列出所有模板
    print("\n[1] 可用模板:")
    for t in mgr.list_templates():
        print(f"    {t['name']:<15} — {t['label']}")
        print(f"    {'':15}   {t['description']}")

    # 测试2: 渲染单个模板
    print(f"\n[2] 渲染 academic 模板...")
    vars = mgr._get_placeholder_variables()
    result = mgr.render("academic", vars)
    print(f"    Subject: {result['subject'][:70]}...")
    print(f"    Body length: {len(result['body'])} chars")

    # 测试3: 预览 academic 模板
    print(f"\n[3] 预览 academic 模板:")
    preview = mgr.preview("academic", vars)
    print(preview)

    # 测试4: 自定义模板
    print(f"\n[4] 添加自定义模板...")
    mgr.add_custom_template(
        "custom_short",
        "超短型",
        "PhD Application — {professor_name}",
        "Dear {professor_name},\n\nI love your work on {paper_title}. Can I join your lab?\n\n- {my_name}",
        "极简版，一句话表达兴趣",
    )
    result_custom = mgr.render("custom_short", vars)
    print(f"    自定义模板: Subject='{result_custom['subject']}'")

    # 测试5: 渲染所有风格
    print(f"\n[5] 所有风格对比 (字数统计):")
    all_results = mgr.render_all_styles(vars)
    for style, r in all_results.items():
        body_words = len(r["body"].split())
        print(f"    {style:<15} — {r['style_label']:<10} — {body_words} 词")

    # 测试6: 验证缺失变量
    print(f"\n[6] 变量验证测试...")
    try:
        mgr.render("academic", {"professor_name": "Test"}, validate=True)
    except Exception as e:
        print(f"    验证通过（不会因缺失变量抛异常，使用 safe_substitute）")

    # 测试7: 中文模板
    print(f"\n[7] 中文模板预览:")
    print(mgr.preview("chinese", vars))

    print(f"\n✅ 自测完成")
