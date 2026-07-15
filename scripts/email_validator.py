"""
邮件内容验证器（Email Validator）

功能：
- 检查邮件是否包含所有必需元素
- 质量检查（长度、套话、语气）
- 可选 LLM 深度质量评估
- 生成详细检查报告

验证维度：
1. 必需要素：教授姓名、论文引用、研究问题、技能连接、行动号召
2. 质量：长度、套话检测、语气
3. 综合评分 0-100
"""

import logging
import re
from typing import Dict, List, Optional, Any, Tuple

logger = logging.getLogger(__name__)

# ============================================================
# 套话/模板感短语库
# ============================================================

CLICHES = [
    # 过度使用的开头
    r"I am very interested in your research",
    r"I am writing to express my (strong|keen|deep) interest",
    r"I have been following your work for (a long time|many years)",
    r"I am a big fan of your (work|research)",
    r"It is with great (interest|enthusiasm) that I",
    r"I hope this email finds you well",  # 过于正式

    # 空洞的赞美
    r"your (groundbreaking|pioneering|seminal|excellent|fascinating) work",
    r"I was (very|really|truly|deeply) impressed by",
    r"your work is (truly|very|extremely) (inspiring|impressive|remarkable)",
    r"I have always been passionate about",

    # 模糊的结尾
    r"I look forward to hearing from you (at your earliest convenience|soon)",
    r"I would be (honored|delighted|thrilled) to (join|hear from|work with)",
    r"Thank you for your (time and )?consideration",
    r"Any (feedback|response|reply) would be (greatly|highly|much) appreciated",

    # 不自信的表达
    r"I (was wondering|just wanted to|thought I would) (ask|check|reach out)",
    r"I (apologize|sorry) for (the|any) (intrusion|disturbance|unsolicited)",
    r"if (you have|there is|there are) any (openings|opportunities|positions)",

    # 过于自我
    r"I believe I (am|would be) (the perfect|an ideal|a great) (fit|candidate)",
    r"I am confident that I (can|will|would) (make|bring|contribute)",
]

# 积极指标（加分项）
POSITIVE_INDICATORS = [
    (r'\b(DOI|doi):\s*10\.\d{4,}', 10, "包含论文 DOI"),
    (r'\b(arXiv|arxiv):\s*\d{4}\.\d{4,}', 10, "包含 arXiv ID"),
    (r'(specific|specifically|in particular|notably)', 3, "具体化表达"),
    (r'(I notice[d]?|you mention|you raise[d]?|your paper (show|demonstrat|present|propose|introduce))', 5, "引用教授论文细节"),
    (r'(I (have|developed|built|created|implemented|trained|designed))', 5, "展示具体技能"),
    (r'(would (it be|you be)|could (we|I)|(happy|available|open) to)', 3, "自然的 Call to Action"),
    (r'\b(CV|resume|transcript)\b', 2, "提到附件"),
    (r'(discuss|chat|talk|meet|call|zoom)', 3, "提议具体行动"),
]

# 建议的 Call to Action 模式
CTA_PATTERNS = [
    r'chat\b',
    r'\bmeet(ing)?\b',
    r'\bdiscuss\b',
    r'Zoom\b',
    r'\bcall\b',
    r'\bconversation\b',
    r'open (to|for)',
    r'available',
    r'would (you|it) be',
    r'I would (love|like|appreciate)',
    r'(happy|glad|eager) to',
    r'schedule',
    r'opportunit',
]


# ============================================================
# 验证器
# ============================================================

class EmailValidator:
    """
    邮件内容验证器。

    使用示例:
        validator = EmailValidator()
        report = validator.validate(subject, body, "Dr. Smith")
        if report["passed"]:
            print("可以发送!")
        else:
            print("需要修改:", report["warnings"])
    """

    def __init__(self, llm_client=None):
        """
        Args:
            llm_client: 可选的 LLMClient，用于深度质量评估
        """
        self.llm = llm_client
        self._stats = {"validated": 0, "passed": 0, "failed": 0}

    # --------------------------------------------------------
    # 主入口
    # --------------------------------------------------------

    def validate(
        self,
        subject: str,
        body: str,
        professor_name: str = "",
        paper_title: str = "",
        deep_check: bool = False,
    ) -> Dict[str, Any]:
        """
        验证邮件内容。

        Args:
            subject: 邮件主题
            body: 邮件正文
            professor_name: 教授姓名（用于检查是否包含）
            paper_title: 论文标题（用于检查是否引用）
            deep_check: 是否使用 LLM 进行深度评估

        Returns:
            {
                "passed": bool,
                "score": int (0-100),
                "checks": {check_name: bool},
                "warnings": [str],
                "suggestions": [str],
                "stats": {word_count, sentence_count, ...},
            }
        """
        self._stats["validated"] += 1

        full_text = f"{subject}\n{body}"

        # 基础统计
        stats = self._compute_stats(subject, body)
        warnings: List[str] = []
        suggestions: List[str] = []

        # ── 必需要素检查 ──
        checks = {
            "has_subject": bool(subject.strip()),
            "has_professor_name": self._check_professor_name(body, professor_name),
            "has_paper_reference": self._check_paper_reference(full_text, paper_title),
            "has_research_question": self._check_research_question(body),
            "has_skills_connection": self._check_skills_connection(body),
            "has_call_to_action": self._check_call_to_action(body),
            "length_ok": 100 <= stats["word_count"] <= 350,
            "no_cliches": True,  # 先设为 True，后续更新
        }

        # 生成警告
        if not checks["has_subject"]:
            warnings.append("缺少邮件主题")
            suggestions.append("添加一个具体的主题行，包含论文关键词")
        if not checks["has_professor_name"]:
            warnings.append("未找到教授姓名")
            pname = professor_name or "[Name]"
            suggestions.append(f"确保在邮件开头使用 'Dear Professor {pname}'")
        if not checks["has_paper_reference"]:
            warnings.append("未明确引用教授的论文")
            ptitle = paper_title or "[Paper Title]"
            suggestions.append(f"在邮件中直接提到论文标题或具体贡献，例如 '{ptitle}'")
        if not checks["has_research_question"]:
            warnings.append("未提及具体的研究问题或开放挑战")
            suggestions.append("引用论文中提到的局限性、未来工作或开放问题")
        if not checks["has_skills_connection"]:
            warnings.append("未展示你的技能如何与教授的研究关联")
            suggestions.append("用 1-2 句话说明你过去的具体项目/论文如何能帮助解决教授的问题")
        if not checks["has_call_to_action"]:
            warnings.append("缺少明确的行动号召 (Call to Action)")
            suggestions.append("添加如 'Would you be open to a brief chat?' 或 'Are you recruiting PhD students for 2027?'")
        if not checks["length_ok"]:
            if stats["word_count"] < 100:
                warnings.append(f"邮件过短 ({stats['word_count']} 词)，建议 150-250 词")
                suggestions.append("扩展技能连接和研究问题部分，增加具体细节")
            else:
                warnings.append(f"邮件过长 ({stats['word_count']} 词)，建议 150-250 词")
                suggestions.append("精简开头和结尾的套话，聚焦核心内容")

        # ── 套话检测 ──
        cliche_count = 0
        detected_cliches = []
        for pattern in CLICHES:
            matches = re.findall(pattern, body, re.IGNORECASE)
            if matches:
                cliche_count += len(matches)
                detected_cliches.append(pattern)

        checks["no_cliches"] = cliche_count <= 2
        if cliche_count > 2:
            warnings.append(f"检测到 {cliche_count} 处套话/模板感表达")
            suggestions.append("用更具体、个性化的表达替换通用套话")

        # 更具体的套话警告
        if cliche_count >= 5:
            warnings.append(f"邮件包含过多模板化表达，可能被识别为批量生成的模板邮件")

        # ── 积极指标加分 ──
        bonus_score = 0
        for pattern, points, label in POSITIVE_INDICATORS:
            if re.search(pattern, full_text, re.IGNORECASE):
                bonus_score += points

        # ── 计算基础分 ──
        base_score = sum(12.5 if v else 0 for v in checks.values())  # 每项 12.5 分
        score = min(base_score + bonus_score, 100)

        # ── 深度检查（LLM） ──
        if deep_check and self.llm:
            llm_result = self._deep_check(subject, body, professor_name)
            if llm_result:
                # LLM 结果权重 30%
                score = int(score * 0.7 + llm_result.get("score", score) * 0.3)
                checks["deep_check"] = llm_result.get("passed", True)
                if llm_result.get("warnings"):
                    warnings.extend(llm_result["warnings"])
                if llm_result.get("suggestions"):
                    suggestions.extend(llm_result["suggestions"])

        # ── 最终判定 ──
        passed = score >= 60 and all(
            checks.get(k, True) for k in [
                "has_subject", "has_professor_name", "has_paper_reference",
                "has_call_to_action", "length_ok",
            ]
        )

        if passed:
            self._stats["passed"] += 1
        else:
            self._stats["failed"] += 1

        return {
            "passed": passed,
            "score": min(score, 100),
            "checks": checks,
            "warnings": warnings,
            "suggestions": suggestions[:5],  # 最多 5 条建议
            "stats": stats,
            "cliche_count": cliche_count,
            "bonus_score": bonus_score,
        }

    # --------------------------------------------------------
    # 各项检查
    # --------------------------------------------------------

    def _check_professor_name(self, body: str, name: str) -> bool:
        """检查是否包含教授姓名"""
        if not name:
            # 检查是否有任何 Dear Professor 格式
            return bool(re.search(
                r'Dear\s+(Prof(essor)?|Dr)\.?\s+\w+',
                body, re.IGNORECASE
            ))
        # 检查具体姓名（支持部分匹配）
        name_parts = name.lower().split()
        body_lower = body.lower()
        # 至少匹配到姓
        if name_parts:
            last_name = name_parts[-1]
            if last_name in body_lower:
                return True
        return name.lower() in body_lower

    def _check_paper_reference(self, text: str, paper_title: str) -> bool:
        """检查是否引用了论文"""
        # 有引号包围的标题
        if re.search(r'"([^"]{10,})"', text):
            return True
        # 有具体论文标题关键词
        if paper_title:
            # 取标题中较长的词（>5字符）作为关键词
            keywords = [w for w in paper_title.split() if len(w) > 5]
            if keywords:
                text_lower = text.lower()
                matches = sum(1 for kw in keywords if kw.lower() in text_lower)
                if matches >= 2:
                    return True
        # 有论文引用标记
        if re.search(r'(your paper|your work on|your recent|your study|you (show|demonstrat|propos))', text, re.IGNORECASE):
            return True
        return False

    def _check_research_question(self, body: str) -> bool:
        """检查是否提到具体研究问题"""
        patterns = [
            r'(open (question|problem|challenge))',
            r'(limitation|limitations)',
            r'(future work|future direction)',
            r'(unresolved|unaddressed|unexplored)',
            r'(you (mention|raise|identif|highlight|point out))',
            r'(gap|bottleneck|shortcoming)',
            r'(remains (unclear|unknown|challenging|difficult|open))',
            r'could be (extended|improved|applied|adapted)',
            r'(would be (interesting|valuable|useful) to)',
            r'(address|tackle|solve|overcome) (this|the|these)',
        ]
        return any(re.search(p, body, re.IGNORECASE) for p in patterns)

    def _check_skills_connection(self, body: str) -> bool:
        """检查是否展示了技能与研究的关联"""
        patterns = [
            r'(my (experience|background|work|research|expertise|skill))',
            r'(I (have|developed|built|trained|worked|implemented))',
            r'(my (previous|past|recent) (project|paper|work|publication))',
            r'(I (can|could|would) (bring|contribute|apply|leverage|use))',
            r'(this (aligns|connects|relates|maps) (to|with) my)',
            r'(I (have|possess) (experience|expertise) in)',
            r'(my (technical|research) (skills|background) (in|include))',
        ]
        return any(re.search(p, body, re.IGNORECASE) for p in patterns)

    def _check_call_to_action(self, body: str) -> bool:
        """检查是否有清晰的行动号召"""
        return any(re.search(p, body, re.IGNORECASE) for p in CTA_PATTERNS)

    # --------------------------------------------------------
    # 统计
    # --------------------------------------------------------

    def _compute_stats(self, subject: str, body: str) -> Dict[str, int]:
        """计算邮件统计信息"""
        body_words = len(body.split())
        subject_words = len(subject.split())
        sentences = len(re.findall(r'[.!?]+', body))
        paragraphs = len([p for p in body.split('\n\n') if p.strip()])

        return {
            "word_count": body_words,
            "subject_word_count": subject_words,
            "sentence_count": max(sentences, 1),
            "paragraph_count": max(paragraphs, 1),
            "reading_time_seconds": max(int(body_words / 3.3), 5),  # ~200 wpm
        }

    # --------------------------------------------------------
    # LLM 深度检查（可选）
    # --------------------------------------------------------

    def _deep_check(
        self, subject: str, body: str, professor_name: str
    ) -> Optional[Dict]:
        """使用 LLM 进行深度质量评估"""
        if not self.llm:
            return None

        prompt = f"""You are an expert academic email reviewer. Evaluate this cold email from a PhD applicant to a professor.

## Email
Subject: {subject}

{body}

## Professor
Name: {professor_name or "Unknown"}

## Evaluation Criteria
1. Does it feel personalized (not templated)?
2. Is the research connection specific and credible?
3. Is the tone appropriate (confident but not arrogant)?
4. Would a busy professor actually read and respond?

Return ONLY a JSON object:
{{
    "passed": true/false,
    "score": 0-100,
    "tone_assessment": "one sentence",
    "warnings": ["issue1", "issue2"],
    "suggestions": ["improvement1", "improvement2"]
}}
"""

        try:
            response = self.llm.call(
                messages=[{"role": "user", "content": prompt}],
                task_type=TaskType.GENERAL,
            )
            import json
            cleaned = response.strip()
            if cleaned.startswith("```"):
                lines = cleaned.split("\n")
                cleaned = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else cleaned
            return json.loads(cleaned)
        except Exception as e:
            logger.warning(f"深度检查失败: {e}")
            return None

    # --------------------------------------------------------
    # 报告格式化
    # --------------------------------------------------------

    def format_report(self, report: Dict[str, Any]) -> str:
        """将验证报告格式化为可读文本"""
        lines = [
            f"{'═' * 55}",
            f"  邮件验证报告",
            f"{'─' * 55}",
            f"  结果: {'✅ 通过' if report['passed'] else '❌ 未通过'}",
            f"  评分: {report['score']}/100",
            f"",
            f"  📊 统计:",
            f"     词数: {report['stats']['word_count']}",
            f"     句数: {report['stats']['sentence_count']}",
            f"     段数: {report['stats']['paragraph_count']}",
            f"     阅读时间: ~{report['stats']['reading_time_seconds']}s",
            f"",
            f"  ✅ 检查项:",
        ]

        icon_map = {True: "✅", False: "❌"}
        check_labels = {
            "has_subject": "邮件主题",
            "has_professor_name": "教授姓名",
            "has_paper_reference": "论文引用",
            "has_research_question": "研究问题",
            "has_skills_connection": "技能关联",
            "has_call_to_action": "行动号召",
            "length_ok": "长度合理",
            "no_cliches": "无套话",
        }
        for check, label in check_labels.items():
            passed = report["checks"].get(check, False)
            icon = icon_map[passed]
            lines.append(f"     {icon} {label}")

        if report.get("cliche_count", 0) > 0:
            lines.append(f"")
            lines.append(f"  ⚠️ 检测到 {report['cliche_count']} 处套话")

        if report["warnings"]:
            lines.append(f"")
            lines.append(f"  ⚠️ 警告:")
            for w in report["warnings"]:
                lines.append(f"     • {w}")

        if report["suggestions"]:
            lines.append(f"")
            lines.append(f"  💡 建议:")
            for s in report["suggestions"]:
                lines.append(f"     • {s}")

        lines.append(f"{'═' * 55}")
        return "\n".join(lines)

    def get_stats(self) -> Dict[str, int]:
        return dict(self._stats)


# ============================================================
# 便捷函数
# ============================================================

def validate_email(
    subject: str,
    body: str,
    professor_name: str = "",
    paper_title: str = "",
) -> Dict[str, Any]:
    """
    便捷函数：验证邮件（无 LLM 深度检查）。

    Args:
        subject: 邮件主题
        body: 邮件正文
        professor_name: 教授姓名
        paper_title: 论文标题

    Returns:
        验证报告字典
    """
    validator = EmailValidator()
    return validator.validate(subject, body, professor_name, paper_title)


# ============================================================
# 自测入口
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )

    print(f"\n{'═' * 55}")
    print("  邮件验证器 - 自测")
    print(f"{'═' * 55}")

    validator = EmailValidator()

    # 测试1: 好邮件
    print("\n[1] 验证优质邮件...")
    good_subject = "PhD Application — Protein Structure Prediction with Graph Neural Networks"
    good_body = """Dear Professor Smith,

I recently read your paper "Deep Learning for Protein Structure Prediction" in Nature Methods, and your GNN-based approach to multi-chain complexes was particularly insightful.

You mention that scaling to protein-protein interactions remains an open challenge. This connects directly to my work — I developed a graph attention network for molecular dynamics that reduced simulation time by 40% (published at NeurIPS 2025). I believe similar graph-based techniques could address the multi-chain problem you identified.

Would you be open to a brief Zoom chat about potential PhD opportunities in your lab starting Fall 2027?

Best,
San Zhang
PhD Candidate, Tsinghua University"""

    report1 = validator.validate(good_subject, good_body, "Dr. Smith", "Deep Learning for Protein Structure Prediction")
    print(validator.format_report(report1))

    # 测试2: 差邮件（套话多）
    print("\n[2] 验证模板化邮件...")
    bad_subject = "Research Interest"
    bad_body = """Dear Professor,

I am writing to express my strong interest in your groundbreaking research. I have been following your work for a long time and I was truly impressed by your pioneering contributions. I am a big fan of your research.

I believe I would be an ideal candidate for your lab. I have always been passionate about machine learning and I am confident that I can make significant contributions.

I look forward to hearing from you at your earliest convenience. Thank you for your time and consideration.

Sincerely,
Student"""

    report2 = validator.validate(bad_subject, bad_body, "Professor")
    print(validator.format_report(report2))

    # 测试3: 边界情况
    print("\n[3] 边界情况测试...")
    # 过短
    report3 = validator.validate("Hi", "Dear Dr. X, I like your paper. Can I join?")
    print(f"    过短邮件: {'通过' if report3['passed'] else '未通过'} ({report3['score']}/100)")

    # 缺少关键元素
    report4 = validator.validate(
        "Hello",
        "Dear Prof. A, I am a student. I want to do a PhD. Thanks.",
        "Prof. A",
    )
    print(f"    缺失元素: {'通过' if report4['passed'] else '未通过'} ({report4['score']}/100)")

    print(f"\n[统计] 验证: {validator.get_stats()['validated']}, "
          f"通过: {validator.get_stats()['passed']}, "
          f"失败: {validator.get_stats()['failed']}")

    print(f"\n✅ 自测完成")
