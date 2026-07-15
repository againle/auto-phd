"""
邮件发送器（Email Sender）

功能：
- 列出所有待发送的教授
- 逐个预览 / 编辑 / 确认发送
- SMTP 发送（支持附件 + BCC 密送）
- 试运行模式 (dry-run)
- 发送日志 + 状态更新

状态流转:
  paper_reading_completed → ready_for_review → sent / send_failed
"""

import json
import logging
import smtplib
import subprocess
import sys
import time
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

from scripts.utils import load_config, load_json, save_json, ensure_directory

logger = logging.getLogger(__name__)

# ============================================================
# 邮件发送器
# ============================================================

class EmailSender:
    """
    邮件发送器 — 审核后发送套磁信。

    使用示例:
        sender = EmailSender()
        ready = sender.list_ready_professors()
        sender.batch_send(ready, dry_run=True)  # 试运行
    """

    def __init__(self, config_path: str = "config.yaml"):
        """
        Args:
            config_path: 配置文件路径
        """
        config = load_config(config_path)
        email_conf = config.get("email", {})

        self.smtp_server = email_conf.get("smtp_server", "smtp.gmail.com")
        self.smtp_port = email_conf.get("smtp_port", 587)
        self.sender_email = email_conf.get("sender_email", "")
        self.sender_password = email_conf.get("sender_password", "")
        self.bcc_self = email_conf.get("bcc_self", True)
        self.attachment_cv = email_conf.get("attachment_cv", "profiles/CV.pdf")

        # 发送日志
        self._log_file = Path("logs/email_sent.log")
        ensure_directory("logs")

        # 统计
        self._stats = {"sent": 0, "failed": 0, "skipped": 0}

        logger.info(
            f"邮件发送器初始化: {self.smtp_server}:{self.smtp_port} "
            f"(发送人: {self.sender_email})"
        )

    # --------------------------------------------------------
    # 列出待发送教授
    # --------------------------------------------------------

    def list_ready_professors(self, base_dir: str = "professors") -> List[Path]:
        """
        列出所有准备发送的教授文件夹。

        条件: info.json 存在 且 status 为 'paper_reading_completed' 或 'ready_for_review'
              且 drafts/ 目录存在

        Args:
            base_dir: 教授文件夹根目录

        Returns:
            教授文件夹 Path 列表
        """
        base = Path(base_dir)
        if not base.exists():
            return []

        ready = []
        for prof_dir in sorted(base.iterdir()):
            if not prof_dir.is_dir():
                continue

            info_path = prof_dir / "info.json"
            drafts_dir = prof_dir / "drafts"

            if not info_path.exists():
                continue

            try:
                info = load_json(str(info_path))
                status = info.get("status", "")

                if status in ("paper_reading_completed", "ready_for_review") and drafts_dir.exists():
                    ready.append(prof_dir)
            except Exception as e:
                logger.debug(f"跳过 {prof_dir.name}: {e}")
                continue

        logger.info(f"找到 {len(ready)} 位待发送教授")
        return ready

    # --------------------------------------------------------
    # 预览
    # --------------------------------------------------------

    def preview_email(self, professor_folder: Path) -> Optional[Dict[str, str]]:
        """
        预览教授的邮件草稿。

        优先使用推荐的版本（metadata.json），否则取第一个。

        Args:
            professor_folder: 教授文件夹路径

        Returns:
            {subject, body, version, draft_file} 或 None
        """
        drafts_dir = professor_folder / "drafts"
        if not drafts_dir.exists():
            logger.warning(f"草稿目录不存在: {drafts_dir}")
            return None

        # 尝试从 metadata 获取推荐版本
        meta_path = drafts_dir / "metadata.json"
        recommended = ""
        if meta_path.exists():
            try:
                meta = load_json(str(meta_path))
                recommended = meta.get("recommended_version", "")
            except Exception:
                pass

        # 查找草稿文件
        draft_files = sorted(drafts_dir.glob("v*_*.md"))
        if not draft_files:
            logger.warning(f"无草稿文件: {drafts_dir}")
            return None

        # 选择草稿：推荐版本 > 第一个
        chosen = None
        if recommended:
            for df in draft_files:
                if recommended in df.stem:
                    chosen = df
                    break
        if chosen is None:
            chosen = draft_files[0]

        content = chosen.read_text(encoding="utf-8")
        subject, body = self._parse_draft(content)

        return {
            "subject": subject,
            "body": body,
            "version": chosen.stem,
            "draft_file": str(chosen),
        }

    # --------------------------------------------------------
    # 编辑
    # --------------------------------------------------------

    def edit_email(self, professor_folder: Path) -> bool:
        """
        在 VS Code（或系统默认编辑器）中打开邮件草稿。

        Args:
            professor_folder: 教授文件夹

        Returns:
            True 成功打开编辑器
        """
        drafts_dir = professor_folder / "drafts"
        draft_files = sorted(drafts_dir.glob("v*_*.md"))
        if not draft_files:
            return False

        draft_file = draft_files[0]  # 编辑第一个

        try:
            # 尝试在 VS Code 中打开
            subprocess.run(
                ["code", str(draft_file)],
                check=False,
                capture_output=True,
                timeout=5,
            )
            print(f"  📝 已在编辑器中打开: {draft_file.name}")
            print(f"  💡 修改后保存文件，然后回到此处继续。")
            return True
        except Exception:
            # 回退：用系统默认程序打开
            try:
                if sys.platform == "win32":
                    subprocess.run(["start", str(draft_file)], shell=True, check=False)
                else:
                    subprocess.run(["open", str(draft_file)], check=False)
                return True
            except Exception as e:
                logger.warning(f"无法打开编辑器: {e}")
                return False

    # --------------------------------------------------------
    # 发送
    # --------------------------------------------------------

    def send_email(
        self,
        professor_folder: Path,
        dry_run: bool = False,
    ) -> bool:
        """
        发送邮件给一位教授。

        Args:
            professor_folder: 教授文件夹
            dry_run: 试运行（只显示信息，不实际发送）

        Returns:
            True 成功发送
        """
        prof_dir = Path(professor_folder)
        name = prof_dir.name

        # 读取教授信息
        info_path = prof_dir / "info.json"
        if not info_path.exists():
            logger.error(f"info.json 不存在: {info_path}")
            return False

        info = load_json(str(info_path))
        prof_name = info.get("name", name)
        prof_email = info.get("email", "")

        if not prof_email:
            logger.warning(f"教授 {prof_name} 无邮箱地址，跳过")
            self._stats["skipped"] += 1
            return False

        # 读取邮件内容
        preview = self.preview_email(prof_dir)
        if not preview:
            logger.error(f"无法读取 {name} 的邮件草稿")
            return False

        subject = preview["subject"]
        body = preview["body"]

        # 检查发送人凭证
        if not self.sender_email or "YOUR_" in self.sender_email:
            logger.error("发件人邮箱未配置")
            print("  ❌ 请在 config.yaml 中设置 email.sender_email")
            return False

        # ── 试运行模式 ──
        if dry_run:
            print(f"\n  {'─' * 50}")
            print(f"  🧪 试运行模式 — 不会实际发送")
            print(f"  {'─' * 50}")
            print(f"  收件人: {prof_name} <{prof_email}>")
            print(f"  发件人: {self.sender_email}")
            print(f"  主题:   {subject}")
            print(f"  正文:   {body[:150]}...")
            if self.attachment_cv and Path(self.attachment_cv).exists():
                print(f"  附件:   {self.attachment_cv}")
            print(f"  BCC:    {'✅ 密送给自己' if self.bcc_self else '❌'}")
            print(f"  {'─' * 50}")

            self._update_status(prof_dir, "ready_for_review", "试运行审查通过")
            self._log_send(prof_name, prof_email, subject, "dry_run")
            return True

        # ── 实际发送 ──
        print(f"\n  ⏳ 正在发送给 {prof_name} ({prof_email})...")

        try:
            attachments = []
            cv_path = Path(self.attachment_cv)
            if cv_path.exists():
                attachments.append(str(cv_path))

            success = self._send_via_smtp(
                to_email=prof_email,
                subject=subject,
                body=body,
                bcc=self.sender_email if self.bcc_self else None,
                attachments=attachments,
            )

            if success:
                self._stats["sent"] += 1
                self._update_status(prof_dir, "sent", f"邮件已发送至 {prof_email}")
                self._log_send(prof_name, prof_email, subject, "success")

                print(f"  ✅ 发送成功！→ {prof_name}")
                return True
            else:
                self._stats["failed"] += 1
                self._update_status(prof_dir, "send_failed", "SMTP 发送失败")
                self._log_send(prof_name, prof_email, subject, "failed")

                print(f"  ❌ 发送失败 — 请检查 SMTP 配置")
                return False

        except Exception as e:
            self._stats["failed"] += 1
            self._update_status(prof_dir, "send_failed", str(e)[:100])
            self._log_send(prof_name, prof_email, subject, "error", str(e))

            logger.error(f"发送异常 {prof_name}: {e}")
            print(f"  ❌ 发送异常: {e}")
            return False

    # --------------------------------------------------------
    # 批量发送（交互式）
    # --------------------------------------------------------

    def batch_send(
        self,
        professor_folders: List[Path],
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """
        批量发送邮件（逐个确认）。

        Args:
            professor_folders: 教授文件夹列表
            dry_run: 试运行

        Returns:
            {total, sent, failed, skipped, details: [...]}
        """
        if not professor_folders:
            print("  📭 没有待发送的教授。")
            return {"total": 0, "sent": 0, "failed": 0, "skipped": 0, "details": []}

        total = len(professor_folders)
        print(f"\n{'═' * 55}")
        print(f"  📬 发现 {total} 位教授待发送")
        print(f"{'═' * 55}")

        if dry_run:
            print(f"  🧪 试运行模式 — 不会实际发送邮件\n")

        details = []
        idx = 0

        while idx < total:
            prof_dir = professor_folders[idx]

            # 显示信息
            info = load_json(str(prof_dir / "info.json"))
            prof_name = info.get("name", prof_dir.name)
            inst = info.get("institution", "?")

            preview = self.preview_email(prof_dir)
            if not preview:
                print(f"\n  ⚠️ {prof_name}: 无草稿，跳过")
                details.append({"name": prof_name, "status": "skipped", "reason": "无草稿"})
                idx += 1
                continue

            print(f"\n  {'─' * 50}")
            print(f"  [{idx + 1}/{total}] {prof_name}")
            print(f"  机构: {inst}")
            print(f"  主题: {preview['subject'][:70]}")
            print(f"  草稿: {preview['version']}")
            print(f"  {'─' * 50}")

            # 用户操作
            action = input(
                f"  操作: [s]发送 [p]预览全文 [e]编辑 [n]跳过 [q]退出: "
            ).strip().lower()

            if action == "q":
                print("  👋 退出发送。")
                break
            elif action == "p":
                # 显示全文
                print(f"\n{'─' * 55}")
                print(f"Subject: {preview['subject']}")
                print(f"{'─' * 55}")
                print(preview["body"])
                print(f"{'─' * 55}")

                confirm = input("  确认发送? [s]发送 [n]跳过: ").strip().lower()
                if confirm == "s":
                    success = self.send_email(prof_dir, dry_run=dry_run)
                    details.append({"name": prof_name, "status": "sent" if success else "failed"})
                else:
                    details.append({"name": prof_name, "status": "skipped"})
                idx += 1

            elif action == "e":
                self.edit_email(prof_dir)
                input("  按 Enter 继续...")

            elif action == "s":
                success = self.send_email(prof_dir, dry_run=dry_run)
                details.append({"name": prof_name, "status": "sent" if success else "failed"})
                idx += 1

            elif action == "n":
                self._stats["skipped"] += 1
                details.append({"name": prof_name, "status": "skipped"})
                idx += 1
            else:
                print("  ❓ 无效输入，请重试。")

        # 汇总
        sent = sum(1 for d in details if d["status"] == "sent")
        failed = sum(1 for d in details if d["status"] == "failed")
        skipped = sum(1 for d in details if d["status"] == "skipped")

        print(f"\n{'═' * 55}")
        print(f"  📊 发送汇总")
        print(f"  {'─' * 55}")
        print(f"  已发送: {sent} ✅")
        print(f"  失败:   {failed} ❌")
        print(f"  跳过:   {skipped} ⏭️")
        print(f"  {'═' * 55}\n")

        return {
            "total": total,
            "sent": sent,
            "failed": failed,
            "skipped": skipped,
            "details": details,
        }

    # --------------------------------------------------------
    # SMTP 发送
    # --------------------------------------------------------

    def _send_via_smtp(
        self,
        to_email: str,
        subject: str,
        body: str,
        bcc: Optional[str] = None,
        attachments: Optional[List[str]] = None,
    ) -> bool:
        """
        通过 SMTP 发送邮件。

        Args:
            to_email: 收件人
            subject: 主题
            body: 正文（纯文本）
            bcc: BCC 地址
            attachments: 附件路径列表

        Returns:
            True 成功
        """
        msg = MIMEMultipart()
        msg["From"] = self.sender_email
        msg["To"] = to_email
        msg["Subject"] = subject
        if bcc:
            msg["Bcc"] = bcc

        # 正文
        msg.attach(MIMEText(body, "plain", "utf-8"))

        # 附件
        if attachments:
            for filepath in attachments:
                path = Path(filepath)
                if not path.exists():
                    logger.warning(f"附件不存在: {filepath}")
                    continue

                with open(path, "rb") as f:
                    part = MIMEBase("application", "octet-stream")
                    part.set_payload(f.read())
                    encoders.encode_base64(part)
                    part.add_header(
                        "Content-Disposition",
                        f'attachment; filename="{path.name}"',
                    )
                    msg.attach(part)

        # 连接 SMTP 并发送
        try:
            if self.smtp_port == 465:
                # SSL
                server = smtplib.SMTP_SSL(self.smtp_server, self.smtp_port, timeout=30)
            else:
                # STARTTLS
                server = smtplib.SMTP(self.smtp_server, self.smtp_port, timeout=30)
                server.ehlo()
                server.starttls()
                server.ehlo()

            server.login(self.sender_email, self.sender_password)
            server.send_message(msg)
            server.quit()

            logger.info(f"邮件已发送: {to_email}")
            return True

        except smtplib.SMTPAuthenticationError as e:
            logger.error(f"SMTP 认证失败: {e}")
            print(f"  ❌ 认证失败 — 请检查邮箱地址和应用专用密码")
            return False
        except smtplib.SMTPException as e:
            logger.error(f"SMTP 错误: {e}")
            return False
        except Exception as e:
            logger.error(f"发送异常: {e}")
            return False

    # --------------------------------------------------------
    # 辅助
    # --------------------------------------------------------

    def _parse_draft(self, content: str) -> Tuple[str, str]:
        """从草稿 Markdown 中解析 Subject 和 Body"""
        subject = ""
        body = content

        # 匹配 **Subject**: ... 或 Subject: ...
        match = re.search(
            r'(?:\*\*)?Subject(?:\*\*)?:\s*(.+?)(?:\n|$)',
            content,
        )
        if match:
            subject = match.group(1).strip()

        # 取 --- 分隔线后的内容作为 body
        parts = content.split("---")
        if len(parts) >= 3:
            # 第二个 --- 之后是实际邮件内容
            body = parts[2].strip()
        elif len(parts) == 2:
            body = parts[1].strip()

        # 清理尾部元数据
        body = re.sub(r'\n---\n\*.*?\*$', '', body, flags=re.DOTALL).strip()

        return subject, body

    def _update_status(self, prof_dir: Path, new_status: str, note: str) -> None:
        """更新教授状态"""
        status_path = prof_dir / "status.json"
        status = {}
        if status_path.exists():
            try:
                status = load_json(str(status_path))
            except Exception:
                pass

        status["current_status"] = new_status
        status["last_updated"] = datetime.now().isoformat()

        history = status.get("status_history", [])
        history.append({
            "status": new_status,
            "timestamp": datetime.now().isoformat(),
            "note": note,
        })
        status["status_history"] = history

        save_json(status, str(status_path))

        # 同步更新 info.json
        info_path = prof_dir / "info.json"
        if info_path.exists():
            info = load_json(str(info_path))
            info["status"] = new_status
            save_json(info, str(info_path))

    def _log_send(
        self,
        prof_name: str,
        email: str,
        subject: str,
        status: str,
        error: str = "",
    ) -> None:
        """记录发送日志"""
        log_entry = json.dumps({
            "timestamp": datetime.now().isoformat(),
            "professor": prof_name,
            "email": email,
            "subject": subject[:100],
            "status": status,
            "error": error[:200] if error else "",
        }, ensure_ascii=False)

        with open(str(self._log_file), "a", encoding="utf-8") as f:
            f.write(log_entry + "\n")

    def get_stats(self) -> Dict[str, int]:
        return dict(self._stats)


# ============================================================
# 自测入口
# ============================================================

if __name__ == "__main__":
    import re

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )

    print(f"\n{'═' * 55}")
    print("  邮件发送器 - 自测")
    print(f"{'═' * 55}")

    sender = EmailSender()

    # 测试1: 列出待发送教授
    print("\n[1] 列出待发送教授...")
    ready = sender.list_ready_professors()
    if ready:
        for p in ready:
            print(f"    📂 {p.name}")
    else:
        print("    📭 无待发送教授")

    # 测试2: 预览邮件
    if ready:
        print(f"\n[2] 预览邮件 ({ready[0].name})...")
        preview = sender.preview_email(ready[0])
        if preview:
            print(f"    草稿:  {preview['version']}")
            print(f"    主题:  {preview['subject'][:70]}")
            print(f"    正文:  {preview['body'][:120]}...")
        else:
            print("    ⚠️ 无草稿")

    # 测试3: 试运行发送
    if ready:
        print(f"\n[3] 试运行发送...")
        result = sender.send_email(ready[0], dry_run=True)
        print(f"    结果: {'✅' if result else '❌'}")

    # 测试4: 邮件解析
    print(f"\n[4] 草稿解析测试...")
    test_draft = """# Draft
**生成时间**: 2026-07-15

---

**Subject**: Test Subject Line

Dear Professor,

This is a test email body.

Best,
Student

---

*生成于 2026-07-15*"""

    subj, bdy = sender._parse_draft(test_draft)
    print(f"    解析主题: '{subj}'")
    print(f"    解析正文: '{bdy[:50]}...'")
    assert subj == "Test Subject Line", f"Expected 'Test Subject Line', got '{subj}'"
    print(f"    ✅ 解析正确")

    print(f"\n[统计] {sender.get_stats()}")

    # 检查发送日志
    if sender._log_file.exists():
        log_content = sender._log_file.read_text(encoding="utf-8")
        print(f"\n[日志] {sender._log_file}: {len(log_content)} bytes")

    print(f"\n✅ 自测完成")
