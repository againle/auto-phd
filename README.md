# 🎓 学术申请自动化 (Auto-PhD)

一个用于自动化博士/博士后申请流程的 Python 项目。支持自动搜索目标教授、分析研究匹配度、生成个性化申请邮件。

## ✨ 功能特性

- 🔍 **自动搜索教授** — 基于研究兴趣关键词，通过 Google Scholar 自动搜索目标教授
- 📊 **研究匹配度分析** — 使用 LLM 分析你与教授研究方向的匹配程度
- 📧 **邮件自动生成** — 根据匹配结果自动生成个性化套磁邮件
- 🛡️ **熔断保护** — 内置 API 调用限制和异常处理，避免过度消耗
- 💾 **断点续传** — 支持检查点保存与恢复，中断后可从上次进度继续

## 📁 项目结构

```
auto-phd/
├── profiles/              # 个人档案文件（CV、研究陈述等）
│   └── CV.pdf
├── professors/            # 每位教授独立子文件夹（搜索和分析结果）
├── scripts/               # 所有 Python 脚本
│   ├── __init__.py
│   └── utils.py           # 通用工具函数
├── papers_cache/          # 下载的 PDF 论文缓存
├── checkpoints/           # 状态检查点（断点续传）
├── logs/                  # 运行日志
├── config.yaml            # 主配置文件（包含API密钥，已在.gitignore中）
├── requirements.txt       # Python 依赖列表
├── .env.example           # 环境变量模板
├── .gitignore
└── README.md
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 克隆或进入项目目录
cd auto-phd

# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置

```bash
# 复制环境变量模板
cp .env.example .env

# 编辑 .env，填入你的真实 API 密钥
# OPENAI_API_KEY=sk-xxxxxxxx
# EMAIL_SENDER=you@gmail.com
# EMAIL_PASSWORD=your_app_password

# 编辑 config.yaml，替换占位符
# 至少需要替换 llm.api_key 和 email.sender_password
```

### 3. 准备个人档案

将你的 CV、个人陈述等文件放入 `profiles/` 目录。

### 4. 运行

```bash
# 测试工具函数是否正常
python -c "from scripts.utils import load_config; print(load_config()['llm']['provider'])"

# 查看时间戳工具
python -c "from scripts.utils import get_timestamp; print(get_timestamp())"
```

## ⚙️ 配置说明

### config.yaml

| 配置段 | 说明 |
|--------|------|
| `llm` | LLM 配置：选择 provider（openai/anthropic）、模型、温度等 |
| `search` | 搜索参数：目标地区、每查询最大结果数、最小发表年限 |
| `quality` | 质量筛选：最小发表数、h-index 阈值、相似度阈值 |
| `email` | 邮件发送：SMTP 服务器、发件人凭证 |
| `circuit_breaker` | 熔断器：API 调用上限、搜索尝试上限、停滞检测 |
| `logging` | 日志：日志级别、文件路径、轮转策略 |

### .env 环境变量

用于存储敏感信息（API 密钥、邮箱密码），与 config.yaml 配合使用。

## ⚠️ 注意事项

1. **API 费用**：使用 OpenAI/Anthropic API 会产生费用，请注意设置 `circuit_breaker` 限制
2. **限流处理**：Google Scholar 对频繁请求有限制，已内置 `api_delay_seconds` 延迟
3. **邮箱配置**：Gmail 需使用"应用专用密码"，在 Google 账户 → 安全性 → 两步验证 → 应用专用密码 中生成
4. **配置文件安全**：`config.yaml` 和 `.env` 已在 `.gitignore` 中排除，请勿提交到公开仓库
5. **学术诚信**：请使用本项目仅为提高效率，邮件内容应真实反映你的研究兴趣

## 📄 许可

本项目仅供个人学术申请使用。
