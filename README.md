# Scene Style Transfer

本项目研究参考图驱动的场景风格迁移，重点观察风格注入对内容结构、几何稳定性和参考语义泄漏的影响，并建立可复现的实验与评价流程。

## 项目结构

```text
src/        模型组件、诊断工具和实验实现
scripts/    实验运行、评价和分析脚本
configs/    实验配置与配对清单
tests/      单元测试
docs/       已整理的项目文档
analysis/   可追踪的汇总表和分析结果
```

原始图片、模型权重、运行结果、内部计划和归档材料保存在本地，不纳入仓库。具体路径约定见 [`.gitignore`](.gitignore)。

## 环境与运行

项目使用 Python 3.10。安装开发依赖：

```bash
python -m pip install -e ".[dev]"
```

运行测试：

```bash
pytest -q
```

实验脚本需要本地准备模型和数据。运行前可先查看脚本参数：

```bash
python scripts/run_v2_2a_safe_strength_frontier.py --help
python scripts/build_v2_4_pair_preflight_analysis.py --help
```

详细实验协议和结果入口见[文档索引](docs/readme.md)。
