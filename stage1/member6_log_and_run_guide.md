# Member 6 日志与运行说明

## 一、工作日志

**日期：** 2026-03-30 ~ 2026-04-01

**负责人：** 成员6

### 1. 已完成内容
1. 补充并整理 `debug / small / full` 三类实验配置模板：
   - `configs/debug.yaml`
   - `configs/small.yaml`
   - `configs/full.yaml`

2. 编写训练启动脚本：
   - `scripts/run_debug.sh`
   - `scripts/run_train.sh`

3. 编写数据量消融配置生成脚本：
   - `make_ablation_configs.py`

4. 编写训练后最小推理验证脚本：
   - `run_infer.py`

5. 修改并补充训练主入口 `train_sft.py`，支持：
   - 按配置文件启动训练
   - 生成运行输出目录
   - 保存最小运行元信息
   - 训练后自动触发最小 inference check（由配置控制）

6. 补充最小 end-to-end smoke pipeline，支持通过一条命令跑通最小流程。

---

### 2. 本地验证情况
已完成以下本地 smoke test：

1. `debug.yaml` 可正常读取并启动最小训练流程  
2. `small.yaml` 可正常读取并启动小规模测试流程  
3. `full.yaml` 可正常读取并启动完整模板流程  
4. 训练后可执行最小 inference check  
5. `make_ablation_configs.py` 可自动生成多份 ablation 配置文件

---

### 3. 下一步计划
1. 与成员2对齐统一主入口和配置字段命名
2. 与成员5对齐真实 checkpoint 保存路径和 inference 加载接口
3. 在统一仓库中参与最终 Stage 1 集成与联调

---

## 二、运行说明

### 1. 相关文件说明

#### 配置文件
- `configs/debug.yaml`  
  最小调试配置，用于快速 smoke test

- `configs/small.yaml`  
  小规模测试配置，用于联调和初步验证

- `configs/full.yaml`  
  完整训练模板配置，用于后续较完整运行

#### 脚本文件
- `scripts/run_debug.sh`  
  一键运行 debug 级别 smoke pipeline

- `scripts/run_train.sh`  
  通用训练启动脚本，可指定不同配置文件

- `run_infer.py`  
  训练后最小推理验证脚本

- `make_ablation_configs.py`  
  数据量消融配置生成脚本

---

### 2. 环境准备

建议在项目根目录执行以下命令：

```bash
python -m pip install pyyaml
chmod +x scripts/run_debug.sh scripts/run_train.sh
```

### 3. 后台运行（含 watchdog + ntfy）

使用统一后台管理脚本：

```bash
bash scripts/run_train_eval_background.sh start
```

查看状态：

```bash
bash scripts/run_train_eval_background.sh status
```

停止全部后台进程：

```bash
bash scripts/run_train_eval_background.sh stop
```

常用环境变量：

- `NTFY_TOPIC`：通知 topic（默认 `finqa-stage1`）
- `NTFY_SERVER`：通知服务地址（默认 `https://ntfy.sh`）
- `ENABLE_NTFY`：是否启用通知（`true/false`）
- `WATCHDOG_TIMEOUT`：watchdog 超时秒数（默认 `1800`）
- `WATCHDOG_CHECK_INTERVAL`：watchdog 轮询间隔秒（默认 `60`）
- `NOTIFY_EVERY`：评测进度通知间隔（默认每 100 样本）

日志与 PID 默认在 `stage1/logs/` 下。
