# NCI Gadi 平台使用说明

## 以 BHSD nnU-Net 项目为完整示例

文档版本：2026-07-22  
示例用户：`ly6399`  
示例 NCI 项目：`ke17`  
代码库：`https://github.com/Kandasorat/bhsd-nnunet.git`

> 本文面向第一次使用 NCI Gadi、Linux、PBS 调度和远程文件传输的教师与
> 组员。示例使用本项目的真实目录、作业编号和故障记录，但每位成员必须
> 使用自己的 NCI 账户，不能共享 `ly6399` 的密码、MFA 或 SSH 私钥。

## 1. Gadi 是什么

Gadi 是 NCI 的高性能计算集群，使用 Rocky Linux 和 PBS Professional
调度系统。它不是一台可以由用户直接长期占用的普通远程电脑。标准流程是：

1. 在个人电脑上准备和审核代码；
2. 将代码推送到 GitHub；
3. SSH 登录 Gadi；
4. 在登录节点拉取代码、检查文件并用 `qsub` 提交任务；
5. PBS 在有资源时把任务放到 CPU/GPU 计算节点；
6. 用 `qstat` 查看状态和资源；
7. 结果完成后通过数据传输主机下载；
8. 核对完整性后再决定是否清理服务器结果。

官方参考：

- [NCI Job Submission Tutorial](https://opus.nci.org.au/spaces/Help/pages/241927319/Job%2BSubmission%2BTutorial)
- [NCI HPC Systems](https://nci.org.au/infrastructure/hpc-systems)
- [NCI User Training](https://nci.org.au/users/user-training)

## 2. 四种环境必须分清

| 环境 | 示例 | 适合做什么 | 不应做什么 |
|---|---|---|---|
| Windows PowerShell | `PS C:\Users\...>` | Git、本地检查、SSH、下载 | 不直接执行 Gadi Linux 训练命令 |
| Gadi 登录节点 | `ly6399@gadi-login-07` | `git pull`、`qsub`、`qstat`、看日志 | 不运行训练和长时间 Python |
| PBS 计算节点 | PBS 自动分配 | 真正的 CPU/GPU 训练与评估 | 不需要手工 SSH 进入 |
| 数据传输主机 | `gadi-dm.nci.org.au` | `scp`、`sftp`、大文件传输 | 不提交或运行训练 |

最重要的原则：**登录节点只组织任务，训练必须通过 PBS。** 关闭本地
PowerShell 或断开 SSH，不会停止已提交的非交互 PBS 作业。

## 3. 账户、项目和安全

新成员需要注册自己的 NCI 账户，通过 MyNCI 申请加入 `ke17`，等待负责人
批准，并按 NCI 要求配置认证。其他成员登录时应替换用户名：

```powershell
ssh YOUR_NCI_USERNAME@gadi.nci.org.au
```

严禁：

- 共用 `ly6399` 的密码、MFA、恢复码或 SSH 私钥；
- 将凭据写入 Git、PBS、说明文档或截图；
- 未经数据治理允许复制医学数据；
- 将患者数据上传到未批准的个人云盘或代码仓库。

## 4. 本项目的 Gadi 目录

```text
/scratch/ke17/bhsd-nnunet
├── software/bhsd-nnunet       Git 仓库
├── envs/bhsd-nnunet-py310     Python 环境
├── data/
│   ├── nnUNet_raw
│   └── nnUNet_preprocessed
├── runs/
│   ├── nnUNet_results         checkpoint、验证结果、summary.json
│   └── experiment_metadata    时间、GPU、PBS、成本记录
├── cache                      软件缓存
└── logs                       PBS stdout/stderr
```

登录后常用变量：

```bash
export BHSD_ROOT=/scratch/ke17/bhsd-nnunet
export REPO_DIR="$BHSD_ROOT/software/bhsd-nnunet"
export nnUNet_raw="$BHSD_ROOT/data/nnUNet_raw"
export nnUNet_preprocessed="$BHSD_ROOT/data/nnUNet_preprocessed"
export nnUNet_results="$BHSD_ROOT/runs/nnUNet_results"
export BHSD_RESULTS_DIR="$BHSD_ROOT/runs/experiment_metadata"
```

大型数据、环境和结果不应放在 `$HOME` 中。

## 5. 登录和确认环境

Windows PowerShell：

```powershell
ssh ly6399@gadi.nci.org.au
```

成功后提示符类似：

```text
[ly6399@gadi-login-07 ~]$
```

在 Gadi：

```bash
whoami
hostname
pwd

export BHSD_ROOT=/scratch/ke17/bhsd-nnunet
export REPO_DIR="$BHSD_ROOT/software/bhsd-nnunet"
cd "$REPO_DIR"
```

## 6. GitHub 到 Gadi 的同步

推荐方向：

```text
本地修改/审核 → GitHub main → Gadi git pull → qsub
```

每次提交任务前：

```bash
cd /scratch/ke17/bhsd-nnunet/software/bhsd-nnunet
git status --short
git pull --ff-only origin main
git rev-parse --short HEAD
```

规则：

- `git status --short` 最好为空；
- `--ff-only` 避免服务器意外产生 merge commit；
- Job ID 必须与 Git revision 一起登记；
- 遇到陌生修改不要运行 `git reset --hard`；
- Gadi 只拉取已审核脚本，不作为主要开发位置。

## 7. Python 和 nnU-Net 环境

```bash
module purge
module load python3/3.10.4
source /scratch/ke17/bhsd-nnunet/envs/bhsd-nnunet-py310/bin/activate

export BHSD_ROOT=/scratch/ke17/bhsd-nnunet
export nnUNet_raw="$BHSD_ROOT/data/nnUNet_raw"
export nnUNet_preprocessed="$BHSD_ROOT/data/nnUNet_preprocessed"
export nnUNet_results="$BHSD_ROOT/runs/nnUNet_results"
export BHSD_RESULTS_DIR="$BHSD_ROOT/runs/experiment_metadata"

which python3
python3 --version
python3 -c "import torch, nnunetv2; print(torch.__version__)"
```

本项目已验证 Python 3.10.4、PyTorch 2.7.1+cu118 和 nnU-Net 2.6.4。
NCI 官方建议显式指定 module 版本，避免默认版本变化。

## 8. 提交前预检

```bash
cd "$REPO_DIR"
python3 scripts/check_gadi_ready.py --server --require-binary
```

频谱与成本感知实验还运行：

```bash
python3 scripts/verify_spectral_slice_fusion.py
python3 scripts/verify_compute_cost.py
```

只有全部通过才提交。预检失败时先解决代码、配置、路径或环境问题，不要反复
`qsub`。

## 9. PBS 脚本怎样理解

以 D0-D6 GPU 数组为例：

```bash
#!/bin/bash
#PBS -P ke17
#PBS -q gpuvolta
#PBS -N bhsd_25d_spec_f0
#PBS -J 0-6
#PBS -r y
#PBS -l walltime=48:00:00
#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -l mem=32GB
#PBS -l jobfs=20GB
#PBS -l storage=scratch/ke17
```

| PBS 项 | 含义 |
|---|---|
| `-P ke17` | 费用计入项目 `ke17` |
| `-q gpuvolta` | Volta GPU 队列 |
| `-N` | 作业名 |
| `-J 0-6` | 创建 7 个数组子任务 |
| `-r y` | 作业可重新运行 |
| `walltime` | 最长允许时间，不是预计完成时间 |
| `ncpus` | CPU 核数 |
| `ngpus` | GPU 数量 |
| `mem` | 内存申请 |
| `jobfs` | 计算节点本地临时空间 |
| `storage` | 允许访问的项目文件系统 |

申请更多资源不保证更快，可能延长排队并浪费额度。资源应依据 walltime、
显存、GPU 利用率和 profile 实测调整。

## 10. 提交任务

从外部日志目录提交，避免 PBS 输出污染 Git 仓库：

```bash
mkdir -p /scratch/ke17/bhsd-nnunet/logs
cd /scratch/ke17/bhsd-nnunet/logs

qsub "$REPO_DIR/hpc/gadi/train_25d_spectral_screen_fold0.pbs"
```

返回示例：

```text
174360420[].gadi-pbs
```

立即记录 Job ID、Git revision、PBS 文件、数据集、fold、数组映射、提交人和
时间。不要因为状态暂时为 `Q` 就再次提交。

## 11. 真实任务示例

### D0-D6

```text
Job ID: 174360420[].gadi-pbs
Dataset001_BHSD, fold 0, indices 0-6
```

| Index | 方法 |
|---:|---|
| 0 | D0 中心描述符容量控制 |
| 1 | D1 低频 Z0 |
| 2 | D2 有符号差分 Z1 |
| 3 | D3 曲率门控 Z2 |
| 4 | D4 完整正交频谱 |
| 5 | D5 自适应方向频谱 |
| 6 | D6 邻片交换不变频谱 |

一次真实状态截图中，`174346794[0-5]` 六个任务为 `R`，而
`174360420[0-6]` 七个任务为 `Q`。这表示前一批任务正在占用 GPU，后一批
等待调度，不表示失败；PBS 会自动启动排队任务。

### 历史任务追溯

| 实验 | Job ID | 结果 |
|---|---|---|
| 多类 2D 五折 | `174241149[]` | 全部成功 |
| 多类 3D 五折 | `174241150[]` | 全部成功 |
| 二类 2D 五折 | `174241151[]` | 全部成功 |
| 二类 3D 五折 | `174241152[]` | 全部成功 |
| 多类标准 2.5D fold 0 | `174332417` | `Exit_status=0` |
| 二类标准 2.5D fold 0 | `174332422` | `Exit_status=0` |
| 多类 A1-A8 | `174338292[]` | 全部成功 |

这些编号用于追溯，不能重新提交相同实验。

## 12. 用 qstat 监控

查看自己的当前任务：

```bash
qstat -t -u "$USER"
```

查看整个数组，包括历史：

```bash
qstat -x -t "174360420[].gadi-pbs"
```

查看一个子任务：

```bash
qstat -x -f "174360420[0].gadi-pbs"
```

提取关键信息：

```bash
qstat -x -f "174360420[0].gadi-pbs" |
grep -E "job_state|Exit_status|resources_used.walltime|resources_used.mem|comment"
```

| 状态 | 含义 | 建议 |
|---|---|---|
| `Q` | 排队 | 等待，查看 `comment`，不要重复提交 |
| `R` | 运行 | 查看日志和耗时 |
| `X` | 已结束/退出阶段 | 查看 `Exit_status` |
| `F` | 历史完成记录 | 查看 `Exit_status` |
| `H` | hold | 查看原因或联系支持 |

成功的最低条件是 `Exit_status = 0`。任务从普通 `qstat` 消失不能证明成功。

数组查询曾遇到：

```bash
qstat -t 174338292
# qstat: Unknown Job Id
```

正确的历史数组形式：

```bash
qstat -x -t "174338292[].gadi-pbs"
```

## 13. 查看日志

```bash
cd /scratch/ke17/bhsd-nnunet/logs
ls -lt | head -30
tail -n 100 LOG_FILE
grep -Ein "error|exception|traceback|out of memory|killed" LOG_FILE
```

实时查看可用 `tail -f LOG_FILE`，按 `Ctrl+C` 只会停止查看，不会取消 PBS。

成本感知任务开头应出现：

```text
Compute-cost verification passed.
Compute profile written to ...
```

## 14. 资源和计算成本

```bash
qstat -x -f "JOB_ID" |
grep -E "resources_used.walltime|resources_used.mem|resources_used.cpupercent|Exit_status"
```

本项目还保存：

```text
experiment_metadata/<experiment>/stage_metrics.csv
nnUNet_results/.../fold_0/run_timing.json
experiment_metadata/<experiment>/compute_profile.json
nnUNet_results/.../fold_0/compute_profile.json
```

这些文件包含总时长、GPU 显存和利用率、参数量、前向延迟与骨干调用次数。
模型选择不能只看 Dice；复杂模型必须证明额外成本带来足够、稳定的收益。

## 15. 正确理解 nnU-Net 指标

本项目使用：

- 在线 foreground Dice EMA：早停和 checkpoint 选择信号；
- `checkpoint_best.pth`：正式验证模型；
- `validation/summary.json`：完整病例最终 Dice。

在线 EMA 与 summary Dice 统计方式不同，不能直接比较。组内结果表统一填写
best checkpoint 对应的 `validation/summary.json`。

## 16. 续训和取消

PBS 会检测已有 checkpoint 并自动传递 resume。只重提 index 3 的示例：

```bash
qsub -r y -J 3-3 -v BHSD_RESUME=1 \
  "$REPO_DIR/hpc/gadi/train_25d_spectral_screen_fold0.pbs"
```

仅在原子任务已经结束、checkpoint 存在且确认没有重复任务时使用。

取消：

```bash
qdel "JOB_ID"
```

`qdel` 会终止计算，必须先核对 Job ID 并通知负责人。

## 17. Windows 下载结果

可以另开 PowerShell；原 Gadi SSH 窗口可保持打开。大量传输使用
`gadi-dm.nci.org.au`。

### 一个目录

先进入本地目标父目录，避免 OneDrive 空格导致 `scp Invalid argument`：

```powershell
$backupRoot = 'C:\Users\92127\OneDrive - UNSW\project_linpeng\server_backups'
New-Item -ItemType Directory -Force -Path $backupRoot
Set-Location -LiteralPath $backupRoot

scp -r `
  ly6399@gadi-dm.nci.org.au:/scratch/ke17/bhsd-nnunet/runs/nnUNet_results/REMOTE_RESULT_DIRECTORY `
  .
```

### 多个目录只认证一次

```powershell
sftp ly6399@gadi-dm.nci.org.au
```

进入 SFTP 后：

```text
lcd "C:\Users\92127\OneDrive - UNSW\project_linpeng\server_backups"
cd /scratch/ke17/bhsd-nnunet/runs/nnUNet_results/Dataset001_BHSD
ls
get -R FIRST_RESULT_DIRECTORY
get -R SECOND_RESULT_DIRECTORY
get -R THIRD_RESULT_DIRECTORY
bye
```

这样只登录一次、正常下载目录，不需要服务器打包。结果树之外还要下载：

```text
/scratch/ke17/bhsd-nnunet/runs/experiment_metadata
```

## 18. 下载后核对

Gadi：

```bash
find RESULT_DIRECTORY -type f | wc -l
du -sb RESULT_DIRECTORY
find RESULT_DIRECTORY -type f \
  \( -name 'checkpoint_best.pth' -o \
     -name 'checkpoint_final.pth' -o \
     -name 'summary.json' -o \
     -name 'run_timing.json' -o \
     -name 'compute_profile.json' \) -print
```

Windows：

```powershell
$result = 'C:\FULL\LOCAL\RESULT\PATH'
$files = Get-ChildItem -LiteralPath $result -Recurse -File
[pscustomobject]@{
    FileCount = $files.Count
    TotalBytes = ($files | Measure-Object Length -Sum).Sum
}

$files |
  Where-Object Name -in @(
    'checkpoint_best.pth',
    'checkpoint_final.pth',
    'summary.json',
    'run_timing.json',
    'compute_profile.json'
  ) |
  Select-Object FullName, Length
```

服务器 `du -sb` 包含目录元数据，可能比 Windows 文件字节略大。应同时比较
文件数量、关键文件和总字节。确认本地不是 OneDrive 离线占位符前，不得删
除服务器结果。

## 19. 存储管理

```bash
du -sh /scratch/ke17/bhsd-nnunet/*
du -sh /scratch/ke17/bhsd-nnunet/runs/*
```

删除结果前必须满足：

1. 所有子任务 `Exit_status=0`；
2. 完整下载结果和 metadata；
3. 核对文件数量与字节；
4. best/final checkpoint、日志和 summary 存在；
5. 负责人同意；
6. 删除命令使用准确完整路径。

永远不要删除整个 `/scratch/ke17`、`bhsd-nnunet`、`data`、`envs` 或
cache，也不要对未经核对的变量执行递归删除。

## 20. 已遇到的典型问题

### `python` 已弃用

```bash
module load python3/3.10.4
python3 SCRIPT.py
```

### `libpython3.10.so.1.0` 找不到

原因通常是未加载依赖 module 就直接调用环境解释器：

```bash
module purge
module load python3/3.10.4
source /scratch/ke17/bhsd-nnunet/envs/bhsd-nnunet-py310/bin/activate
python3 SCRIPT.py
```

### `SyntaxError: future feature annotations is not defined`

曾由脚本首行错误导致。应在本地修复、测试、推送，然后 Gadi 执行：

```bash
git pull --ff-only origin main
```

不要在多份服务器脚本中分别手改。

### `qstat: Unknown Job Id`

对历史数组加 `-x` 并使用完整 `[]`：

```bash
qstat -x -t "JOB_ID[].gadi-pbs"
```

### 所有任务为 `Q`

`Q` 是等待，不是失败：

```bash
qstat -x -f "JOB_ID[0].gadi-pbs" |
grep -E "job_state|comment|estimated.start_time"
```

不要重复提交。

### PowerShell `scp ... Invalid argument`

常见原因是本地 OneDrive 路径含空格。先 `Set-Location` 到目标父目录，将
目标写成 `.`；多个目录使用交互式 `sftp`。

### 关闭窗口是否停止任务

不会。`qsub` 后由 PBS 管理。只有尚未提交的前台命令会随 SSH 中断受影响。

### 为什么运行时间不同

早停 epoch、模型复杂度、前向次数、数据加载、完整验证和文件系统波动都可能
影响时间。D6 明确执行两次骨干前向。因此要一起记录 Dice、epoch、walltime、
显存、参数和 profile。

## 21. 团队实验登记

建议维护共享表：

| 字段 | 示例 |
|---|---|
| 提交人 | `ly6399` |
| 日期 | `2026-07-22` |
| Git revision | `43fb3a2` |
| PBS 文件 | `train_25d_spectral_screen_fold0.pbs` |
| Job ID | `174360420[]` |
| 数据集 | `Dataset001_BHSD` |
| fold/index | `fold 0, D0-D6` |
| 选择规则 | best checkpoint + full-case summary |
| 状态 | Q/R/finished/verified/downloaded |
| 本地备份 | 完整路径 |
| 备注 | 恢复、错误、成本和异常 |

团队规则：

- 每人使用自己的 NCI 账户；
- 代码修改后由至少一人审核；
- 同一批实验只有一个人负责提交；
- Job ID 立即写入共享记录；
- 不混用在线 EMA 与 summary Dice；
- 不因模型复杂就默认更好；
- fold 0 只筛选，正式结论需多 seed、多 fold；
- 不把 harmonized adaptation 写成官方复现；
- 删除服务器结果前双人确认。

## 22. 完整新实验模板

本地：修改 → 测试 → commit → push → 记录 revision。

Gadi 登录节点：

```bash
ssh ly6399@gadi.nci.org.au
export BHSD_ROOT=/scratch/ke17/bhsd-nnunet
export REPO_DIR="$BHSD_ROOT/software/bhsd-nnunet"
cd "$REPO_DIR"
git status --short
git pull --ff-only origin main
git rev-parse --short HEAD

module purge
module load python3/3.10.4
source "$BHSD_ROOT/envs/bhsd-nnunet-py310/bin/activate"

export nnUNet_raw="$BHSD_ROOT/data/nnUNet_raw"
export nnUNet_preprocessed="$BHSD_ROOT/data/nnUNet_preprocessed"
export nnUNet_results="$BHSD_ROOT/runs/nnUNet_results"
export BHSD_RESULTS_DIR="$BHSD_ROOT/runs/experiment_metadata"

python3 scripts/check_gadi_ready.py --server --require-binary

cd "$BHSD_ROOT/logs"
qsub "$REPO_DIR/hpc/gadi/SELECTED_JOB.pbs"
```

监控：

```bash
qstat -t -u "$USER"
qstat -x -t "RETURNED_JOB_ID[].gadi-pbs"
```

完成后依次：检查 `Exit_status` → 检查日志 → 读取 best summary → 生成成本
汇总 → 从 `gadi-dm` 下载 → 核对 → 更新交接 → 经确认后清理。

## 23. 快速命令表

```bash
# 登录
ssh USERNAME@gadi.nci.org.au

# 拉取代码
cd /scratch/ke17/bhsd-nnunet/software/bhsd-nnunet
git pull --ff-only origin main

# 环境
module purge
module load python3/3.10.4
source /scratch/ke17/bhsd-nnunet/envs/bhsd-nnunet-py310/bin/activate

# 提交
cd /scratch/ke17/bhsd-nnunet/logs
qsub /scratch/ke17/bhsd-nnunet/software/bhsd-nnunet/hpc/gadi/JOB.pbs

# 当前任务
qstat -t -u "$USER"

# 数组历史
qstat -x -t "JOB_ID[].gadi-pbs"

# 退出状态和资源
qstat -x -f "JOB_ID[INDEX].gadi-pbs" |
grep -E "job_state|Exit_status|resources_used.walltime|resources_used.mem"

# 日志和空间
cd /scratch/ke17/bhsd-nnunet/logs
ls -lt | head
du -sh /scratch/ke17/bhsd-nnunet/*
```

## 24. 何时联系 NCI

账户/MFA、项目 membership、额度或存储授权、平台文件系统、队列 hold、官方
module 或网络传输问题应联系 NCI：

- [NCI Helpdesk](https://help.nci.org.au/)
- `help@nci.org.au`
- [NCI User Training](https://nci.org.au/users/user-training)

项目代码、实验定义、医学指标和 Git 冲突先由项目负责人处理。

## 25. 权威项目状态

平台通用规则以 NCI 官方文档为准；项目当前状态、已完成任务和 Immediate
next work 以以下文件最顶部 authoritative snapshot 为准：

```text
C:\Users\92127\OneDrive - UNSW\project_linpeng\PROJECT_HANDOFF.md
```

仓库副本是 `docs/PROJECT_HANDOFF.md`。不要根据旧聊天或旧截图重新提交已经
完成或正在运行的任务。
