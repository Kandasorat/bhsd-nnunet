# NCI Gadi 简明使用说明

本文面向第一次使用 Gadi 的老师和组员，以 BHSD nnU-Net 项目为例，只介绍日常最常用的操作。

## 1. 基本流程

一次实验通常按下面的顺序进行：

1. 在本地修改代码并推送到 GitHub；
2. 登录 Gadi；
3. 从 GitHub 拉取最新代码；
4. 使用 `qsub` 提交任务；
5. 使用 `qstat` 查看任务状态；
6. 查看日志和结果；
7. 将结果下载到本地保存。

Gadi 的登录节点主要用于管理文件和提交任务。训练等计算工作必须通过 PBS 作业运行，不应直接在登录节点执行。

## 2. 登录 Gadi

在 Windows PowerShell 中运行：

```powershell
ssh YOUR_NCI_USERNAME@gadi.nci.org.au
```

将 `YOUR_NCI_USERNAME` 替换为自己的 NCI 用户名。每位成员都应使用自己的账户，不要共享密码、MFA 或 SSH 私钥。

本项目在 Gadi 上的主要目录为：

```text
/scratch/ke17/bhsd-nnunet
```

## 3. 拉取最新代码

登录 Gadi 后运行：

```bash
cd /scratch/ke17/bhsd-nnunet/software/bhsd-nnunet
git pull --ff-only origin main
```

这样即可获得已经推送到 GitHub 的最新脚本。正式提交前可以运行项目检查：

```bash
python3 scripts/check_gadi_ready.py
```

## 4. 提交任务

本项目已经准备好 PBS 脚本，一般只需运行：

```bash
qsub hpc/gadi/脚本名称.pbs
```

例如提交一个任务数组：

```bash
qsub hpc/gadi/train_25d_fusion_screen_array.pbs
```

提交成功后会返回任务编号，例如：

```text
174360420[].gadi-pbs
```

请记录任务编号、实验名称和提交日期，以便之后查看和整理结果。

## 5. 查看任务状态

查看自己当前的全部任务：

```bash
qstat -t -u "$USER"
```

常见状态：

| 状态 | 含义 |
|---|---|
| `Q` | 正在排队等待资源 |
| `R` | 正在运行 |
| `X` 或 `F` | 已结束，可进一步检查退出状态 |

查看某个已完成任务的详细信息：

```bash
qstat -x -f 任务编号
```

重点查看：

```text
job_state
Exit_status
resources_used.walltime
```

`Exit_status = 0` 通常表示任务正常结束。任务数组需要保留方括号，例如：

```bash
qstat -x -t "174360420[].gadi-pbs"
```

## 6. 查看日志和结果

本项目日志通常保存在：

```text
/scratch/ke17/bhsd-nnunet/logs
```

查看最新日志：

```bash
cd /scratch/ke17/bhsd-nnunet/logs
ls -lt | head
```

查看日志末尾：

```bash
tail -n 50 日志文件名
```

持续观察正在写入的日志：

```bash
tail -f 日志文件名
```

训练结果主要位于：

```text
/scratch/ke17/bhsd-nnunet/runs/nnUNet_results
```

实验时间和资源记录主要位于：

```text
/scratch/ke17/bhsd-nnunet/runs/experiment_metadata
```

## 7. 下载结果到本地

下载操作应在本地 Windows PowerShell 中执行，不是在 Gadi 终端中执行。

先创建本地保存目录：

```powershell
$backupRoot = 'C:\Users\92127\OneDrive - UNSW\project_linpeng\server_backups'
New-Item -ItemType Directory -Force -Path $backupRoot
```

然后下载结果目录：

```powershell
scp -r YOUR_NCI_USERNAME@gadi-dm.nci.org.au:/scratch/ke17/bhsd-nnunet/runs/需要下载的目录 "$backupRoot\"
```

如果要在一次登录中下载多个目录，可以使用：

```powershell
sftp YOUR_NCI_USERNAME@gadi-dm.nci.org.au
```

进入 `sftp>` 后依次运行：

```text
lcd C:\Users\92127\OneDrive - UNSW\project_linpeng\server_backups
get -r /scratch/ke17/bhsd-nnunet/runs/目录一
get -r /scratch/ke17/bhsd-nnunet/runs/目录二
exit
```

下载完成后，应检查文件数量、大小和关键结果文件，再考虑清理服务器文件。

## 8. 常用命令速查

```bash
# 拉取代码
cd /scratch/ke17/bhsd-nnunet/software/bhsd-nnunet
git pull --ff-only origin main

# 提交任务
qsub hpc/gadi/脚本名称.pbs

# 查看任务
qstat -t -u "$USER"

# 查看已完成任务
qstat -x -f 任务编号

# 查看最新日志
cd /scratch/ke17/bhsd-nnunet/logs
ls -lt | head

# 取消误提交的任务（确认编号后使用）
qdel 任务编号
```

关闭 PowerShell 或断开 SSH，不会终止已经通过 `qsub` 提交的非交互式任务。

## 9. 遇到问题时

先保存以下信息：

- 执行的命令；
- 完整报错；
- PBS 任务编号；
- 输出和错误日志；
- 使用的脚本和 Git 提交版本。

项目当前状态和下一步工作以项目根目录中的 `PROJECT_HANDOFF.md` 最顶部 **AUTHORITATIVE CURRENT SNAPSHOT** 为准。Gadi 平台问题也可查阅 [NCI Job Submission Tutorial](https://opus.nci.org.au/spaces/Help/pages/241927319/Job%2BSubmission%2BTutorial) 或联系 [NCI Helpdesk](https://help.nci.org.au/)。
