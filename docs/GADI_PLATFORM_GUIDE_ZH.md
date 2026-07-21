# NCI Gadi 简明使用说明

本文介绍在 Gadi 上运行个人研究项目的基本流程。每位使用者的 GitHub 仓库、NCI 项目、数据集和保存路径可能不同，因此不要直接照抄其他人的路径。

## 1. 先替换自己的信息

后面的命令使用以下占位符：

| 占位符 | 替换为 |
|---|---|
| `YOUR_NCI_USERNAME` | 自己的 NCI 用户名 |
| `NCI_PROJECT` | 自己获准使用的 NCI 项目代码 |
| `PROJECT_NAME` | 自己的研究项目名称 |
| `REPOSITORY_URL` | 自己的 GitHub 仓库地址 |
| `REPOSITORY_NAME` | GitHub 仓库名称 |
| `DATASET_NAME` | 自己的数据集名称 |
| `LOCAL_DATASET_PATH` | 数据集在自己电脑上的完整路径 |
| `LOCAL_RESULT_PATH` | 结果在自己电脑上的保存路径 |

建议个人项目使用下面的目录结构：

```text
/scratch/NCI_PROJECT/YOUR_NCI_USERNAME/PROJECT_NAME/
├── code/
├── data/
├── logs/
└── results/
```

这里的目录只是通用示例。应按照自己所属 NCI 项目的权限和课题组约定确定实际目录。

## 2. 基本流程

1. 登录 Gadi；
2. 创建自己的项目目录；
3. 克隆自己的 GitHub 仓库；
4. 上传自己的数据集；
5. 修改并检查 PBS 脚本中的项目代码和路径；
6. 使用 `qsub` 提交任务；
7. 使用 `qstat` 查看状态；
8. 下载自己的结果。

训练必须通过 PBS 作业运行，不应直接在 Gadi 登录节点长时间计算。

## 3. 登录并创建目录

在本地 Windows PowerShell 中登录：

```powershell
ssh YOUR_NCI_USERNAME@gadi.nci.org.au
```

登录后创建自己的目录：

```bash
mkdir -p /scratch/NCI_PROJECT/YOUR_NCI_USERNAME/PROJECT_NAME/code
mkdir -p /scratch/NCI_PROJECT/YOUR_NCI_USERNAME/PROJECT_NAME/data
mkdir -p /scratch/NCI_PROJECT/YOUR_NCI_USERNAME/PROJECT_NAME/logs
mkdir -p /scratch/NCI_PROJECT/YOUR_NCI_USERNAME/PROJECT_NAME/results
```

每位成员都应使用自己的 NCI 账户，不要共享密码、MFA 或 SSH 私钥。

## 4. 使用自己的 GitHub 仓库

第一次使用时，在 Gadi 上克隆自己的仓库：

```bash
cd /scratch/NCI_PROJECT/YOUR_NCI_USERNAME/PROJECT_NAME/code
git clone REPOSITORY_URL
cd REPOSITORY_NAME
```

例如，`REPOSITORY_URL` 可以是：

```text
https://github.com/YOUR_GITHUB_USERNAME/YOUR_REPOSITORY.git
```

以后代码更新后，只需进入自己的仓库并拉取：

```bash
cd /scratch/NCI_PROJECT/YOUR_NCI_USERNAME/PROJECT_NAME/code/REPOSITORY_NAME
git pull --ff-only origin main
```

如果仓库是私有的，需要先按照 GitHub 要求配置访问权限。不要把 GitHub 密码、令牌或私钥写入代码和 PBS 脚本。

## 5. 上传自己的数据集

先在 Gadi 登录节点创建并确认远端数据目录：

```bash
mkdir -p /scratch/NCI_PROJECT/YOUR_NCI_USERNAME/PROJECT_NAME/data
```

然后退出 Gadi，回到**本地 Windows PowerShell**。上传方向是“自己电脑 → Gadi”，使用数据传输主机 `gadi-dm`：

```powershell
scp -r "LOCAL_DATASET_PATH\DATASET_NAME" YOUR_NCI_USERNAME@gadi-dm.nci.org.au:/scratch/NCI_PROJECT/YOUR_NCI_USERNAME/PROJECT_NAME/data/
```

上传单个文件：

```powershell
scp "LOCAL_DATASET_PATH\file.zip" YOUR_NCI_USERNAME@gadi-dm.nci.org.au:/scratch/NCI_PROJECT/YOUR_NCI_USERNAME/PROJECT_NAME/data/
```

上传后登录 Gadi 检查：

```bash
ls -lah /scratch/NCI_PROJECT/YOUR_NCI_USERNAME/PROJECT_NAME/data
du -sh /scratch/NCI_PROJECT/YOUR_NCI_USERNAME/PROJECT_NAME/data/DATASET_NAME
```

不同数据集的文件结构和预处理方式不同，应使用自己项目对应的代码和配置。涉及患者或敏感数据时，上传位置和使用方式必须符合伦理审批、数据协议和 NCI 项目要求。

## 6. 提交任务

提交前检查 PBS 脚本，至少确认：

- `#PBS -P` 是自己获准使用的 NCI 项目代码；
- 代码路径属于自己的项目；
- 输入路径指向自己的数据集；
- 日志和结果输出到自己的目录；
- CPU、GPU、内存和运行时间申请合理。

`qsub hpc/gadi/YOUR_SCRIPT.pbs` 使用的是**相对路径**，因此必须先进入包含 `hpc` 文件夹的代码仓库。建议先用 `ls` 确认脚本存在，再提交：

```bash
cd /scratch/NCI_PROJECT/YOUR_NCI_USERNAME/PROJECT_NAME/code/REPOSITORY_NAME
ls hpc/gadi/YOUR_SCRIPT.pbs
qsub hpc/gadi/YOUR_SCRIPT.pbs
```

如果不想先进入仓库，也可以把 PBS 脚本的**完整绝对路径**交给 `qsub`：

```bash
qsub /scratch/NCI_PROJECT/YOUR_NCI_USERNAME/PROJECT_NAME/code/REPOSITORY_NAME/hpc/gadi/YOUR_SCRIPT.pbs
```

两种写法效果相同。文件扩展名应为 `.pbs`，不是 `.pbz`。如果 PBS 脚本内部使用相对路径，仍建议先 `cd` 到仓库根目录再提交。

成功后会返回任务编号，请记录任务编号、实验名称和提交日期。

## 7. 查看任务和日志

查看自己的全部任务：

```bash
qstat -t -u "$USER"
```

常见状态：`Q` 表示排队，`R` 表示运行。查看已结束任务：

```bash
qstat -x -f JOB_ID
```

重点查看 `Exit_status` 和 `resources_used.walltime`。`Exit_status = 0` 通常表示正常结束。

查看自己的日志：

```bash
cd /scratch/NCI_PROJECT/YOUR_NCI_USERNAME/PROJECT_NAME/logs
ls -lt | head
tail -n 50 LOG_FILE
```

持续查看正在更新的日志：

```bash
tail -f LOG_FILE
```

## 8. 下载自己的结果

下载操作在本地 Windows PowerShell 中执行。先建立自己的本地保存目录：

```powershell
$localResultDir = 'LOCAL_RESULT_PATH'
New-Item -ItemType Directory -Force -Path $localResultDir
```

下载自己的结果：

```powershell
scp -r YOUR_NCI_USERNAME@gadi-dm.nci.org.au:/scratch/NCI_PROJECT/YOUR_NCI_USERNAME/PROJECT_NAME/results/RESULT_NAME "$localResultDir\"
```

一次登录下载多个目录可以使用：

```powershell
sftp YOUR_NCI_USERNAME@gadi-dm.nci.org.au
```

进入 `sftp>` 后运行：

```text
lcd LOCAL_RESULT_PATH
get -r /scratch/NCI_PROJECT/YOUR_NCI_USERNAME/PROJECT_NAME/results/RESULT_ONE
get -r /scratch/NCI_PROJECT/YOUR_NCI_USERNAME/PROJECT_NAME/results/RESULT_TWO
exit
```

下载完成后先检查文件数量、大小和关键结果，再决定是否清理服务器文件。

## 9. 常用命令

```bash
# 更新自己的代码
cd /scratch/NCI_PROJECT/YOUR_NCI_USERNAME/PROJECT_NAME/code/REPOSITORY_NAME
git pull --ff-only origin main

# 提交任务：先确认当前位于仓库根目录
ls hpc/gadi/YOUR_SCRIPT.pbs
qsub hpc/gadi/YOUR_SCRIPT.pbs

# 查看任务
qstat -t -u "$USER"

# 查看已完成任务
qstat -x -f JOB_ID

# 取消误提交的任务（确认编号后使用）
qdel JOB_ID
```

关闭本地 PowerShell 或断开 SSH，不会终止已经通过 `qsub` 提交的非交互式作业。

遇到问题时，应保存执行命令、完整报错、PBS 任务编号、日志文件和 Git 提交版本。平台问题可查阅 [NCI Job Submission Tutorial](https://opus.nci.org.au/spaces/Help/pages/241927319/Job%2BSubmission%2BTutorial) 或联系 [NCI Helpdesk](https://help.nci.org.au/)。
