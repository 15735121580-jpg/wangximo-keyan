请完成一次独立的 OpenQuantum VQA joint harness agent 使用验证。你是 GPT-6.1 Sol。仅依据本条任务、安装的 harness 接口与本次运行文件工作，使用一个 agent 完成，控制模型调用用量。

本机开源仓库位于 <absolute-repository-root>。先读取仓库 skills/vqa-joint-compilation/SKILL.md。应优先调用实际可用的 vqa-joint-harness MCP 的 harness_status 和 compile_vqa_job。若本对话 MCP 目录尚未刷新，可用仓库 .venv/bin/vqa-joint CLI，必须使用 --backend openquantum 并报告实际路径。通过 OpenQuantum 的已固定 Qiskit SDK 运行。使用当前 Codex 登录提供模型访问。使用已有认证入口，凭据留在本机。

请根据下面的问题与设备接口自行构造 job.json，然后调用 harness，实际生成一个新的优化编译程序及完整策略，运行训练与新的终端测量，给出问题答案。

任务为三比特变分基态能量估计。初态 |000>，ansatz 为 RY(theta) 作用 q0，再 CX(q0,q1)，再 CX(q1,q2)。Hamiltonian H = -0.73 ZII -0.41 XXX -0.12 IZZ。Pauli 字符串最左边是 q0。theta 参数网格为 j*pi/16，j=0,...,16，保持此变分任务与网格。每个 Pauli 项每个训练点请求 1024 个样本，选择训练估计能量最低的参数，再用每项 4096 个全新终端样本估计能量。seed=41。使用有限 shots，不用参考答案替代训练或终端结果。

实验平台是 OpenQuantum Qiskit 的三站点中性原子数字门本地模型，执行本地数值验证。硬件为 3 个物理比特，边为 (0,1),(1,2)，native basis 为 rz,sx,x,cz。门时长（微秒）rz=0、sx=x=0.04、cz=0.30，reset=1、readout=1。单比特平均门保真度0.9999，双比特0.999，采用 harness 声明的独立局部退极化信道，reset/readout/memory理想。没有主动编码，逻辑错误率按不适用记录。目标是保证可观测量语义正确，联合改写门分解、准备、mapping/routing、读出，并核算整个变分作业的资源和答案。确定性恒等观测量可由精确经典记录返回，分别列出请求样本与设备shots。

保持仓库源文件只读，将本次输入、原始结果和输出程序放在当前新工作区的独立子目录。先检验 OpenQuantum 环境与 harness，再实际执行。读取结果文件，检查理想读出等价、完整资源计数、门与连接合法性，以及采样答案相对独立精确参考的误差。终端 OpenQASM 文件应能重新读取。重放程序执行一次，核对固定 seed 下输出一致。

在本对话用中文完整汇报：输入任务与平台；agent实际调用的接口；发现并选中的跨阶段改写；门分解、准备、mapping/routing、调度、测量、经典反馈与答案构成的完整执行策略；宿主和联合优化的完整门数、CZ、峰值比特、深度和时间；实际终端参数、能量与不确定性；搜索范围和验证结果。给出所有输入、策略、可执行编译程序和结果的绝对路径链接。最后写 validation-summary.md，记录使用的实际模型名称（以本对话可见信息为准）、执行入口、runtime版本与结果。
