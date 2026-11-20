# 变分量子作业的端到端最优编译

主稿将定理、编译时算法、运行时算法与三幅图排在四页。配套推导给出数学假设、复杂度、可达性、反例与两个编译例子。软件是这些构造的参考实现，核验使用预先指定的符号模型。

## 阅读

- `main.tex` 编译为四页的主稿。`algobox.tex` 和 `figures.tex` 分别包含算法和交换图的可编辑版本。
- `proofs.tex` 连同 `continuous.tex`、`subclass.tex`、`limits.tex` 给出展开证明。
- 如需从源码生成 PDF，使用带 Fandol 中文字体、ctex、TikZ 和 tikz-cd 的 XeLaTeX，分别从同一目录运行 `xelatex main.tex` 与 `xelatex proofs.tex`。

## 运行方法

验证使用 Python 3.12 和固定版本的 Z3。解压后在本目录运行以下命令。

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest discover -v
.venv/bin/python feedback_vqe.py
```

反馈 VQE 的演示输入是有理数转移表。程序打印精确最优时间 83/10 个模型时间单位和起始方位 e，并展示一条读出记录的交付结果。QAOA 的三个改写见证及不同价表触发的方案切换，可通过以下命令查看是否通过验证。

```sh
.venv/bin/python -m unittest -v test_diagram_vqa test_compile_qaoa_diagram
```

## 编译接口

- `diagram_vqa.py` 生成带类型的 QAOA 图。检查置换是否保留代价算符和混合算符，保存每一步改写及其回放见证。
- `compile_qaoa_diagram.py` 把同一个逻辑内核的三种程序落实为串行原语，在给定的标记擦除模型中比较时间、故障、保真度及资源。这是一个 QAOA 内核的完整改写轨道。
- `policy_ir.py` 由后端的允许动作、可见端口、结果与视界生成有限策略树。截断分支需要记为失败或证明不可达。后端须实现逐树费用和验证条件。
- `feedback_vqe.py` 实现两轮自适应 VQE，覆盖准备、抽象容错核、分支读出、参数更新和交付。抽象核的可用性与成本由输入转移表约定。此程序支持在同一表上把求解结果与闭式公式相互核查。
- `optimal_compiler.py` 接收有限个图、每图的实控制变量、可行条件、时间和原生程序见证，返回不可行、可达最优或不可达的下确界。检查器重做模型内可行性及不存在更快实现的判定。遇到求解后端无法确认的查询会显式中止。

全硬件最优还要求给出控制目录完备性及每个抽象核可执行的证据。此包没有提供 Rb、Rb–Cs 或 Sr 装置校准、真实脉冲序列或设备上的性能数据。

## 无预设执行轮数的扩展

扩展稿从同目录运行 `xelatex stopping.tex` 生成。原有两份 PDF 与该扩展针对同一个完整变分编译目标。可以用以下命令复核新算例。

```sh
.venv/bin/python stopping_vqe.py
.venv/bin/python -m unittest -v test_prefix_compiler test_cyclic_compiler test_cyclic_quotient test_stopping_vqe test_continuous_cycle
```

- `prefix_compiler.py` 以可运行的合格策略提供初始上界，逐个计算正概率前缀的时间贡献，枚举所有仍有机会满足该上界的策略。终止证明需要该模型的增长条件。验证回调负责完整的任务语义和四项预算。
- `cyclic_compiler.py` 对有限可见状态和有理原生动作求精确占用流，支持不限制轮数的重试。终态合同使用给定的理想分布和逻辑故障标记。可选的累计深度上限适用于每条可达分支。允许的经典随机选择、时长和外部输出定义属于后端合同。
- `cyclic_quotient.py` 对这些状态迭代细分，保存商图、每条原生动作与稳定分块的见证，在原始转移表上复核回提策略。
- `continuous_cycle.py` 对后端给定的半代数动作与联合公开接口展开有限支撑的实闭域公式，对提取的代数控制和混合概率再计算真实访问次数和终端统计。
- `stopping_vqe.py` 将两轮 VQE 的原始更新、两次读出、布局和在线就绪确认放入同一完整模型。确切答案是 103/10 个模型时间单位，第一次误读后验没有提供给控制器。
- 求解器需要后端证明所供动作覆盖要比较的物理操作及公开接口充分性。连续核的数值、成本表与故障定义均为理论输入，尚无真实中性原子设备的全域最优性能数据。

## 两条主定理与指数响应例

从 `dual_main.tex` 生成精确和有证近似两条主定理及统一算法。正文引用的 `two_tracks.tex` 包含收敛和适用边界的证明。运行：

```sh
.venv/bin/python smooth_vqe.py
.venv/bin/python -m unittest -v test_interval_compiler test_smooth_vqe
```

`interval_compiler.py` 接收覆盖所有合法程序结构的有限紧盒和可信的整盒区间传播，输出有理见证、含上下界的覆盖证书，或已证不可行／未决。`verify` 独立重建每个域的根盒分割，重算预算、执行时间与全域界。明确提高轮数预算即可在定理条件下达到任意正时间容差。

`smooth_vqe.py` 覆盖两种原生布局与完整共享连续控制区间，原设备服务时间与读出翻转率的关系含指数函数。求得控制值 99/256，四项预算全部按原阈值验收，最终时间差的严格上界为 0.007680 个模型时间单位。未做抽样。QAOA 的变分目标为任务语义，原算法的更新保持在编译图中。

完整 QAOA 例在 `full_qaoa.py`，用 `.venv/bin/python full_qaoa.py` 运行。两轮单边 Max-Cut 均进行编码制备、代价与混合层、综合征处理、读出及解码。程序根据第一次读数选择第二轮角度并交付最佳实测串，覆盖全部 16 条读数历史。`test_full_qaoa.py` 验证理想四串概率、布局比较、四预算与最终评估。区间编译器还支持 `max_rounds=None` 按定理持续细分并复用已有界；有限预算模式保留“未决”结果。

`pareto_ties.py` 在时间最优并列的程序中按用户给定顺序处理其余五项指标，使六项值由同一程序和同一组参数实现。`verify_lex` 独立核验全目录没有词典序更优的程序。开放域内若某一级最低值只能逼近，结果明确记录该下确界以及已经达到的前缀。
