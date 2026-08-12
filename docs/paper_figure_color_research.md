# 多子图“柱状图 + 折线图”复合图配色研究与可执行方案

> 研究对象：一个大图中有多幅结构相同的小图；每幅小图包含若干柱体和一条或多条折线。目标是同时保证跨子图比较、类别辨识、美观、色觉缺陷可读性，以及屏幕与印刷表现。
>
> 结论基于期刊官方图稿规范、颜色无障碍原始资料和科学色图研究；其中“期刊明确要求”与“据此推导的设计建议”在文中分开表述。

## 一、结论先行

最稳妥、也最接近高水平论文常见做法的不是“每个子图换一种主题色”，而是建立**全图统一的语义色彩系统**：

1. **同一数据类别在所有子图中始终使用同一颜色。**例如“方法 A”在 a–h 图中都为蓝色，而不是在 a 图为蓝、b 图为橙。这样读者能直接横向比较，不必在每个面板重新学习图例。
2. **子图主要靠位置、粗体小写字母 a/b/c、短标题和留白区分。**Nature 的官方 Figure Guide 明确要求多面板图以粗体、正体、小写 a、b、c 等标注；其可访问性指南还要求使用无障碍色板、避免色盲难辨组合和避免彩色文字。[Nature Figure Guide：Building and exporting figure panels](https://research-figure-guide.nature.com/figures/preparing-figures-our-specifications/)
3. **颜色只编码数据语义，不要同时承担“子图身份”和“柱类别”两个维度。**否则颜色数量会变成“子图数 × 柱类别数”，跨图比较困难。确需强化面板分组时，只在标题条、面板字母旁的小色块或极细上边线使用浅色强调，不要重染全部柱体。
4. **柱体用类别色或中低饱和填色；折线用深色、高对比、较粗线宽，并辅以线型和点形。**单折线优先用近黑色 `#222222`；两条以上折线使用“颜色 + 实/虚线 + 圆/三角/方形点”冗余编码。
5. **不可只靠颜色传递信息。**Okabe–Ito 的原始 Color Universal Design 页面明确建议组合颜色、形状、位置、线型和填充纹理，并建议图内直接标注；这样即使黑白复印也不丢失信息。[Okabe & Ito：Color Universal Design](https://jfly.uni-koeln.de/color/)
6. **最终一定在实际排版尺寸下检查三种状态：正常彩色、色觉缺陷模拟、灰度/黑白打印。**Nature 说明在线 PDF 保留 RGB，但印刷品会自动转为 CMYK；因此屏幕上能分辨不等于印刷后仍能分辨。[Nature Figure Guide：Colour space](https://research-figure-guide.nature.com/figures/preparing-figures-our-specifications/)

## 二、“用颜色区分不同子图”是否恰当

### 通常不恰当：同构小多图的首选是统一编码

当多个子图展示同一组指标、方法或类别，只是样本、时间、区域或实验条件不同，面板本身已经由空间位置、字母和标题区分。此时给每个面板一套不同柱色会带来三类问题：

- 同色不再代表同一语义，读者无法快速跨面板追踪同一类别；
- 颜色数量急剧增加，图例负担和误认概率上升；
- 灰度打印或红绿色觉缺陷条件下，若干颜色会塌缩成相近明度。

这是从**小多图的比较任务**和“颜色应与数据变量一一对应”的原则推导出的设计建议，不是某家期刊逐字规定。Nature 的硬性/明确建议是：面板用 a/b/c 标注、采用无障碍色板、避免色盲难辨组合和彩色文字，并在最终输出条件下保证可读。[Nature Figure Guide](https://research-figure-guide.nature.com/figures/preparing-figures-our-specifications/)

### 可以用，但只能作为次级编码

以下情形可给面板少量颜色提示：

- 面板天然分为 2–4 个实验家族，如“模型”“实验”“现场”；
- 阅读顺序复杂，单靠位置和标题不够；
- 面板颜色本身具有领域语义，如不同气候区、处理组或风险状态。

推荐只给面板标题条/字母旁色块使用很浅的色调：

| 面板组 | 浅色强调 HEX | 对应深色（仅用于细线/小标记） |
|---|---:|---:|
| 组 1 | `#DCEAF4` | `#0072B2` |
| 组 2 | `#FCE8CC` | `#E69F00` |
| 组 3 | `#D6EFE7` | `#009E73` |
| 组 4 | `#F3E0EC` | `#CC79A7` |

浅色仅作导航，柱体仍按全图统一类别色绘制。面板字母和标题文字保持 `#111111` 或黑色，不建议用彩色文字；Nature 官方示例明确将彩色文字列为应避免项，并推荐彩色方框配黑字或引线指向黑字。[Nature Figure Guide：accessibility examples](https://research-figure-guide.nature.com/figures/preparing-figures-our-specifications/)

## 三、三套可直接使用的配色模式

### 方案 A：投稿最稳妥版（推荐默认）

适用：4 类以内的柱体 + 每个面板 1 条折线；面板很多，需要横向比较。

| 图形角色 | HEX | 说明 |
|---|---:|---|
| 柱类别 1 | `#56B4E9` | 天蓝 |
| 柱类别 2 | `#E69F00` | 橙 |
| 柱类别 3 | `#009E73` | 蓝绿 |
| 柱类别 4 | `#CC79A7` | 紫红 |
| 折线 | `#222222` | 近黑，避免与任何柱类别混淆 |
| 折线点填充 | `#FFFFFF` | 白心点 |
| 折线点描边 | `#222222` | 与折线一致 |
| 轴/文字 | `#222222` | 不用纯黑也有足够对比 |
| 次要网格 | `#D9D9D9` | 仅保留水平主网格，细且浅 |
| 背景 | `#FFFFFF` | 白色 |

前四个柱色来自 Okabe–Ito 色觉友好色板。完整经典色板如下：

`#000000`, `#E69F00`, `#56B4E9`, `#009E73`, `#F0E442`, `#0072B2`, `#D55E00`, `#CC79A7`

原始设计目标包括：常见色觉类型均可分辨、颜色名称较清楚、屏幕与印刷近似可复现；原始页面同时强调这套颜色必须在冗余编码之后使用，而不是替代线型/纹理。[Okabe & Ito 原始页面](https://jfly.uni-koeln.de/color/)；Nature Methods 的相关说明见 Wong, “Colour blindness”, [DOI: 10.1038/nmeth.1618](https://doi.org/10.1038/nmeth.1618)。

实施细节：柱体可用 80%–90% 不透明度，但投稿前最好转成实际 RGB 填色，避免透明叠加在 PDF/印刷流程中产生意外颜色；折线建议 1.2–1.8 pt，点标记直径 4–6 pt。折线在柱体之上绘制，白心点能防止折线与深色柱融合。

### 方案 B：低饱和、期刊感更强的彩色版

适用：3–5 类柱体，图较密，希望比高饱和色更柔和，但仍保持类别区分。

可采用 Paul Tol 的色觉友好定性色板中的一组：

| 图形角色 | HEX |
|---|---:|
| 柱类别 1 | `#4477AA` |
| 柱类别 2 | `#66CCEE` |
| 柱类别 3 | `#228833` |
| 柱类别 4 | `#CCBB44` |
| 柱类别 5 | `#EE6677` |
| 单折线 | `#222222` |
| 第二条折线（如必须） | `#AA3377` |
| 中性元素 | `#BBBBBB` |

Paul Tol 的完整 “bright” qualitative scheme 为：

`#4477AA`, `#EE6677`, `#228833`, `#CCBB44`, `#66CCEE`, `#AA3377`, `#BBBBBB`

权威原始资料：[Paul Tol, *Colour Schemes* 技术说明 PDF](https://sronpersonalpages.nl/~pault/colourschemes.pdf)。

注意：黄色/黄褐色 `#CCBB44` 用于大面积柱体尚可，但不适合白底上的细线和小字；Okabe–Ito 也明确提醒黄色和浅蓝用于细线/小对象时较难辨，细线优先使用深蓝或橙。[Okabe & Ito](https://jfly.uni-koeln.de/color/)

### 方案 C：黑白打印优先版

适用：论文可能黑白印刷、审稿人打印阅读，或柱类别多且无法保证色彩输出。

| 图形角色 | 填色 | 冗余编码 |
|---|---:|---|
| 柱类别 1 | `#F2F2F2` | 无纹理或点纹 |
| 柱类别 2 | `#D0D0D0` | `/` 斜线 |
| 柱类别 3 | `#9E9E9E` | `\` 反斜线 |
| 柱类别 4 | `#666666` | `xx` 交叉纹 |
| 折线 1 | `#111111` | 实线 + 圆点 |
| 折线 2 | `#111111` | 虚线 + 三角点 |
| 折线 3 | `#111111` | 点划线 + 方点 |

柱体边框统一 `#333333`、0.5–0.8 pt。不要使用极密纹理，以免缩小后出现摩尔纹。Okabe–Ito 的图形示例明确推荐颜色与填充纹理、实虚线和不同点形结合，并指出这样传真或黑白复印也不会丢信息。[Okabe & Ito：redundant coding](https://jfly.uni-koeln.de/color/)

## 四、柱体与折线的视觉层级

复合图中两类图形的视觉语法不同，最好让颜色也反映这种层级：

- **柱体表示离散类别/量级：**用面积填色，饱和度不必很高；同类别跨面板颜色不变。
- **折线表示趋势/序列：**用深色轮廓、稍粗线宽、明确点形；线比柱体更“锐利”。
- **单折线不要为了“好看”再占一个高饱和类别色。**近黑线最清晰，也不与柱体图例争夺注意力。
- **两条以上折线必须冗余编码。**例如蓝 `#0072B2` + 实线 + 圆点，以及朱红 `#D55E00` + `--` 虚线 + 三角点。颜色与点形应在所有面板保持一致。
- **避免双 Y 轴造成语义误导。**如果确实使用，左/右轴标题可以分别贴近对应图形，但文字仍以黑色为主；在图例或直接标注中明确单位。颜色不能弥补比例尺选择不当。

若柱体只有一个语义类别、不同柱只是 x 轴项目，则无需“彩虹柱”：所有柱统一使用 `#7FAAC8` 或 `#4477AA`，折线用 `#222222`。只有当柱色确实编码一个需要读者辨识的变量时才使用多色。

## 五、如何选“定性 / 顺序 / 发散”色板

ColorBrewer 将色板分为三类，这个分类直接决定了柱体应该怎样着色：

- **定性（qualitative）**：无天然大小顺序的类别，如算法 A/B/C、组织类型；使用不同色相且明度大致均衡。
- **顺序（sequential）**：从低到高、从早到晚；应使用单调变化的明度/色度，而不是任意彩虹色。
- **发散（diverging）**：围绕有意义的中点（0、基准值、平均值）向两侧变化；中点应近中性，两端用对立色。

ColorBrewer 官方交互工具可筛选 colorblind safe、print friendly、photocopy safe，并显示各色板的精确颜色：[ColorBrewer 2.0](https://colorbrewer2.org/)；方法原论文：Harrower & Brewer (2003), [DOI: 10.1559/152304003100010929](https://doi.org/10.1559/152304003100010929)。

对于连续数值、热图或柱体由数值渐变着色，优先使用感知均匀且色觉友好的科学色图，而不是 jet/rainbow。Crameri 等指出不均匀/彩虹色图会产生视觉伪影、隐藏真实结构并对色觉缺陷读者不友好；其 Scientific Colour Maps 提供感知均匀、色觉友好的顺序和发散色图。[Crameri, Shephard & Heron (2020), *Nature Communications*, DOI: 10.1038/s41467-020-19160-7](https://doi.org/10.1038/s41467-020-19160-7)；[Scientific Colour Maps 官方资源](https://www.fabiocrameri.ch/colourmaps/)。

重要限制：Crameri 的连续色图适合连续量，不应被随意抽取几个颜色当作无序类别色；离散类别优先使用专门的定性色板。

## 六、色觉友好与无障碍的硬规则

1. **避免红–绿作为唯一对立。**也避免明度相近的绿/棕、蓝/紫、浅蓝/粉等组合。Okabe–Ito 对这些混淆对有直接说明。[原始页面](https://jfly.uni-koeln.de/color/)
2. **用冗余通道：**柱体增加纹理/描边，折线增加实虚线和点形，关键系列直接标注。
3. **保证前景与背景明度差。**W3C WCAG 的“Use of Color”成功准则要求颜色不能是传达信息、动作或区分元素的唯一视觉手段；普通文字最低对比度通常为 4.5:1，大文字为 3:1。[WCAG 2.1—Use of Color](https://www.w3.org/WAI/WCAG21/Understanding/use-of-color.html)；[Contrast Minimum](https://www.w3.org/WAI/WCAG21/Understanding/contrast-minimum.html)
4. **不要把 WCAG 文字对比阈值机械套到所有图形上。**它是很好的底线参考，但细线、点标记和小面积色块还受线宽与面积影响；实际尺寸测试不可省略。
5. **优先图内直接标注而非远距离图例匹配。**若空间允许，把系列名放在线尾或柱组附近；Okabe–Ito 指出色觉缺陷读者尤其难以比较相隔较远的同色对象。[Okabe & Ito](https://jfly.uni-koeln.de/color/)
6. **不要只运行色盲模拟器就宣布合格。**原始页面提醒模拟只用于发现可能混淆的颜色，不能完整代表所有色觉缺陷者的体验；印刷和光照也会改变最终颜色。[Okabe & Ito：proofing cautions](https://jfly.uni-koeln.de/color/)

## 七、印刷与投稿文件设置

- 按目标期刊最终栏宽绘制和检查，不要只在全屏放大状态判断。
- Nature 要求图稿使用 RGB 色彩空间；在线 PDF 保持 RGB，但印刷将自动转换为 CMYK。官方还建议照片至少 300 dpi，而线条、文字等尽量作为可编辑矢量输出。[Nature Figure Guide：Colour space / Image resolution / vector artwork](https://research-figure-guide.nature.com/figures/preparing-figures-our-specifications/)
- RGB → CMYK 时，鲜亮蓝绿、荧光色和浅黄最容易变化；因此不要让色相成为唯一编码，需有明度、描边、纹理、点形或文字冗余。
- 输出优先 PDF/SVG/EPS 等矢量格式（以期刊接受格式为准）；嵌入字体，不要栅格化文字和线条。
- 做一次普通办公打印机的黑白打印测试。若任意两柱在灰度下合并，修改明度或增加纹理，而不是仅微调色相。

## 八、建议的完整样式规范

可把以下规则直接写进绘图代码或实验室制图规范：

- 画布：白底 `#FFFFFF`；不使用彩色面板背景。
- 面板：统一尺寸、统一坐标范围（若数据允许）、统一图例顺序；左上角粗体小写 a/b/c。
- 柱体：同一类别全图同色；宽度一致；0.5–0.8 pt 深灰边框；不使用渐变、阴影或 3D。
- 折线：`#222222`、1.2–1.8 pt；白心圆点、深色边框；两条以上加入不同线型/点形。
- 坐标轴：`#333333`；0.6–0.8 pt；去掉上、右边框或保持极浅。
- 网格：只留必要的水平主网格 `#D9D9D9`、0.4–0.6 pt；不与数据同色。
- 文字：`#111111`；避免彩色轴标题和彩色图例文字。
- 图例：全图共享一个，顺序与柱体从左到右一致；可直接标注则优先直接标注。
- 误差线：`#333333` 或与柱体更深的同色；线帽清晰；不要用颜色区分统计显著性。
- 显著性：优先括号、星号或精确 p 值，避免只用红/绿表示“显著/不显著”。

## 九、根据子图数量的具体策略

| 场景 | 推荐策略 |
|---|---|
| 2–4 个子图，每图 2–4 类柱 | 全图共享 Okabe–Ito 类别色；黑色折线；子图用 a–d + 短标题 |
| 5–12 个同构子图 | 更应统一类别色；共享图例；统一坐标范围；不要给每图换主题色 |
| 子图分成 2–4 个家族 | 类别色不变；仅标题条/面板字母旁使用浅色分组强调 |
| 每图只有一类柱，柱只是 x 轴项目 | 所有柱统一蓝灰；不要彩虹化；折线近黑 |
| 柱类别 ≥ 6 | 先考虑拆图、直接标注或减少同时比较的类别；再使用定性色板；必须加入纹理/分组 |
| 连续数值决定柱色 | 用顺序色图；如跨 0 或相对基准则用发散色图；不要使用类别色板 |
| 必须黑白打印 | 灰度明度 + 填充纹理；折线用线型 + 点形；颜色只作附加线索 |

## 十、常见失败模式

- **每个子图一套完全不同颜色：**好看但失去跨图语义一致性。
- **同一图同时用颜色编码面板和柱类别：**产生难以解释的颜色笛卡尔积。
- **彩虹色排列无序类别：**暗示不存在的顺序，并带来不均匀明度。
- **浅黄色细折线：**屏幕白底和印刷都不清楚。
- **红线压在绿色柱上：**红绿色觉缺陷读者可能只看到明度差不足的一团。
- **全部低饱和粉彩：**在大色块上柔和，但小柱、细线、图例色块往往互相接近。
- **只靠图例颜色：**读者需要频繁远距离匹配；直接标注更友好。
- **透明度叠加太多：**折线经过不同柱色时会不断变色，破坏系列恒常性。
- **只在高分辨率屏幕检查：**缩到单栏宽或转 CMYK 后才暴露问题。

## 十一、最简可执行答案

若没有特殊领域语义，直接采用：

- 所有子图共享柱色：`#56B4E9`, `#E69F00`, `#009E73`, `#CC79A7`；
- 所有子图折线：`#222222`，白心圆点；
- 子图：粗体小写 a/b/c + 黑色短标题，不换面板主题色；
- 两条折线时：`#0072B2` 实线圆点与 `#D55E00` 虚线三角点；
- 灰度备份：`#F2F2F2`, `#D0D0D0`, `#9E9E9E`, `#666666` + 四种纹理；
- 最后做色觉缺陷模拟、100%/最终栏宽检查、黑白打印检查。

这套方案在“视觉统一、跨面板比较、色觉友好、印刷稳定”四项之间最均衡。

## 参考来源（优先一手/官方）

1. Nature Research Figure Guide, “Preparing figures—our specifications.” https://research-figure-guide.nature.com/figures/preparing-figures-our-specifications/
2. Okabe, M. & Ito, K. “Color Universal Design (CUD)—How to make figures and presentations that are friendly to Colorblind people.” https://jfly.uni-koeln.de/color/
3. Wong, B. “Colour blindness.” *Nature Methods* 8, 441 (2011). https://doi.org/10.1038/nmeth.1618
4. Harrower, M. & Brewer, C. A. “ColorBrewer.org: An Online Tool for Selecting Colour Schemes for Maps.” *The Cartographic Journal* 40(1), 27–37 (2003). https://doi.org/10.1559/152304003100010929
5. ColorBrewer 2.0 official tool. https://colorbrewer2.org/
6. Crameri, F., Shephard, G. E. & Heron, P. J. “The misuse of colour in science communication.” *Nature Communications* 11, 5444 (2020). https://doi.org/10.1038/s41467-020-19160-7
7. Crameri, F. “Scientific colour maps.” https://www.fabiocrameri.ch/colourmaps/
8. Tol, P. “Colour Schemes.” https://sronpersonalpages.nl/~pault/colourschemes.pdf
9. W3C Web Content Accessibility Guidelines 2.1, “Use of Color.” https://www.w3.org/WAI/WCAG21/Understanding/use-of-color.html
10. W3C Web Content Accessibility Guidelines 2.1, “Contrast (Minimum).” https://www.w3.org/WAI/WCAG21/Understanding/contrast-minimum.html
