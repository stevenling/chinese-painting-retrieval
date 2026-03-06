"""
项目名称：Chinese Painting Retrieval（中国画检索与分类）

一、项目简介
----------------

本项目实现了一个基于深度学习的中国画图像分类与检索系统，主要功能包括：
- 使用预训练 VGG16 网络提取国画图像的高维特征；
- 基于提取的特征训练一个全连接分类器，区分「花鸟 / 人物 / 山水」三类国画；
- 将图像特征与标签写入 SQLite 数据库，支持基于特征的图像检索；
- 提供命令行预测脚本和基于 PyQt5 的桌面图形界面，用于分类与检索展示；
- 附带爬虫脚本，用于从雅昌艺术网下载国画图像构建数据集。


二、目录结构总览
----------------

仓库根目录下的关键内容：
- README.md：仓库说明（原始）
- recognizePaint/：项目主要代码和资源
  - app.py：命令行方式对单张图片进行分类预测
  - menu.py：PyQt5 图形界面（分类 + 检索）
  - image_retrieval.py：另一份 PyQt5 图形界面实现（功能类似）
  - get_features.py：批量提取训练数据集的 VGG16 特征，生成 codes.npy 和 labels
  - ftrain.py：基于 VGG16 特征训练 3 类国画分类器
  - insert_sql.py：将图像特征与标签写入 SQLite 数据库 paint.db 的 image 表
  - show_sql.py：用于查看 / 清空 image 表中的记录
  - create_sql.py：创建 SQLite 数据库和 image 表
  - download_image.py：从雅昌艺术网国画列表页爬取图片并下载到本地
  - download_conf_matrix.py：从网页爬取图像（早期脚本，与 download_image.py 类似）
  - confusion_matrix.py：批量预测测试集并统计混淆矩阵
  - transfer_train.py / transfer_test.py：与迁移学习相关的训练 / 测试脚本
  - test.py / test2.py / test/test_image.py：测试用脚本和简单 UI 演示
  - codes.npy：VGG16 特征数组（由 get_features.py 生成，供 ftrain.py 使用）
  - labels：与 codes.npy 对应的类别标签列表（文本文件，一行一个标签）
  - imageData.pkl：使用 pickle 存储的特征数组（供 insert_sql.py 写入数据库使用）
  - paint.db：SQLite 数据库文件（由 create_sql.py + insert_sql.py 生成）
  - paint_photos/：训练用国画图片数据集（按子文件夹划分类别）
  - test_photos/：测试 / 验证集图片
  - checkpoints/：TensorFlow 训练好的模型权重（paint.ckpt 等）
  - tensorflow_vgg/：外部 VGG16/VGG19 实现和预训练权重读取工具
  - doc/：
    - config.md：配置说明（原有）
    - project_overview.md：本项目说明文档


三、核心流程说明
----------------

1. 数据准备与特征提取
^^^^^^^^^^^^^^^^^^^^
- 原始国画图片按类别放在 recognizePaint/paint_photos/ 下：
  - paint_photos/flowerBird/
  - paint_photos/human/
  - paint_photos/landscape/
- 运行 `get_features.py`：
  - 使用 VGG16（`tensorflow_vgg.vgg16.Vgg16`）对每张图片提取 relu6 层特征；
  - 将所有图片的特征堆叠成二维数组 codes（形状约为 [样本数, 特征维度]），使用 `np.save("codes.npy", codes)` 保存；
  - 同时把每张图片的类别名称以同样顺序写入文本文件 `labels`。


2. 模型训练（分类器）
^^^^^^^^^^^^^^^^^^^^
- 训练脚本：`ftrain.py`
  - 从 `codes.npy` 与 `labels` 文件加载特征与字符串标签；
  - 使用 `LabelBinarizer` 将标签转为 one-hot 向量；
  - 使用 `StratifiedShuffleSplit` 按约 8:1:1 划分训练集、验证集和测试集；
  - 构建一个简单的全连接网络：
    - 输入层：维度与 codes 的特征维度一致；
    - 隐藏层：256 维全连接层；
    - 输出层：3 维全连接层（对应三类国画），接 softmax 得到概率；
  - 损失函数：`tf.nn.softmax_cross_entropy_with_logits`；
  - 优化器：`tf.train.AdamOptimizer()`；
  - 使用 `tf.train.Saver()` 将训练好的模型保存到 `checkpoints/` 目录下（paint.ckpt 等）。


3. 单张图片分类预测
^^^^^^^^^^^^^^^^^^^^
- 命令行脚本：`app.py`
  - 调用 `per_picture()` 从命令行读取图片路径，按照 VGG16 要求加载并预处理为 `(1, 224, 224, 3)`；
  - 在 `get_image_retrieval_result()` 中：
    - 使用 VGG16 对该图片提取 relu6 特征；
    - 恢复训练好的分类网络参数（`ftrain.MODEL_SAVE_PATH` 下的 checkpoint）；
    - 计算 `tf.argmax(ftrain.predicted, 1)` 得到预测类别索引；
    - 将索引映射到 `['flowerBird', 'human', 'landscape']` 并打印结果。


4. 图形界面（分类与检索）
^^^^^^^^^^^^^^^^^^^^^^^^
- 主要脚本：`menu.py` 与 `image_retrieval.py`
  - 使用 PyQt5 创建窗口界面，包含：
    - 文本框输入图片路径；
    - 按钮：
      - 显示图像（预览输入图片）；
      - 点击预测（调用与 `app.py` 类似的流程，显示类别文字“花鸟/人物/山水”）；
      - 检索图像（从数据库中搜索与当前输入图像同类、距离最近的图像并展示）。
  - 分类流程：
    - 从界面文本框获取图片路径；
    - 使用 VGG16 计算特征；
    - 使用训练好的分类网络获取预测类别索引 `pre_value`；
    - 将结果显示在界面标签控件中。
  - 检索流程（以 `menu.py` 为例）：
    - 使用全局的 `pre_value` 作为当前类别；
    - 连接 SQLite 数据库 paint.db，从 `image` 表中选择 label 等于当前类别的所有记录；
    - 逐条读取存储在 BLOB 字段中的特征（通过 `pickle.loads` 反序列化）；
    - 与当前查询图像的特征计算欧式距离，选取距离最小的一条记录；
    - 根据记录中的文件名拼接出本地图像路径，并在界面中显示检索结果图像。


5. 数据库存储结构
^^^^^^^^^^^^^^^^^
- 数据库创建脚本：`create_sql.py`
  - 建表语句：
    - `image(id integer primary key, label varchar(30), imgPath varchar(100), feature BLOB)`
  - 字段含义：
    - id：自增主键；
    - label：类别（字符串，如 "flowerBird"）；
    - imgPath：图像文件名或相对路径；
    - feature：图像特征向量，经 `pickle.dumps` 后存为 BLOB。
- 数据插入脚本：`insert_sql.py`
  - 从 `labels` 文件加载标签顺序；
  - 从 `imageData.pkl` 中加载特征数组（或列表），与标签一一对应；
  - 按类别目录遍历 `paint_photos/` 下的所有图片，构造 (id, label, imgPath, feature) 并写入 image 表。
- 查询与清空辅助脚本：`show_sql.py`
  - `search()`：示例查询指定标签下的所有图像路径；
  - `delete_image()`：删除 image 表中全部记录（谨慎使用）。


6. 爬虫与数据集构建
^^^^^^^^^^^^^^^^^^^^
- `download_image.py`：
  - 使用 `requests` + `BeautifulSoup` 从雅昌艺术网国画频道抓取作品详情页 URL；
  - 从详情页解析出图片真实地址，存入列表；
  - 使用 `urllib.request.urlretrieve()` 将图片下载到本地指定目录，如 `D:/Desktop//国画图片/花鸟X.jpg`；
  - 脚本中通过 `BASE_URL` 常量统一管理站点根地址。
- `download_conf_matrix.py`：
  - 早期版本的爬虫脚本，逻辑与 `download_image.py` 类似，也用于批量下载国画图片。


7. 混淆矩阵与批量评估
^^^^^^^^^^^^^^^^^^^^^^
- `confusion_matrix.py`：
  - 从 `test_photos/` 下的各类别目录收集测试图片路径；
  - 使用 VGG16 + 训练好的分类网络对多张测试图片逐张预测；
  - 将预测结果填入一个 3×3 的矩阵 `res` 中（每行/列代表真实/预测类别），用来查看分类混淆情况；
  - 提供 `show()` 函数按行打印混淆矩阵内容。


四、典型使用顺序
----------------

1. 准备数据集
   - 将原始国画图片按类别放入 `paint_photos/flowerBird`、`paint_photos/human`、`paint_photos/landscape`。
   - 若需要扩充数据，可运行 `download_image.py` 从网站下载更多图片，再手工整理到对应目录。

2. 提取特征
   - 在 `recognizePaint` 目录下运行：
     - `python get_features.py`
   - 生成：
     - `codes.npy`：所有图片的 VGG16 特征；
     - `labels`：每张图片的类别名称。

3. 训练分类器
   - 运行：
     - `python ftrain.py`
   - 在 `checkpoints/` 目录下生成 / 更新 `paint.ckpt` 等模型文件。

4. 初始化数据库（用于检索）
   - 创建数据库表：
     - `python create_sql.py`
   - 准备特征文件（如 `imageData.pkl`，或改造脚本直接使用 `codes.npy`）；
   - 插入记录：
     - `python insert_sql.py`

5. 使用命令行预测
   - 运行：
     - `python app.py`
   - 按提示输入图片路径，查看类别预测结果。

6. 使用图形界面体验分类与检索
   - 运行 GUI：
     - `python menu.py`
     - 或：`python image_retrieval.py`
   - 在界面中输入图片路径：
     - 点击“显示图像”预览图片；
     - 点击“点击预测”查看分类结果；
     - 点击“检索图像”查看数据库中最相似的同类图片。


五、后续改进建议
----------------

- 统一特征存储格式：可考虑只保留 codes.npy，一处生成、多处读取，简化 imageData.pkl 的使用。
- 使用更现代的深度学习框架（如 TensorFlow 2.x 或 PyTorch）重写训练与推理部分。
- 增加配置文件（YAML/JSON）集中管理路径、类别名称、数据库配置等，减少代码硬编码。
- 为脚本增加命令行参数解析（argparse），提升可复用性与可维护性。

