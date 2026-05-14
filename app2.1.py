# 材料疲劳寿命预测小程序（增加试验参数展示版）
import streamlit as st
import torch
import numpy as np
import os

# -------------------------- 模型配置（和你的文件名一致） --------------------------
STRESS_MODEL = "material_fatigue_lifetime_predictor.pth"
STRAIN_MODEL = "strain_control_lifetime_predictor.pth"

# -------------------------- 通用工具类 --------------------------
class MaterialDataScaler:
    def __init__(self, mean, std):
        self.mean_ = np.array(mean, dtype=np.float32)
        self.std_ = np.array(std, dtype=np.float32)
    def transform(self, x):
        x = np.array(x, dtype=np.float32)
        return (x - self.mean_) / self.std_
    def inverse_transform(self, x):
        x = np.array(x, dtype=np.float32)
        return (x * self.std_) + self.mean_

# -------------------------- 动态构建模型（100%匹配训练时的结构） --------------------------
class PredictorModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dims, is_strain_model=True, dropout_rate=0.2):
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        # 核心：区分应力/应变模型的结构差异
        for i, hdim in enumerate(hidden_dims):
            layers.append(torch.nn.Linear(prev_dim, hdim))
            # 应变模型：所有隐藏层后加ReLU
            # 应力模型：仅非最后一层隐藏层后加ReLU
            if is_strain_model or i < len(hidden_dims) - 1:
                layers.append(torch.nn.ReLU())
            prev_dim = hdim
        
        # 应变模型：最后加Dropout层
        if is_strain_model:
            layers.append(torch.nn.Dropout(dropout_rate))
        
        self.hidden_layers = torch.nn.Sequential(*layers)
        self.output_layer = torch.nn.Linear(prev_dim, 1)
        
    def forward(self, x):
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        return x.squeeze()

# -------------------------- 加载模型（动态识别结构） --------------------------
@st.cache_resource
def load_target_model(model_path, is_strain_model):
    device = torch.device("cpu")
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    
    # 从checkpoint读取模型配置，动态构建结构
    model_config = ckpt["model_config"]
    hidden_dims = model_config["hidden_dims"]
    input_dim = model_config["input_dim"]
    dropout_rate = ckpt.get("dropout_rate", 0.2) if is_strain_model else 0.0
    
    # 构建和训练时完全一致的模型
    model = PredictorModel(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        is_strain_model=is_strain_model,
        dropout_rate=dropout_rate
    )
    
    # 加载权重（强制匹配，不兼容会直接报错）
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    
    # 加载标准化器
    x_scaler = MaterialDataScaler(ckpt["X_scaler_mean"], ckpt["X_scaler_std"])
    y_scaler = MaterialDataScaler(ckpt["y_scaler_mean"], ckpt["y_scaler_std"])
    log_x = ckpt["need_log_X"]
    log_y = ckpt["need_log_y"]
    
    # 调试：打印模型配置和标准化参数
    st.info(f"✅ 模型加载成功！\n- 结构：{'应变模型（含Dropout）' if is_strain_model else '应力模型（无Dropout）'}\n- 隐藏层：{hidden_dims}\n- 输入维度：{input_dim}")
    st.info(f"📊 标准化参数：\n- X均值：{x_scaler.mean_}\n- X标准差：{x_scaler.std_}\n- y均值：{y_scaler.mean_}\n- y标准差：{y_scaler.std_}")
    
    return model, x_scaler, y_scaler, log_x, log_y, device

# -------------------------- 预测函数（修复维度+调试输出） --------------------------
def do_predict(model, x_s, y_s, log_x, log_y, inputs, device):
    # 1. 输入预处理
    input_array = np.array(inputs, dtype=np.float32).reshape(1, -1)
    st.info(f"🔍 原始输入：{input_array}")
    
    if log_x:
        input_transformed = np.log1p(input_array)
        st.info(f"🔍 对数变换后：{input_transformed}")
    else:
        input_transformed = input_array.copy()
    
    input_scaled = x_s.transform(input_transformed)
    st.info(f"🔍 标准化后输入：{input_scaled}")
    
    # 2. 模型推理
    input_tensor = torch.FloatTensor(input_scaled).to(device)
    with torch.no_grad():
        pred_scaled = model(input_tensor).cpu().numpy()
    st.info(f"🔍 模型输出（标准化后）：{pred_scaled}")
    
    # 3. 逆变换
    pred_scaled = pred_scaled.reshape(-1, 1)  # 强制2D，避免维度错误
    pred_transformed = y_s.inverse_transform(pred_scaled).flatten()
    st.info(f"🔍 逆标准化后：{pred_transformed}")
    
    if log_y:
        pred_original = np.expm1(pred_transformed)
        st.info(f"🔍 逆对数变换后：{pred_original}")
    else:
        pred_original = pred_transformed
    
    # 4. 约束为正整数
    life = max(round(float(pred_original[0])), 1)
    st.info(f"🔍 最终预测寿命：{life} 次循环")
    return life

# -------------------------- 网页界面（增加试验参数展示） --------------------------
st.set_page_config(page_title="材料疲劳寿命预测", page_icon="🔥")
st.title("🔥 材料疲劳寿命预测小程序")
st.markdown("### 选择控制方式 → 输入参数 → 一键预测")

# ---------- 新增：展示试验基础参数（动态跟随预测类型） ----------
# 材料统一信息
st.markdown("---")
st.subheader("📋 试验背景信息")
st.markdown("**材料**：FGH95 &nbsp;&nbsp;|&nbsp;&nbsp; **温度**：620℃ &nbsp;&nbsp;|&nbsp;&nbsp; **试验类型**：疲劳-蠕变耦合试样试验")

# 根据当前选择的控制方式，显示对应的具体试验参数
predict_type = st.radio("请选择预测类型：", ["应力控制", "应变控制"], horizontal=True)

if predict_type == "应力控制":
    st.info(
        "🔹 **当前试验参数（应力控制）**\n\n"
        "- 应力比 R = 0\n"
        "- 加载速率 0.5 Hz\n"
        "- 波形：三角波/正弦波（典型疲劳-蠕变）"
    )
else:
    st.info(
        "🔹 **当前试验参数（应变控制）**\n\n"
        "- 应变比 R = 0\n"
        "- 加载速率 0.4 %/s\n"
        "- 控制模式：应变闭环"
    )
st.markdown("---")

# 2. 根据选择，动态显示对应参数输入框
if predict_type == "应力控制":
    st.subheader("📌 应力控制参数")
    p1 = st.number_input("应力峰值 (MPa)", value=1200.0, step=10.0)
    p2 = st.number_input("峰值保载时间 (s)", value=0.0, step=1.0)
    inputs = [p1, p2]
    model_path = STRESS_MODEL
    is_strain_model = False
else:
    st.subheader("📌 应变控制参数")
    p1 = st.number_input("总应变范围 (%)", value=0.8, step=0.1)
    p2 = st.number_input("峰值保载时间 (s)", value=0.0, step=1.0)
    inputs = [p1, p2]
    model_path = STRAIN_MODEL
    is_strain_model = True

# 3. 预测按钮
if st.button("🚀 开始预测疲劳寿命", type="primary", use_container_width=True):
    try:
        # 检查模型文件是否存在
        if not os.path.exists(model_path):
            st.error(f"❌ 模型文件不存在！请检查文件路径：{model_path}")
        else:
            with st.spinner("正在加载模型并预测..."):
                # 加载模型（根据类型动态构建结构）
                model, x_s, y_s, log_x, log_y, device = load_target_model(model_path, is_strain_model)
                # 预测（含调试输出）
                life = do_predict(model, x_s, y_s, log_x, log_y, inputs, device)
            st.success(f"✅ 预测完成！疲劳寿命 = {life} 次循环")
    except Exception as e:
        st.error(f"❌ 预测失败：{str(e)}")

st.divider()
st.caption("✅ 小程序使用说明：在上方选择预测类型 → 输入参数 → 点击预测按钮")